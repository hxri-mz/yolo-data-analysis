import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_entity_distribution(csv_file):
    # Read the CSV file
    print("Reading CSV data...")
    df = pd.read_csv(csv_file)
    
    # Define entities to analyze
    entities = ['Car', 'Motorcycle', 'Autorickshaw', 'Pedestrian']
    
    # Create a unique identifier for each scene (combination of date+file)
    df['scene'] = df['date'] + '_' + df['file']
    
    # Calculate the total number of entities in each frame
    print("Processing data...")
    for entity in tqdm(entities, desc="Calculating entity counts"):
        if entity not in df.columns:
            print(f"Warning: Entity '{entity}' not found in the CSV file.")
            return
    
    # Calculate total entities per frame
    df['total_entities'] = df[entities].sum(axis=1)
    
    # Get the count of frames for each total entity count
    total_counts = df['total_entities'].value_counts().sort_index()
    
    # Prepare data for stacked bar chart
    entity_counts = {}
    
    print("Preparing plot data...")
    for total in tqdm(total_counts.index, desc="Analyzing distributions"):
        frames_with_total = df[df['total_entities'] == total]
        entity_counts[total] = {entity: frames_with_total[entity].value_counts().to_dict() 
                               for entity in entities}
    
    # Prepare stacked bar data
    x_values = list(total_counts.index)
    stacked_data = {entity: [] for entity in entities}
    todat = []
    for total in x_values:
        tsum = 0
        for entity in entities:
            # Count frames where this entity appears with specific counts
            entity_distribution = entity_counts[total].get(entity, {})
            # Sum the number of frames where this entity has count > 0
            count = sum(frames * count for count, frames in entity_distribution.items())
            stacked_data[entity].append(count)
            tsum += count
        todat.append(tsum)
    
    # Create the stacked bar chart
    print("Creating plot...")
    plt.figure(figsize=(12, 8))
    
    # Set up the bottom for stacking
    bottom = np.zeros(len(x_values))
    
    # Color map for entities
    colors = {'Car': 'skyblue', 'Motorcycle': 'skyblue', 'Autorickshaw': 'skyblue', 'Pedestrian': 'skyblue'}
    col2 = {'Car': 'orange', 'Motorcycle': 'green', 'Autorickshaw': 'blue', 'Pedestrian': 'red'}
    edg = {'Car': [-2, 5], 'Motorcycle': [5, 15], 'Autorickshaw': [15, 25], 'Pedestrian': [25, 38]}
    for entity in entities:
        plt.axvspan(edg[entity][0], edg[entity][1], 
                    alpha=0.2, color=col2[entity])
    plt.bar(x_values, todat, bottom=bottom, label=entity, color=colors[entity], edgecolor='black', alpha=1.0)
    bottom += np.array(todat)
        
    # df['total_entities'] = df[entities].sum(axis=1)
    # total_counts = df['total_entities'].value_counts()
    # plt.bar(total_counts.index, total_counts.values, color='skyblue', edgecolor='black')

    
    plt.xlabel('Number of obstacles in frame')
    plt.ylabel('Number of frames')
    plt.xlim([-2, 38])
    plt.title('Distribution of obstacles across frames')
    # plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.xticks(x_values)
    plt.yticks([])
    
    # Add value labels on top of each stacked bar
    # for i, total in enumerate(x_values):
    #     plt.text(i, total_counts[total] + 0.5, str(total_counts[total]), 
    #              ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('entity_distribution.png')
    plt.show()
    
    print(f"Plot saved as 'entity_distribution.png'")

if __name__ == "__main__":
    plot_entity_distribution("clscc.csv")