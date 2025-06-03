import cv2
import numpy as np
import pickle
from tqdm import tqdm
from ultralytics import YOLO
import json

model = YOLO("yolo11n.pt")

dat = pickle.load(open('/media/mz/ada2de7f-a8f3-4cdf-94f3-1e8fbd0581dc/home/loki/tps_yolo_data_analysis/tps_rural_v12/vad_mzdata_infos_temporal_train.pkl', 'rb'))

detections = {}

for i in tqdm(range(0, len(dat['infos']), 10)):
    filename = '/media/mz/ada2de7f-a8f3-4cdf-94f3-1e8fbd0581dc/home/loki/tps_yolo_data_analysis/tps_rural_v12/' + dat['infos'][i]['cams']['CAM_FRONT']['data_path']
    
    results = model(filename)
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    unique_ids, counts = np.unique(class_ids, return_counts=True)
    class_frequency = dict(zip(unique_ids, counts))

    for class_id, count in class_frequency.items():
        detections[class_id] = detections.get(class_id, 0) + count

detections = {int(k): int(v) for k, v in detections.items()}
with open('results.json', 'w') as fp:
    json.dump(detections, fp)
