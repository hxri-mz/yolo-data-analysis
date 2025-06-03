import pandas as pd
import os
import json
import pickle
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import cv2

model = YOLO("yolo11x.pt")
dat = pickle.load(open('/media/mz/ada2de7f-a8f3-4cdf-94f3-1e8fbd0581dc/home/loki/tps_yolo_data_analysis/tps_rural_v12/vad_mzdata_infos_temporal_train.pkl', 'rb'))

detections = {}
per_image_data = []

def overlay(results, img, rectangle_thickness=2, text_thickness=1):
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img

for i in tqdm(range(0, len(dat['infos']), 10)):
    info = dat['infos'][i]
    filename = os.path.join('/media/mz/ada2de7f-a8f3-4cdf-94f3-1e8fbd0581dc/home/loki/tps_yolo_data_analysis/tps_rural_v12', info['cams']['CAM_FRONT']['data_path'])

    results = model(filename)
    # img = cv2.imread(filename)
    # overlayed = overlay(results, img)
    # cv2.imshow("Image", overlayed)
    # cv2.waitKey(0)
    # cv2.imwrite(f"/home/mz/yolo_data_analysis/test/{i}.png", overlayed)
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    unique_ids, counts = np.unique(class_ids, return_counts=True)
    class_frequency = dict(zip(unique_ids, counts))

    for class_id, count in class_frequency.items():
        detections[class_id] = detections.get(class_id, 0) + count

    item = {"file": info['cams']['CAM_FRONT']['data_path']}
    for class_id, count in class_frequency.items():
        item[int(class_id)] = int(count)
    per_image_data.append(item)

detections = {int(k): int(v) for k, v in detections.items()}
with open('results.json', 'w') as fp:
    json.dump(detections, fp, indent=2)

df = pd.DataFrame(per_image_data)
df = df.fillna(0)
numeric_cols = df.columns.drop('file')
df[numeric_cols] = df[numeric_cols].astype(int)
df.to_csv('class_counts.csv', index=False)
