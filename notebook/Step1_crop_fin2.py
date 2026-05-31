#!/usr/bin/env python
# coding: utf-8
# # 从海豚原始图像识别并剪裁背鳍
# 使用训练后的 Yolo 模型，需指定 JPG 文件夹.
# 结束后背鳍存在子文件夹 FIN 中，生成元数据 csv.

import os
import glob
import cv2
from tqdm import tqdm
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from blur_detector_torch import BlurDetector
from matplotlib import pyplot as plt

from check_duplicate_detections import filter_duplicate_detections

# Initlize YOLO with best model file
fin_detector = YOLO("models/fin_yolo_best.pt")
# Initlize Blur detector from model file
blur_detector = BlurDetector("models/blur_detection_resnet101_final.pth")

# The path contains camera jpeg files
root_dir = r'/media/filming/2025-白海豚/20240825-JM_02-2/'
dataset_name = root_dir.split("/")[-2]
JPG_paths = glob.glob(os.path.join(root_dir, '*.JPG'))
JPG_paths.sort() # Sort in human readable

# Test one of image
test_img = JPG_paths[20]
result = fin_detector(test_img)
annoted_image = result[0].plot()
annoted_image = Image.fromarray(annoted_image[..., ::-1])  # RGB-order PIL image
plt.imshow(annoted_image)


# Extract and Crop Fin from Orignal Image, One by One
meta_info = pd.DataFrame(columns=["identity", "path", "crop_conf", 
                                  "x_min", "x_max", "y_min", "y_max", 
                                  "orig_img", "orig_img_h", "orig_img_w",
                                  "clearness"])
IOU_THRESHOLD = 0.6 
for JPG_path  in tqdm(JPG_paths):
    ori_img_name = os.path.basename(JPG_path)
    results = fin_detector(JPG_path, verbose=False)
    for result in results:
        boxes = result.boxes        
        if boxes is None or len(boxes) == 0 :
            continue
        # Remove duplicated fin which are detected multiple times
        keep_indices = filter_duplicate_detections(boxes, IOU_THRESHOLD)
        for new_idx, fin_idx in enumerate(keep_indices):
            xyxy = boxes[fin_idx].xyxy
            x0, y0, x1, y1 = [int(i) for i in xyxy[0]]
            conf = float(boxes[fin_idx].conf)
            orig_img_h, orig_img_w = result[0].boxes.orig_shape
            cropped_img = result.orig_img[y0:y1, x0:x1, :]
            # Save cropped fin image
            save_dir = dataset_path + "/FIN/"
            os.makedirs(save_dir, exist_ok=True)
            img_name = ori_img_name[:-4] + "_FIN%02d.JPG"%new_idx
            fin_img_path =  "FIN/" + img_name 
            cv2.imwrite(save_dir + img_name, cropped_img)

            # Predict fin image's clearness from model
            clearness = blur_detector.predict(cropped_img)
            
            # Package data in dictionary as new row
            new_row = {"path": fin_img_path, "crop_conf": conf, 
                       "x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1,
                       "orig_img": ori_img_name, "orig_img_h": orig_img_h, 
                       "orig_img_w": orig_img_w, "clearness": clearness}
            # Append new row to pandas dataframe
            meta_info.loc[len(meta_info)] = new_row
# Generate identity for each fin image
meta_info["identity"] = range(len(meta_info))
# Save METAINFO
meta_info.to_csv(dataset_path + "/METAINFO/FIN_METAINFO.csv")
print("Total fin number:", len(meta_info))

plt.subplot(2, 1, 1)
plt.title("Fin Crop Confidence-"+dataset_name)
plt.plot(meta_info.crop_conf, "*")
plt.subplot(2, 1, 2)
plt.hist(meta_info.crop_conf, bins=256)
plt.savefig(root_dir + "METAINFO/FinCropConfidence.png")
plt.show()

