#!/usr/bin/env python
# coding: utf-8
"""从海豚原始图像识别并剪裁背鳍
使用训练后的 YOLO 模型检测背鳍，使用模糊检测模型评估清晰度。
"""
import os
import glob
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from matplotlib import pyplot as plt

from blur_detector_torch import BlurDetector
from check_duplicate_detections import filter_duplicate_detections

class FinCropper:
    """背鳍检测与剪裁器"""
    def __init__(
        self,
        yolo_model_path: str = "models/fin_yolo_3class_PICWD_best.pt",
        blur_model_path: str = "models/blur_detection_resnet101_final.pth",
        iou_threshold: float = 0.5):
        """
        Args:
            yolo_model_path: YOLO 模型权重路径
            blur_model_path: 模糊检测模型权重路径
            iou_threshold: 去重 IOU 阈值
        """
        self.iou_threshold = iou_threshold
        self.fin_detector = YOLO(yolo_model_path, task='detect')
        self.blur_detector = BlurDetector(blur_model_path)

        # initilize DataFrame
        self.fin_info = pd.DataFrame(
            columns=[
                "identity", "path", "class", "crop_conf", "x_min",
                "x_max", "y_min", "y_max",
            ]
        )
        self.root_dir = "./"
        self.meta_dir = "METAINFO"
        self.fin_meta_name = "FIN_METAINFO.csv"
        self.fin_save_dir = ""
               
    def _process_result(self, jpg_path: str, result):
        """对单张原始图像的检测结果进行去重、剪裁并保存背鳍图像。
        Args:
            jpg_path: 原始 JPG 图像路径
            result: ultralytics 对该图像的检测结果
        Returns:
            (fin_rows, fin_crops): 元数据行（clearness 留待批量模糊检测后填）
            与对应的裁剪图列表，两者一一对应
        """
        ori_img_name = os.path.basename(jpg_path)

        fin_rows = []
        fin_crops = []
        boxes = result.boxes
        orig_img_h, orig_img_w = boxes.orig_shape
        # if no fin was found, pass it
        if boxes is None or len(boxes) == 0:
            return fin_rows, fin_crops
        # check and remove dulicated fin(s)
        keep_indices = filter_duplicate_detections(boxes, self.iou_threshold)
        # check each found fin
        for new_idx, fin_idx in enumerate(keep_indices):
            xyxy = boxes[fin_idx].xyxy
            x0, y0, x1, y1 = [int(i) for i in xyxy[0]]
            conf = float(boxes[fin_idx].conf)
            fin_class_name = result.names[int(boxes[fin_idx].cls)]
            cropped_img = result.orig_img[y0:y1, x0:x1, :].copy() 
            fin_name = f"{ori_img_name[:-4]}_FIN{new_idx:02d}.JPG"
            fin_path = os.path.join(self.fin_save_dir, fin_name)
            cv2.imwrite(fin_path, cropped_img)

            fin_rows.append(
                {   
                    "identity": "",
                    "orig_img_name": ori_img_name,
                    "orig_img_w": orig_img_w,
                    "orig_img_h": orig_img_h,
                    "path": f"FIN/{fin_name}",
                    "class": fin_class_name,
                    "crop_conf": conf,
                    "x_min": x0,
                    "x_max": x1,
                    "y_min": y0,
                    "y_max": y1,
                }
            )
            fin_crops.append(cropped_img)
        return fin_rows, fin_crops

    @staticmethod
    def _read_image(path):
        """读取单张图像（cv2 解码，BGR）。供预取线程调用。"""
        img = cv2.imread(path)
        return path, img

    def crop(
            self,
            root_dir: str,
            batch_size: int = 1,
            decode_workers: int = 6,
            prefetch_batches: int = 2,
        ) :
        """批量检测并剪裁背鳍。
        Args:
            root_dir: 存放原始 *.JPG 的目录（可带末尾斜杠）
            batch_size: YOLO 推理批大小
            decode_workers: JPEG 解码线程数（cv2.imread 释放 GIL，可多线程并行）
            prefetch_batches: 预取批数（内存中最多同时存在 prefetch_batches+1 批解码图）
        """
        # set root_dir
        self.root_dir = root_dir
        self.meta_path = os.path.join(self.root_dir, self.meta_dir)
        self.fin_save_dir = os.path.join(self.root_dir, "FIN")
        os.makedirs(self.fin_save_dir, exist_ok=True)
        self.img_info_csv = os.path.join(self.meta_path, "IMAGE_METAINFO.csv")
        self.img_info_df = pd.read_csv(self.img_info_csv, sep=",", encoding="utf-8")
        dataset_name = os.path.basename(root_dir)

        # load jpg paths from IMAGE_METAINFO.csv
        jpg_paths = [f"{self.root_dir}/{name}" for name in self.img_info_df.orig_img_name]

        # collect all fin rows, build DataFrame once at the end
        all_rows = []
        batches = [jpg_paths[i:i + batch_size] for i in range(0, len(jpg_paths), batch_size)]
        # 预取：每张图作为独立任务提交到线程池并行解码，
        # 主线程做 GPU 推理/后处理时，后续若干批已在后台解码
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            inflight = deque()
            for j in range(min(prefetch_batches + 1, len(batches))):
                inflight.append([pool.submit(self._read_image, p) for p in batches[j]])
            for bi in tqdm(range(len(batches)), desc="Cropping fin image"):
                pending = inflight.popleft()
                nxt = bi + prefetch_batches + 1
                if nxt < len(batches):
                    inflight.append([pool.submit(self._read_image, p) for p in batches[nxt]])
                valid_paths, imgs = [], []
                for fut in pending:
                    p, img = fut.result()
                    if img is None:
                        continue
                    valid_paths.append(p)
                    imgs.append(img)
                if not imgs:
                    continue
                results = self.fin_detector(imgs, verbose=False)
                batch_rows, batch_crops = [], []
                for jpg_path, result in zip(valid_paths, results):
                    rows, crops = self._process_result(jpg_path, result)
                    batch_rows.extend(rows)
                    batch_crops.extend(crops)
                # 整批裁剪图一次跑模糊检测，避免逐张单样本 ResNet-101 推理
                if batch_crops:
                    blur_rets = self.blur_detector.predict_batch(batch_crops)
                    for row, ret in zip(batch_rows, blur_rets):
                        row["clearness"] = ret["probabilities"]["clear"]
                all_rows.extend(batch_rows)
        self.fin_info = pd.DataFrame(all_rows)
        # 生成唯一编号
        self.fin_info["identity"] = range(len(self.fin_info))
        
    def update_meta_info(self):
        # append shot_id to fin_info
        for fin_i in range(len(self.fin_info)):
            orig_img_name  = self.fin_info.loc[fin_i, "orig_img_name"]
            shot_id = self.img_info_df.loc[self.img_info_df["orig_img_name"]==orig_img_name, "shot_id"].values[0]
            self.fin_info.loc[fin_i, "shot_id"] = shot_id
        self.fin_info["shot_id"] = self.fin_info["shot_id"].astype(int)
        # append fin_count to img_info
        for img_i in range(len(self.img_info_df)):
            orig_img_name = self.img_info_df.loc[img_i, "orig_img_name"]
            fin_count = (self.fin_info["orig_img_name"] == orig_img_name).sum()
            self.img_info_df.loc[img_i, "fin_count"] = fin_count.astype(int)
        self.img_info_df["fin_count"] = self.img_info_df["fin_count"].astype(int)

    def save_info_to_csv(self):
        self.update_meta_info()
        self.fin_info_csv = os.path.join(self.meta_path, self.fin_meta_name)
        self.fin_info.to_csv(self.fin_info_csv, index=False)
        self.img_info_df.to_csv(self.img_info_csv, index=False)
        print(f"已保存元数据: {self.fin_info_csv}, {self.img_info_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    # 初始化并运行
    cropper = FinCropper()
    # 执行批量剪裁
    cropper.crop(root_dir)
    #TODO: automatic set batch size based on computer resource
    cropper.save_info_to_csv()
