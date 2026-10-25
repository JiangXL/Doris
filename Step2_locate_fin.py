# coding: utf-8
"""从海豚原始相片识别并剪裁背鳍图像
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

class FinLocator:
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
                "Identity", "Path", "Aspect", "CropConf", "XMin",
                "XMax", "YMin", "YMax",
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
            (fin_rows, fin_imgs): 元数据行（clearness 留待批量模糊检测后填）
            与对应的背鳍图列表，两者一一对应
        """
        photo_name = os.path.basename(jpg_path)

        fin_rows = []
        fin_imgs = []
        boxes = result.boxes
        photo_h, photo_w = boxes.orig_shape
        # if no fin was found, pass it
        if boxes is None or len(boxes) == 0:
            return fin_rows, fin_imgs
        # check and remove dulicated fin(s)
        keep_indices = filter_duplicate_detections(boxes, self.iou_threshold)
        # check each found fin
        for new_idx, fin_idx in enumerate(keep_indices):
            xyxy = boxes[fin_idx].xyxy
            x0, y0, x1, y1 = [int(i) for i in xyxy[0]]
            conf = float(boxes[fin_idx].conf)
            fin_class_name = result.names[int(boxes[fin_idx].cls)]
            fin_img = result.orig_img[y0:y1, x0:x1, :].copy() 
            prefix, extension = os.path.splitext(photo_name)
            fin_name = f"{prefix}_FIN{new_idx:02d}{extension}"
            fin_path = os.path.join(self.fin_save_dir, fin_name)
            cv2.imwrite(fin_path, fin_img)

            fin_rows.append(
                {   
                    "identity": "",
                    "PhotoName": photo_name,
                    "PhotoWidth": photo_w,
                    "PhotoHeight": photo_h,
                    "Path": f"FIN/{fin_name}",
                    "Aspect": fin_class_name,
                    "CropConf": conf,
                    "XMin": x0,
                    "XMax": x1,
                    "YMin": y0,
                    "YMax": y1,
                }
            )
            fin_imgs.append(fin_img)
        return fin_rows, fin_imgs

    @staticmethod
    def _read_photo(path):
        """读取单张图像（cv2 解码，BGR）。供预取线程调用。"""
        photo = cv2.imread(path)
        return path, photo

    def locate_fin(
            self,
            root_dir: str,
            batch_size: int = 4,
            decode_workers: int = 6,
            prefetch_batches: int = 4,
        ) :
        """批量检测并剪裁背鳍。
        Args:
            root_dir: 存放原始 *.JPG 的目录
            batch_size: YOLO 推理批大小
            decode_workers: JPEG 解码线程数（cv2.imread 释放 GIL，可多线程并行）
            prefetch_batches: 预取批数（内存中最多同时存在 prefetch_batches+1 批解码图）
        """
        self.root_dir = root_dir
        self.meta_path = os.path.join(root_dir, self.meta_dir)
        self.fin_save_dir = os.path.join(root_dir, "FIN")
        os.makedirs(self.fin_save_dir, exist_ok=True)

        self.photo_info_csv = os.path.join(self.meta_path, "PHOTO_METAINFO.csv")
        self.photo_info_df = pd.read_csv(self.photo_info_csv, sep=",", encoding="utf-8")

        # load jpg paths from PHOTO_METAINFO.csv
        jpg_paths = [f"{self.root_dir}/{name}" for name in self.photo_info_df.PhotoName]

        # collect all fin rows, build DataFrame once at the end
        all_rows = []
        batches = [jpg_paths[i:i + batch_size] for i in range(0, len(jpg_paths), batch_size)]
        # 预取：每张图作为独立任务提交到线程池并行解码，
        # 主线程做 GPU 推理/后处理时，后续若干批已在后台解码
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            inflight = deque()
            for j in range(min(prefetch_batches + 1, len(batches))):
                inflight.append([pool.submit(self._read_photo, p) for p in batches[j]])
            for bi in tqdm(range(len(batches)), desc="Locating fin image"):
                pending = inflight.popleft()
                nxt = bi + prefetch_batches + 1
                if nxt < len(batches):
                    inflight.append([pool.submit(self._read_photo, p) for p in batches[nxt]])
                valid_paths, photos = [], []
                for fut in pending:
                    p, photo = fut.result()
                    if photo is None:
                        continue
                    valid_paths.append(p)
                    photos.append(photo)
                if not photos:
                    continue
                results = self.fin_detector(photos, verbose=False)
                batch_rows, batch_imgs = [], []
                for jpg_path, result in zip(valid_paths, results):
                    rows, imgs = self._process_result(jpg_path, result)
                    batch_rows.extend(rows)
                    batch_imgs.extend(imgs)
                # 整批裁剪图一次跑模糊检测，避免逐张单样本 ResNet-101 推理
                if batch_imgs:
                    blur_rets = self.blur_detector.predict_batch(batch_imgs)
                    for row, ret in zip(batch_rows, blur_rets):
                        row["Clearness"] = ret["probabilities"]["clear"]
                all_rows.extend(batch_rows)
        self.fin_info = pd.DataFrame(all_rows)
        # 生成唯一编号
        self.fin_info["identity"] = range(len(self.fin_info))
        self.save_info_to_csv()

    def update_meta_info(self):
        # append ShotID to fin_info TODO: order by PhotoName
        for fin_i in range(len(self.fin_info)):
            PhotoName  = self.fin_info.loc[fin_i, "PhotoName"]
            shot_id = self.photo_info_df.loc[self.photo_info_df["PhotoName"]==PhotoName, 
                                           "ShotID"].values[0]
            self.fin_info.loc[fin_i, "ShotID"] = shot_id
        self.fin_info["ShotID"] = self.fin_info["ShotID"].astype(int)
        # append fin_count to photo_info
        for photo_i in range(len(self.photo_info_df)):
            PhotoName = self.photo_info_df.loc[photo_i, "PhotoName"]
            fin_count = (self.fin_info["PhotoName"] == PhotoName).sum()
            self.photo_info_df.loc[photo_i, "FinCount"] = fin_count.astype(int)
        self.photo_info_df["FinCount"] = self.photo_info_df["FinCount"].astype(int)

    def save_info_to_csv(self):
        self.update_meta_info()
        self.fin_info_csv = os.path.join(self.meta_path, self.fin_meta_name)
        self.fin_info.to_csv(self.fin_info_csv, index=False)
        self.photo_info_df.to_csv(self.photo_info_csv, index=False)
        print(f"已保存元数据: {self.fin_info_csv}, {self.photo_info_csv}")

if __name__ == "__main__":
    import sys
    import torch
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    # 初始化并运行
    locator = FinLocator()
    # 执行批量剪裁

    cpu_num = os.cpu_count()
    if torch.cuda.is_available(): # gpu 
        print("Using GPU")
        batch = 32
        decode = int(cpu_num * 0.8)
        prefetch = 3*cpu_num
    else: # cpu only
        print("Using CPU")
        batch = int(cpu_num*0.5)
        decode = int(cpu_num*0.5)
        prefetch = int(cpu_num*0.5)
    locator.locate_fin(root_dir, batch_size = batch,
            decode_workers = decode,
            prefetch_batches = prefetch)
