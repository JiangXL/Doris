#!/usr/bin/env python
# coding: utf-8
"""从海豚原始图像识别并剪裁背鳍
使用训练后的 YOLO 模型检测背鳍，使用模糊检测模型评估清晰度。
"""
import os
import glob
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
        self.root_dir = ""
        self.meta_dir = "METAINFO"
        self.fin_meta_name = "FIN_METAINFO.csv"
       
    def _detect_and_crop(self, jpg_path: str, output_dir: str):
        """对单张原始图像进行检测、去重、剪裁并保存背鳍图像。
        Args:
            jpg_path: 原始 JPG 图像路径
            output_dir: 背鳍剪裁图保存目录（FIN 子目录会在此创建）
        """
        ori_img_name = os.path.basename(jpg_path)
        fin_save_dir = os.path.join(output_dir, "FIN")
        os.makedirs(fin_save_dir, exist_ok=True)

        fin_rows = []
        # detect fin in image
        results = self.fin_detector(jpg_path, verbose=False)
        for result in results:
            boxes = result.boxes
            # if no fin was found, pass it
            if boxes is None or len(boxes) == 0:
                continue
            # check and remove dulicated fin(s)
            keep_indices = filter_duplicate_detections(boxes, self.iou_threshold)
            # check each found fin
            for new_idx, fin_idx in enumerate(keep_indices):
                xyxy = boxes[fin_idx].xyxy
                x0, y0, x1, y1 = [int(i) for i in xyxy[0]]
                conf = float(boxes[fin_idx].conf)
                fin_class_name = result.names[int(boxes[fin_idx].cls)]
                cropped_img = result.orig_img[y0:y1, x0:x1, :]
                fin_name = f"{ori_img_name[:-4]}_FIN{new_idx:02d}_{fin_class_name}.JPG"
                fin_path = os.path.join(fin_save_dir, fin_name)
                cv2.imwrite(fin_path, cropped_img)

                blur_ret = self.blur_detector.predict(cropped_img)
                clearness = blur_ret['probabilities']['clear']
                fin_rows.append(
                    {
                        "path": f"FIN/{fin_name}",
                        "class": fin_class_name,
                        "crop_conf": conf,
                        "x_min": x0,
                        "x_max": x1,
                        "y_min": y0,
                        "y_max": y1,
                        "orig_img": ori_img_name,
                        "clearness": clearness,
                    }
                )
        return fin_rows

    def crop(
            self,
            root_dir: str,
            output_dir: str = None
        ) -> pd.DataFrame:
        """批量检测并剪裁背鳍。
        Args:
            root_dir: 存放原始 *.JPG 的目录（可带末尾斜杠）
            output_dir: 结果输出目录，默认与 root_dir 相同
        """
        # set root_dir
        root_dir = root_dir.rstrip(os.sep)
        if output_dir is None:
            output_dir = root_dir
            print("Automatic set output dir to", output_dir)
        else:
            output_dir = output_dir.rstrip(os.sep)
        dataset_name = os.path.basename(root_dir)
       
        self.root_dir = root_dir
        self.img_info_csv = os.path.join(self.root_dir, self.meta_dir, "IMAGE_METAINFO.csv")
        self.img_info_df = pd.read_csv(self.img_info_csv, sep=",", encoding="utf-8")

        # load jpg paths from IMAGE_METAINFO.csv
        jpg_paths = [f"{self.root_dir}/{name}" for name in self.img_info_df.orig_name]

        for jpg_path in tqdm(jpg_paths, desc="Cropping fin image"):
            fin_rows = self._detect_and_crop(jpg_path, output_dir)
            if fin_rows is not None:
                # append new found fin metainfo to final row of table
                for row in  fin_rows:
                    self.fin_info.loc[len(self.fin_info)] = row
        # 生成唯一编号
        self.fin_info["identity"] = range(len(self.fin_info))
        
    def save_info_to_csv(self):
        # 保存元数据
        meta_dir = os.path.join(output_dir, "METAINFO")
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, self.fin_meta_name)
        self.fin_info.to_csv(meta_path, index=False)
        print(f"元数据已保存: {meta_path}")

    def preview(self, image_path: str):
        """对单张图像进行背鳍检测并返回标注后的 PIL Image（用于快速预览）。
        Args:
            image_path: 图像文件路径

        Returns:
            PIL.Image.Image: 带检测框的图像
        """
        from PIL import Image
        result = self.fin_detector(image_path)[0]
        annotated = result.plot()
        return Image.fromarray(annotated[..., ::-1])  # BGR -> RGB

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    # 初始化并运行
    cropper = FinCropper()
    # 执行批量剪裁
    cropper.crop(root_dir)
    cropper.save_info_to_csv()

    # 可选：预览单张图像
    # jpg_paths = sorted(glob.glob(os.path.join(root_dir, '*.JPG')))
    # preview_img = cropper.preview(jpg_paths[20])
    # preview_img.show()
