#!/usr/bin/env python
# coding: utf-8
"""从海豚原始图像识别并剪裁背鳍

使用训练后的 YOLO 模型检测背鳍，使用模糊检测模型评估清晰度。
"""

import os
import sys
import glob
import cv2
from tqdm import tqdm
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from matplotlib import pyplot as plt

from blur_detector_torch import BlurDetector
from check_duplicate_detections import filter_duplicate_detections


class FinCropper:
    """背鳍检测与剪裁器"""

    def __init__(
        self,
        yolo_model_path: str = "models/fin_yolo_best.pt",
        blur_model_path: str = "models/blur_detection_resnet101_final.pth",
        iou_threshold: float = 0.6):
        """
        Args:
            yolo_model_path: YOLO 模型权重路径
            blur_model_path: 模糊检测模型权重路径
            iou_threshold: 去重 IOU 阈值
        """
        self.iou_threshold = iou_threshold
        self.fin_detector = YOLO(yolo_model_path)
        self.blur_detector = BlurDetector(blur_model_path)

    def _detect_and_crop(self, jpg_path: str, output_dir: str):
        """对单张原始图像进行检测、去重、剪裁并保存背鳍图像。
        Args:
            jpg_path: 原始 JPG 图像路径
            output_dir: 背鳍剪裁图保存目录（FIN 子目录会在此创建）
        Returns:
            list[dict]: 该图像检测到的所有背鳍元数据列表
        """
        ori_img_name = os.path.basename(jpg_path)
        results = self.fin_detector(jpg_path, verbose=False)

        rows = []
        fin_save_dir = os.path.join(output_dir, "FIN")
        os.makedirs(fin_save_dir, exist_ok=True)

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            keep_indices = filter_duplicate_detections(boxes, self.iou_threshold)
            orig_img_h, orig_img_w = boxes.orig_shape

            for new_idx, fin_idx in enumerate(keep_indices):
                xyxy = boxes[fin_idx].xyxy
                x0, y0, x1, y1 = [int(i) for i in xyxy[0]]
                conf = float(boxes[fin_idx].conf)

                cropped_img = result.orig_img[y0:y1, x0:x1, :]
                img_name = f"{ori_img_name[:-4]}_FIN{new_idx:02d}.JPG"
                fin_img_path = os.path.join("FIN", img_name)
                cv2.imwrite(os.path.join(fin_save_dir, img_name), cropped_img)

                blur_ret = self.blur_detector.predict(cropped_img)
                clearness = blur_ret['probabilities']['clear']

                rows.append(
                    {
                        "path": fin_img_path,
                        "crop_conf": conf,
                        "x_min": x0,
                        "x_max": x1,
                        "y_min": y0,
                        "y_max": y1,
                        "orig_img": ori_img_name,
                        "orig_img_h": orig_img_h,
                        "orig_img_w": orig_img_w,
                        "clearness": clearness,
                    }
                )
        return rows

    def crop(
            self,
            root_dir: str,
            output_dir: str = None,
            save_meta: bool = True,
            save_plot: bool = True,
        ) -> pd.DataFrame:
        """批量检测并剪裁背鳍。
        Args:
            root_dir: 存放原始 *.JPG 的目录（可带末尾斜杠）
            output_dir: 结果输出目录，默认与 root_dir 相同
            save_meta: 是否保存 METAINFO/FIN_METAINFO.csv
            save_plot: 是否保存置信度分布图 METAINFO/FinCropConfidence.png

        Returns:
            pd.DataFrame: 包含所有背鳍元数据的 DataFrame
        """
        root_dir = root_dir.rstrip(os.sep)
        if output_dir is None:
            output_dir = root_dir.rstrip(os.sep) +  "_GM"
            print("Automatic set output dir to", output_dir)
        else:
            output_dir = output_dir.rstrip(os.sep)
        dataset_name = os.path.basename(root_dir)
        jpg_paths = sorted(glob.glob(
            os.path.join(glob.escape(root_dir), "*.JPG")))

        if not jpg_paths:
            print(f"警告: 在 {root_dir} 中未找到 *.JPG 文件")
            return pd.DataFrame()

        meta_info = pd.DataFrame(
            columns=[
                "identity",
                "path",
                "crop_conf",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "orig_img",
                "orig_img_h",
                "orig_img_w",
                "clearness",
            ]
        )

        for jpg_path in tqdm(jpg_paths, desc="Cropping fins"):
            rows = self._detect_and_crop(jpg_path, output_dir)
            # append new found fin metainfo to final row of table
            for row in rows:
                meta_info.loc[len(meta_info)] = row

        # 生成唯一编号
        meta_info["identity"] = range(len(meta_info))

        # 保存元数据
        if save_meta:
            meta_dir = os.path.join(output_dir, "METAINFO")
            os.makedirs(meta_dir, exist_ok=True)
            meta_path = os.path.join(meta_dir, "FIN_METAINFO.csv")
            meta_info.to_csv(meta_path, index=False)
            print(f"元数据已保存: {meta_path}")

        # 保存可视化
        if save_plot:
            meta_dir = os.path.join(output_dir, "METAINFO")
            os.makedirs(meta_dir, exist_ok=True)
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.title(f"Fin Crop Confidence-{dataset_name}")
            plt.plot(meta_info["crop_conf"], "*")
            plt.subplot(2, 1, 2)
            plt.hist(meta_info["crop_conf"], bins=256)
            plot_path = os.path.join(meta_dir, "FinCropConfidence.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"分布图已保存: {plot_path}")

        print(f"Total fin number: {len(meta_info)}")
        return meta_info

    def preview(self, image_path: str):
        """对单张图像进行背鳍检测并返回标注后的 PIL Image（用于快速预览）。

        Args:
            image_path: 图像文件路径

        Returns:
            PIL.Image.Image: 带检测框的图像
        """
        result = self.fin_detector(image_path)[0]
        annotated = result.plot()
        return Image.fromarray(annotated[..., ::-1])  # BGR -> RGB


# ============================================
# 直接运行示例
# ============================================

if __name__ == "__main__":
    # 配置路径
    #root_dir = r"/media/filming/2025-白海豚/20240825-JM_02-3/"
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    else: 
        print("No root directory is provided")

    # 初始化并运行
    cropper = FinCropper()

    # 可选：预览单张图像
    # jpg_paths = sorted(glob.glob(os.path.join(root_dir, '*.JPG')))
    # preview_img = cropper.preview(jpg_paths[20])
    # preview_img.show()

    # 执行批量剪裁
    meta_df = cropper.crop(root_dir)
