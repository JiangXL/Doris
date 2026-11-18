# coding: utf-8
"""从海豚原始相片识别并剪裁背鳍图像
使用训练后的 YOLO 模型检测背鳍（ONNX Runtime 推理）。
"""
import os
import glob
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import pandas as pd
from tqdm import tqdm

from resource_paths import resource_path
from onnx_infer import OnnxFinDetector, is_gpu_available
from check_duplicate_detections import filter_duplicate_detections

class FinLocator:
    """背鳍检测与剪裁器"""
    def __init__(
        self,
        yolo_model_path: str = None,
        iou_threshold: float = 0.5):
        """
        Args:
            yolo_model_path: YOLO ONNX 模型路径; None 用内置默认权重
            iou_threshold: 去重 IOU 阈值
        """
        if yolo_model_path is None:
            yolo_model_path = resource_path(
                os.path.join("models", "fin_yolo_3class_PICWD_best.onnx"))
        self.iou_threshold = iou_threshold
        self.fin_detector = OnnxFinDetector(yolo_model_path)

        # initilize DataFrame
        self.fin_info = pd.DataFrame(
            columns=[
                "Identity", "Path", "CropConf", "XMin",
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
            fin_rows: 元数据行列表 
        """
        photo_name = os.path.basename(jpg_path)

        fin_rows = []
        boxes = result.boxes
        photo_h, photo_w = boxes.orig_shape
        # if no fin was found, pass it
        if boxes is None or len(boxes) == 0:
            return fin_rows
        # check and remove dulicated fin(s)
        keep_indices = filter_duplicate_detections(boxes, self.iou_threshold)
        # check each found fin
        for new_idx, fin_idx in enumerate(keep_indices):
            xyxy = boxes[fin_idx].xyxy
            x0, y0, x1, y1 = [int(i) for i in xyxy[0]]
            conf = float(boxes[fin_idx].conf)
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
                    "CropConf": conf,
                    "XMin": x0,
                    "XMax": x1,
                    "YMin": y0,
                    "YMax": y1,
                }
            )
        return fin_rows

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
            max_inflight_images: int = 128,
        ) :
        """批量检测并剪裁背鳍。
        Args:
            root_dir: 存放原始 *.JPG 的目录
            batch_size: JPEG 解码分组的批大小（仅影响解码并行粒度；
                YOLO 推理批由 onnx_infer.OnnxFinDetector 的 batch 控制，
                显存不足时自动减半，与显存占用无关）
            decode_workers: JPEG 解码线程数（cv2.imread 释放 GIL，可多线程并行）
            prefetch_batches: 预取批数
            max_inflight_images: 内存中最多同时存在的解码图数量上限，
                实际的预取批数会被压缩到满足 (prefetch_batches+2)*batch_size <= max_inflight_images
                （消费当前批时其 future 仍存活，需多留一批的余量）
        """
        # 预取窗口内全是解码后的全分辨率原图，必须按图像数量上限压缩批数，
        # 否则大 batch_size * 大 prefetch_batches 会直接耗尽内存
        prefetch_batches = max(
            0, min(prefetch_batches, max_inflight_images // batch_size - 2))
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
                for jpg_path, result in zip(valid_paths, results):
                    all_rows.extend(self._process_result(jpg_path, result))
        self.fin_info = pd.DataFrame(all_rows)
        # 生成唯一编号
        self.fin_info["identity"] = range(len(self.fin_info))
        self.save_info_to_csv()

    def update_meta_info(self):
        # append ShotID to fin_info (向量化映射，避免逐行扫描 photo_info_df)
        shot_id_map = self.photo_info_df.set_index("PhotoName")["ShotID"]
        self.fin_info["ShotID"] = self.fin_info["PhotoName"].map(shot_id_map)
        self.fin_info["ShotID"] = self.fin_info["ShotID"].astype(int)

    def save_info_to_csv(self):
        self.update_meta_info()
        self.fin_info_csv = os.path.join(self.meta_path, self.fin_meta_name)
        self.fin_info.to_csv(self.fin_info_csv, index=False)
        self.photo_info_df.to_csv(self.photo_info_csv, index=False)
        print(f"已保存元数据: {self.fin_info_csv}, {self.photo_info_csv}")

if __name__ == "__main__":
    import sys
    import multiprocessing
    multiprocessing.freeze_support()
    if len(sys.argv) != 2:
        print("用法: Step2_locate_fin <照片目录>")
        sys.exit(1)
    root_dir = sys.argv[1]
    # 初始化并运行
    locator = FinLocator()
    # 执行批量剪裁

    cpu_num = os.cpu_count()
    if is_gpu_available(): # gpu
        print("Using GPU")
        batch = 4
        decode = int(cpu_num * 0.8)
        prefetch = 4
    else: # cpu only
        print("Using CPU")
        batch = int(cpu_num*0.5)
        decode = int(cpu_num*0.5)
        prefetch = 4
    locator.locate_fin(root_dir, batch_size = batch,
            decode_workers = decode,
            prefetch_batches = prefetch)
