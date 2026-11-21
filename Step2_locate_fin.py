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
        iou_threshold: float = 0.5,
        intra_op_threads: int = None):
        """
        Args:
            yolo_model_path: YOLO ONNX 模型路径; None 用内置默认权重
            iou_threshold: 去重 IOU 阈值
            intra_op_threads: ORT CPU intra-op 线程数（仅纯 CPU 推理时
                设置，给解码/后处理线程留核；CUDA 路径传 None）
        """
        if yolo_model_path is None:
            yolo_model_path = resource_path(
                os.path.join("models", "fin_yolo_3class_PICWD_best.onnx"))
        self.iou_threshold = iou_threshold
        self.fin_detector = OnnxFinDetector(
            yolo_model_path, intra_op_threads=intra_op_threads)

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

    def _read_photo(self, path):
        """读取单张图像（cv2 解码，BGR）并做 letterbox，供预取线程调用。
        返回 (path, photo, letterboxed)；解码失败返回 (path, None, None)。"""
        photo = cv2.imread(path)
        if photo is None:
            return path, None, None
        return path, photo, self.fin_detector.letterbox([photo])[0]

    def locate_fin(
            self,
            root_dir: str,
            decode_workers: int = None,
            prefetch_batches: int = 1,
            max_inflight_images: int = 64,
        ) :
        """批量检测并剪裁背鳍（二级流水线：解码预取 ∥ 推理+后处理）。

        解码线程池并行 cv2.imread + letterbox 并预取后续批次；主线程攒满
        一个推理批后顺序执行 GPU 推理与 NMS/裁剪/写盘。推理批大小由
        onnx_infer 自动适配显存（OOM 降批并记住），无需调用方指定；
        输出天然按图顺序，无需排序。

        Args:
            root_dir: 存放原始 *.JPG 的目录
            decode_workers: JPEG 解码线程数（cv2.imread 释放 GIL，可多线程并行；
                None 默认吃满全部核——解码是端到端瓶颈）
            prefetch_batches: 解码预取批数
            max_inflight_images: 解码预取窗口内最多同时存在的原图数量上限，
                预取批数会被压缩到 (prefetch_batches+2)*分组 <= 上限
        """
        # 预取/消费分组与推理批大小对齐（detector.batch，初始 32，OOM 后自适应）
        group = self.fin_detector.batch
        # 预取窗口内全是解码后的全分辨率原图，必须按图像数量上限压缩批数，
        # 否则大分组 * 大 prefetch_batches 会直接耗尽内存
        prefetch_batches = max(
            0, min(prefetch_batches, max_inflight_images // group - 2))
        if decode_workers is None:
            # 解码是端到端瓶颈，默认吃满全部核
            decode_workers = os.cpu_count() or 4
        self.root_dir = root_dir
        self.meta_path = os.path.join(root_dir, self.meta_dir)
        self.fin_save_dir = os.path.join(root_dir, "FIN")
        os.makedirs(self.fin_save_dir, exist_ok=True)

        self.photo_info_csv = os.path.join(self.meta_path, "PHOTO_METAINFO.csv")
        self.photo_info_df = pd.read_csv(self.photo_info_csv, sep=",", encoding="utf-8")

        # load jpg paths from PHOTO_METAINFO.csv
        jpg_paths = [f"{self.root_dir}/{name}" for name in self.photo_info_df.PhotoName]

        batches = [jpg_paths[i:i + group] for i in range(0, len(jpg_paths), group)]
        accumulator = []
        all_rows = []

        def flush(n=None):
            """把 accumulator 前 n 张（默认全部）推理并后处理，行按图顺序追加。"""
            nonlocal accumulator
            if n is None or n > len(accumulator):
                n = len(accumulator)
            if n == 0:
                return
            chunk, accumulator = accumulator[:n], accumulator[n:]
            outputs = self.fin_detector.infer([c[2] for c in chunk])
            results = self.fin_detector.build_results([c[1] for c in chunk], outputs)
            for (jpg_path, _, _), result in zip(chunk, results):
                all_rows.extend(self._process_result(jpg_path, result))

        with ThreadPoolExecutor(max_workers=decode_workers) as decode_pool:
            # 预取：每张图作为独立任务提交到线程池并行解码，
            # 主线程做推理/后处理时，后续若干批已在后台解码
            inflight = deque()
            for j in range(min(prefetch_batches + 1, len(batches))):
                inflight.append([decode_pool.submit(self._read_photo, p) for p in batches[j]])
            pbar = tqdm(total=len(jpg_paths), desc="Locating fin image")
            for bi in range(len(batches)):
                pending = inflight.popleft()
                nxt = bi + prefetch_batches + 1
                if nxt < len(batches):
                    inflight.append([decode_pool.submit(self._read_photo, p) for p in batches[nxt]])
                for fut in pending:
                    p, photo, letterboxed = fut.result()
                    if photo is None:
                        continue
                    accumulator.append((p, photo, letterboxed))
                pbar.update(len(pending))
                while len(accumulator) >= self.fin_detector.batch:
                    flush(self.fin_detector.batch)
            flush()
            pbar.close()
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
    cpu_num = os.cpu_count()
    if is_gpu_available(): # gpu
        print("Using GPU")
        # 解码是瓶颈（实测 16 线程才打满），推理/后处理开销小，
        # 二级流水线即可，解码吃满全部核
        decode = cpu_num
        prefetch = 1
        intra_op = None
    else: # cpu only
        print("Using CPU")
        # 收敛线程预算：decode + ORT intra-op ≈ 核数，避免超订
        decode = int(cpu_num * 0.25)
        prefetch = 1
        intra_op = max(1, cpu_num - decode)
    locator = FinLocator(intra_op_threads=intra_op)
    # 执行批量剪裁
    locator.locate_fin(root_dir,
            decode_workers = decode,
            prefetch_batches = prefetch)
