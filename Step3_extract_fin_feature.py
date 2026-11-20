# coding: utf-8
# 背鳍模糊检测和部位分类，并提取深度特征
#   - FinQualityAspectAnalyser: 模糊检测 (Clearness) + 部位分类 (Aspect/Aspect_Conf)
#     + 质量筛选 (Select/Clear), 结果写回 FIN_METAINFO.csv
#   - FinDeepFeatureExtractor:
#     （批量图像 -> MegaDescriptor-L-384 深度特征）
#   - 全部推理经 ONNX Runtime（onnx_infer 模块），运行时不依赖 torch/timm
#   - 输出：  FIN_DEEPFEATURES.npy（特征矩阵）+ FIN_DEEPFEATURES.csv（对应图片路径）

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from resource_paths import resource_path
from onnx_infer import OnnxClassifier, OnnxFeatureExtractor, is_gpu_available

# 质量筛选阈值
CLEARNESS_THRESHOLD = 0.15


class FinQualityAspectAnalyser:
    """鳍图质量与部位分析器: 批量评估清晰度 (Clearness) 与部位 (Aspect),
    并按阈值计算 Clear 质量列, 写回 FIN_METAINFO.csv。"""

    DEFAULT_BLUR_MODEL = resource_path(
        'models/fin_quality_detection_resnet101_final.onnx')
    DEFAULT_ASPECT_MODEL = resource_path(
        'models/fin_aspect_classification_resnet101_final.onnx')

    def __init__(
        self,
        blur_model_path=None,       # None 用内置默认权重
        aspect_model_path=None,     # None 用内置默认权重
        batch_size=64,
        clearness_threshold=CLEARNESS_THRESHOLD,
    ):
        if blur_model_path is None:
            blur_model_path = self.DEFAULT_BLUR_MODEL
        if aspect_model_path is None:
            aspect_model_path = self.DEFAULT_ASPECT_MODEL
        self.batch_size = batch_size
        self.clearness_threshold = clearness_threshold
        self.blur_detector = OnnxClassifier(blur_model_path, class_names=['clear', 'blur'])
        self.aspect_detector = OnnxClassifier(aspect_model_path, class_names=['DL', 'DR', 'MISC'])

    def analyse(self, root_dir):
        """分析鳍图并保存到 FIN_METAINFO.csv。

        Args:
            root_dir: 项目根目录 (含 METAINFO/FIN_METAINFO.csv 与 FIN/ 鳍图)
        """
        csv_path, metainfo = self.load_metainfo(root_dir)
        paths = [os.path.join(root_dir, str(p)) for p in metainfo["Path"]]
        valid_idx = [i for i, p in enumerate(paths) if os.path.isfile(p)]
        if len(valid_idx) < len(paths):
            print(f"警告: {len(paths) - len(valid_idx)} 张鳍图文件缺失, "
                  f"对应行的 Clearness/Aspect 保持空值")
        self._predict_batches(metainfo, paths, valid_idx)
        self._apply_quality_thresholds(metainfo)
        metainfo.to_csv(csv_path, index=False)

    @staticmethod
    def load_metainfo(root_dir):
        """读取 FIN_METAINFO.csv; 兼容旧版写回时残留的无名索引列。"""
        csv_path = os.path.join(root_dir, "METAINFO", "FIN_METAINFO.csv")
        metainfo = pd.read_csv(csv_path)
        if metainfo.columns[0].startswith("Unnamed"):
            metainfo = metainfo.drop(columns=[metainfo.columns[0]])
        return csv_path, metainfo

    def _predict_batches(self, metainfo, paths, valid_idx):
        """分批跑模糊/部位模型, 把结果填进 metainfo 对应行。"""
        # TODO: fin images are read and decoded twice
        for start in tqdm(range(0, len(valid_idx), self.batch_size),
                desc="Calculate Quality and Aspect"):
            idx_batch = valid_idx[start:start + self.batch_size]
            img_batch = [paths[i] for i in idx_batch]
            blur_rets = self.blur_detector.predict_batch(img_batch)
            aspect_rets = self.aspect_detector.predict_batch(img_batch)
            for i, blur_ret, aspect_ret in zip(idx_batch, blur_rets, aspect_rets):
                metainfo.loc[i, "Clearness"] = blur_ret["probabilities"]["clear"]
                metainfo.loc[i, "Aspect"] = aspect_ret["class"]
                metainfo.loc[i, "Aspect_Conf"] = aspect_ret["confidence"]

    def _apply_quality_thresholds(self, metainfo):
        """按清晰度置信度阈值计算 Clear 列。"""
        metainfo["Clear"] = metainfo["Clearness"] > self.clearness_threshold

    def close(self):
        """释放检测器内部的线程池等资源。"""
        for detector in (self.blur_detector, self.aspect_detector):
            close = getattr(detector, "close", None)
            if close is not None:
                close()


class FinDeepFeatureExtractor:
    """Extract deep features from fin images (MegaDescriptor-L-384, ONNX)."""

    DEFAULT_MODEL = resource_path('models/megadescriptor_dolphin_ft.onnx')
    DEFAULT_IMG_SIZE = 384
    DEFAULT_BATCH_SIZE = 8  # ORT 激活内存高于 torch eager，2080 Ti 11GB 实测 batch=32 OOM
    DEFAULT_DEVICE = 'cuda' if is_gpu_available() else 'cpu'

    def __init__(
        self,
        model_path=None,            # None 用内置默认 ONNX 权重
        batch_size=DEFAULT_BATCH_SIZE,
        img_size=DEFAULT_IMG_SIZE,
        csv_name='METAINFO/FIN_METAINFO.csv',
        output_name=None,
    ):
        self.feature_extractor = OnnxFeatureExtractor(
            model_path or self.DEFAULT_MODEL, input_size=img_size)
        self.batch_size = batch_size
        self.img_size = img_size
        self.csv_name = csv_name
        self.output_name = output_name or 'METAINFO/FIN_DEEPFEATURES'
        self.device = self.DEFAULT_DEVICE

    def extract(self, root_dir):
        """
        Extract and save deep features.
         Root directory containing images and metadata CSV.
        """
        csv_path = os.path.join(root_dir, self.csv_name)
        metadata = pd.read_csv(csv_path).reset_index(drop=True)
        items = [os.path.join(root_dir, str(p)) for p in metadata["Path"]]
        features = self.feature_extractor.extract(items, batch_size=self.batch_size)

        out_path = os.path.join(root_dir, self.output_name)
        self._save(out_path, features, metadata)
        return features

    def _save(self, out_path, features, metadata, root_dir=None):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # 通用格式：特征矩阵 + 行对应的图片路径
        np.save(out_path + '.npy', features)
        metadata.to_csv(out_path + '.csv', columns=["PhotoName", "Path"], index=False)
        print("Save", out_path + '.npy', '\n&', out_path + '.csv')


if __name__ == '__main__':
    import sys
    import multiprocessing
    multiprocessing.freeze_support()

    if len(sys.argv) >= 2:
        root_dir = sys.argv[1]
    else:
        print("No root directory is provided")
        sys.exit(1)
    analyser = FinQualityAspectAnalyser()
    try:
        analyser.analyse(root_dir)
    finally:
        analyser.close()
    extractor = FinDeepFeatureExtractor()
    extractor.extract(root_dir)
