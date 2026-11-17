# coding: utf-8
# 背鳍模糊检测和部位分类，并提取深度特征
#   - FinQualityAnalyser: 模糊检测 (Clearness) + 部位分类 (Aspect/Aspect_Conf)
#     + 质量筛选 (Select/Clear), 结果写回 FIN_METAINFO.csv
#   - FinDeepFeatureExtractor:
#     （ImageDataset = 普通 Dataset；DeepFeatures = 批量前向传播）
#   - 支持任意 timm 模型（默认 MegaDescriptor-L-384，用本地权重完全离线加载），
#     可加载本地微调权重
#   - 输出：  FIN_DEEPFEATURES.npy（特征矩阵）+ FIN_DEEPFEATURES.csv（对应图片路径）

import os

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from resource_paths import resource_path
from fin_quality_detector_torch import BlurDetector
from fin_aspect_classifier_torch import FinAspectClassifier

# 质量筛选阈值
CLEARNESS_THRESHOLD = 0.15


class FinQualityAspectAnalyser:
    """鳍图质量与部位分析器: 批量评估清晰度 (Clearness) 与部位 (Aspect),
    并按阈值计算 Clear 质量列, 写回 FIN_METAINFO.csv。"""

    DEFAULT_BLUR_MODEL = resource_path(
        'models/fin_quality_detection_resnet101_final.pth')
    DEFAULT_ASPECT_MODEL = resource_path(
        'models/fin_aspect_classification_resnet101_final.pth')

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
        self.blur_detector = BlurDetector(blur_model_path)
        self.aspect_detector = FinAspectClassifier(aspect_model_path)

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


class FinImageDataset(Dataset):
    """Minimal image dataset driven by a metadata dataframe."""

    def __init__(self, root_dir, metadata, transform, col_path="Path"):
        self.root_dir = root_dir
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform
        self.col_path = col_path

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        rel_path = self.metadata.loc[idx, self.col_path]
        image = Image.open(os.path.join(self.root_dir, rel_path)).convert("RGB")
        return self.transform(image)


class FinDeepFeatureExtractor:
    """Extract deep features from fin images using any timm model."""

    # MegaDescriptor-L-384 即 timm 的 
    # swin_large_patch4_window12_384（num_classes=0），
    # 用本地权重加载即可完全离线；
    # 如需联网版可传 model_name='hf-hub:BVRA/MegaDescriptor-L-384'
    DEFAULT_MODEL = 'swin_large_patch4_window12_384'
    DEFAULT_PRETRAINED = resource_path('models/megadescriptor_L_384.pth')
    # MegaDescriptor 官方权重（本地）
    DEFAULT_CHECKPOINT = resource_path('models/megadescriptor_dolphin_ft.pth')
    DEFAULT_IMG_SIZE = 384
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_NUM_WORKERS = 4
    DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(
        self,
        model_name=DEFAULT_MODEL,   # any timm model identifier
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        device=DEFAULT_DEVICE,      # 'cuda' or 'cpu'
        img_size=DEFAULT_IMG_SIZE,
        csv_name='METAINFO/FIN_METAINFO.csv',
        output_name=None,           
        # default depends on checkpoint, see below
        checkpoint= DEFAULT_CHECKPOINT,         
        # local fine-tuned weights (.pth); None = 用预训练权重
        pretrained_weights=DEFAULT_PRETRAINED,  
        # 本地预训练权重；hf-hub 模型时忽略（从 HF 加载）
    ):
        self.model_name = model_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.csv_name = csv_name
        self.checkpoint = checkpoint
        self.pretrained_weights = pretrained_weights
        self.output_name = 'METAINFO/FIN_DEEPFEATURES'

        use_hf = self.model_name.startswith('hf-hub:')
        # MegaDescriptor 这类特征提取模型 num_classes=0；
        # hf-hub 模型由 config.json 自带该配置
        create_kwargs = {} if use_hf else {'num_classes': 0}

        if self.checkpoint:
            self.model = timm.create_model(self.model_name, pretrained=False, **create_kwargs)
            self._load_state_dict(self.model, self.checkpoint)
        elif use_hf:
            self.model = timm.create_model(self.model_name, pretrained=True)
        else:
            self.model = timm.create_model(self.model_name, pretrained=False, **create_kwargs)
            self._load_state_dict(self.model, self.pretrained_weights)
        self.model.eval().to(self.device)

        self.transform = T.Compose([
            T.Resize([self.img_size, self.img_size]),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    @staticmethod
    def _load_state_dict(model, path):
        # 训练存档（含 model/optimizer/epoch 等键）取其中的 model 权重
        obj = torch.load(path, map_location='cpu')
        sd = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
        try:
            model.load_state_dict(sd)
        except RuntimeError:
            # 旧版 timm Swin 存档：downsample 键移位、去掉 attn_mask 等 buffer
            from timm.models.swin_transformer import checkpoint_filter_fn
            model.load_state_dict(checkpoint_filter_fn(sd, model))

    @torch.no_grad()
    def extract(self, root_dir):  
        """
        Extract and save deep features.
         Root directory containing images and metadata CSV.
        """
        csv_path = os.path.join(root_dir, self.csv_name)
        #metadata = pd.read_csv(csv_path).query('select==True').reset_index(drop=True)
        metadata = pd.read_csv(csv_path).reset_index(drop=True)

        dataset = FinImageDataset(root_dir, metadata, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        features = []
        for images in tqdm(loader,
                           desc="Extract DeepFeatures"):
            feats = self.model(images.to(self.device)).cpu()
            features.append(feats)
        features = torch.cat(features).numpy()

        out_path = os.path.join(root_dir, self.output_name)
        self._save(out_path, features, metadata, root_dir)
        return features

    def _save(self, out_path, features, metadata, root_dir):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # 通用格式：特征矩阵 + 行对应的图片路径
        np.save(out_path + '.npy', features)
        metadata.to_csv(out_path + '.csv', columns=["PhotoName", "Path"], index=False)
        print("Save",  out_path + '.npy', '&', out_path + '.csv')


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
