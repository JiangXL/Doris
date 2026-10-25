#!/usr/bin/env python
# coding: utf-8
# 提取背鳍特征（通用版，不依赖 wildlife_tools）
#   - 只用 torch + timm + pandas + PIL，wildlife_tools 只是一个薄封装
#     （ImageDataset = 普通 Dataset；DeepFeatures = 批量前向传播）
#   - 支持任意 timm 模型（默认 MegaDescriptor-L-384，用本地权重完全离线加载），
#     可加载本地微调权重
#   - 同时输出两种格式：
#       1. wildlife_tools FeatureDataset 兼容的 pickle（下游 Step4-7 可直接用）
#       2. 通用格式 FIN_DEEPFEATURES.npy（特征矩阵）+ FIN_DEEPFEATURES.csv（对应图片路径）

import os
import pickle

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


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


class FinFeatureExtractorGeneral:
    """Extract deep features from fin images using any timm model."""

    # MegaDescriptor-L-384 即 timm 的 swin_large_patch4_window12_384（num_classes=0），
    # 用本地权重加载即可完全离线；如需联网版可传 model_name='hf-hub:BVRA/MegaDescriptor-L-384'
    DEFAULT_MODEL = 'swin_large_patch4_window12_384'
    DEFAULT_PRETRAINED = 'models/megadescriptor_L_384.pth'  # MegaDescriptor 官方权重（本地）
    DEFAULT_CHECKPOINT = 'models/megadescriptor_dolphin_ft.pth'
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
        output_name=None,           # default depends on checkpoint, see below
        checkpoint= DEFAULT_CHECKPOINT,            # local fine-tuned weights (.pth); None = 用预训练权重
        pretrained_weights=DEFAULT_PRETRAINED,     # 本地预训练权重；hf-hub 模型时忽略（从 HF 加载）
        save_portable=True,         # also save .npy + .csv next to the pickle
    ):
        self.model_name = model_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.csv_name = csv_name
        self.checkpoint = checkpoint
        self.pretrained_weights = pretrained_weights
        self.save_portable = save_portable
        self.output_name = 'METAINFO/FIN_DEEPFEATURES'

        use_hf = self.model_name.startswith('hf-hub:')
        # MegaDescriptor 这类特征提取模型 num_classes=0；hf-hub 模型由 config.json 自带该配置
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
    def extract(self, root_dir):  # Root directory containing images and metadata CSV.
        """Extract and save deep features."""
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
        for images in loader:
            feats = self.model(images.to(self.device)).cpu()
            features.append(feats)
        features = torch.cat(features).numpy()

        out_path = os.path.join(root_dir, self.output_name)
        self._save(out_path, features, metadata, root_dir)
        return features

    def _save(self, out_path, features, metadata, root_dir):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # wildlife_tools FeatureDataset 兼容格式（下游 Step4-7 的 from_file 可直接读取）
        with open(out_path, 'wb') as f:
            pickle.dump({
                'features': features,
                'metadata': metadata,
                'col_label': 'identity',
                'load_label': True,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.save_portable:
            # 通用格式：特征矩阵 + 行对应的图片路径，任何工具都能读
            np.save(out_path + '.npy', features)
            metadata.to_csv(out_path + '.csv', index=False)


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        root_dir = sys.argv[1]
    else:
        print("No root directory is provided")
        sys.exit(1)
    extractor = FinFeatureExtractorGeneral()
    extractor.extract(root_dir)
