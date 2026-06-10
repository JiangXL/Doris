#!/usr/bin/env python
# coding: utf-8
# 提取背鳍特征

import os
import timm
import pandas as pd
import torchvision.transforms as T

from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import ImageDataset

class FinFeatureExtractor:
    """Extract deep features from fin images using MegaDescriptor."""
    DEFAULT_MODEL = 'hf-hub:BVRA/MegaDescriptor-L-384'
    DEFAULT_IMG_SIZE = 384
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_DEVICE = 'cuda'

    def __init__(
        self,
        model_name=DEFAULT_MODEL, # timm model identifier.
        batch_size=DEFAULT_BATCH_SIZE,
        device=DEFAULT_DEVICE, # 'cuda' or 'cpu'
        img_size = DEFAULT_IMG_SIZE, 
        csv_name='METAINFO/FIN_METAINFO.csv',
        output_name='METAINFO/FIN_DEEPFEATURES',
    ):
        self.model_name = model_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device
        self.csv_name = csv_name
        self.output_name  = output_name

        self.dataset = None
        self.extractor = None
        self.model = timm.create_model(self.model_name, pretrained=True)
        self.transform = T.Compose([
            T.Resize([self.img_size, self.img_size]),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def extract(self, root_dir): #Root directory containing images and metadata CSV.
        """ extract and save deepfeature. """
        csv_path = os.path.join(root_dir, self.csv_name)
        metadata = pd.read_csv(csv_path)
        self.dataset = ImageDataset(
            root=root_dir,
            metadata=metadata.query(f'select==True'),
            transform=self.transform,
            col_label="identity",
            col_path="path",
        )
        self.extractor = DeepFeatures(
            self.model,
            device=self.device,
            batch_size=self.batch_size,
        )
        features = self.extractor(self.dataset)
        features.save( os.path.join(root_dir, self.output_name) )
        return features

if __name__ == '__main__':
    import sys
    #root_dir = r'/media/filming/2025-白海豚/20240825-JM_02-3/'
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
    else:
        print("No root directory is provided")
    
    extractor = FinFeatureExtractor()
    extractor.extract(root_dir)
