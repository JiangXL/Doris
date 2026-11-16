"""
训练脚本：Fin Aspect (DL/DR/MISC) 分类模型训练 (PyTorch 版本)
"""

import sys
import os

# 允许从 training/ 子目录运行时导入项目根目录模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fin_aspect_classifier_torch import train_model

# 设置 GPU
if torch.cuda.is_available():
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    print("使用 CPU 训练")

if __name__ == "__main__":
    # 配置路径
    DATA_DIR = '/media/dolphin/PICWD/QinZhou/CropJPEGImages_LRSplit'

    # 训练参数
    EPOCHS = 30
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    print("=" * 50)
    print("开始训练 ResNet-101 Fin Aspect 分类模型 (PyTorch)")
    print("=" * 50)

    # 开始训练
    model, history1, history2 = train_model(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        save_dir='../models'
    )

    print("\n训练完成！")
    print("模型已保存为: models/fin_aspect_resnet101_final.pth")
