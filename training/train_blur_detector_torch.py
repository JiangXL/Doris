"""
训练脚本：快速开始模糊检测模型训练 (PyTorch 版本)
"""

import torch
from blur_detector_torch import train_model, BlurDetector

# 设置 GPU
if torch.cuda.is_available():
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    print("使用 CPU 训练")

if __name__ == "__main__":
    # 配置路径
    TRAIN_DIR = '/home/hf/working/中华白海豚/mydata/PICWD/QinZhou/blur_dataset/train'
    VAL_DIR = '/home/hf/working/中华白海豚/mydata/PICWD/QinZhou/blur_dataset/val'
    
    # 训练参数
    EPOCHS = 30
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    print("=" * 50)
    print("开始训练 ResNet-101 模糊检测模型 (PyTorch)")
    print("=" * 50)
    
    # 开始训练
    model, history1, history2 = train_model(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        save_dir='./models'
    )
    
    print("\n训练完成！")
    print("模型已保存为: models/blur_detection_resnet101_final.pth")
