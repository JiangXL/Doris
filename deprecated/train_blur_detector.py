"""
训练脚本：快速开始模糊检测模型训练
"""

import tensorflow as tf
from blur_detector import train_model, BlurDetector

# 设置 GPU 内存增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU {gpu} 已设置内存增长")

if __name__ == "__main__":
    # 配置路径
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    
    # 训练参数
    EPOCHS = 30
    BATCH_SIZE = 32
    
    print("=" * 50)
    print("开始训练 ResNet-101 模糊检测模型")
    print("=" * 50)
    
    # 开始训练
    model, history1, history2 = train_model(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    print("\n训练完成！")
    print("模型已保存为: blur_detection_resnet101_final.h5")
