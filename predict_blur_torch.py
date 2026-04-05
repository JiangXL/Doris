"""
预测脚本：使用训练好的模型进行模糊检测 (PyTorch 版本)
"""

import glob
import os
from blur_detector_torch import BlurDetector

def main():
    # 模型路径
    MODEL_PATH = 'models/blur_detection_resnet101_final.pth'
    
    # 初始化检测器
    if os.path.exists(MODEL_PATH):
        detector = BlurDetector(MODEL_PATH)
        print(f"已加载模型: {MODEL_PATH}")
    else:
        print(f"警告: 未找到模型文件 {MODEL_PATH}，使用未训练的模型")
        detector = BlurDetector()
    
    # 示例1: 单张预测
    print("\n" + "=" * 50)
    print("单张图像预测示例")
    print("=" * 50)
    
    test_image = './test_images/sample.jpg'
    if os.path.exists(test_image):
        result = detector.predict(test_image)
        print(f"图像: {test_image}")
        print(f"预测结果: {result['class']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"概率分布: {result['probabilities']}")
    else:
        print(f"示例图像不存在: {test_image}")
    
    # 示例2: 批量预测
    print("\n" + "=" * 50)
    print("批量图像预测示例")
    print("=" * 50)
    
    test_dir = './test_images'
    if os.path.exists(test_dir):
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(test_dir, ext)))
            images.extend(glob.glob(os.path.join(test_dir, ext.upper())))
        
        if images:
            print(f"找到 {len(images)} 张图像")
            results = detector.predict_batch(images[:10])  # 最多预测10张
            
            blur_count = 0
            for r in results:
                status = "🔴 模糊" if r['class'] == '模糊' else "🟢 清晰"
                print(f"{status} | {os.path.basename(r['path'])} | 置信度: {r['confidence']:.2%}")
                if r['class'] == '模糊':
                    blur_count += 1
            
            print(f"\n统计: {blur_count}/{len(results)} 张模糊图像")
        else:
            print(f"目录中没有找到图像: {test_dir}")
    else:
        print(f"示例目录不存在: {test_dir}")

if __name__ == "__main__":
    main()
