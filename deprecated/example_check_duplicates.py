"""
YOLO 重复检测检查示例脚本

此脚本演示如何：
1. 检查 YOLO 模型是否对同一张图片中的同一对象多次识别
2. 可视化重复检测的结果
3. 批量处理图片并自动去重
"""

import os
import glob
from ultralytics import YOLO
from check_duplicate_detections import (
    check_duplicate_detections,
    visualize_detections,
    process_images_with_deduplication,
    calculate_iou
)


def main():
    # 配置路径
    MODEL_PATH = "models/fin_yolo_best.pt"
    DATASET_PATH = r'/media/filming/2025-白海豚/20240824_JM_01_v2/'
    
    print("="*60)
    print("YOLO 重复检测检查工具")
    print("="*60)
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在: {MODEL_PATH}")
        print("请确保模型文件路径正确")
        return
    
    # 加载 YOLO 模型
    print(f"\n加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # 获取图片列表
    JPG_paths = glob.glob(os.path.join(DATASET_PATH, '*.JPG'))
    if not JPG_paths:
        # 尝试小写扩展名
        JPG_paths = glob.glob(os.path.join(DATASET_PATH, '*.jpg'))
    
    if not JPG_paths:
        print(f"❌ 错误: 在 {DATASET_PATH} 中未找到图片文件")
        return
    
    JPG_paths.sort()
    print(f"找到 {len(JPG_paths)} 张图片")
    
    # 示例 1: 检查单张图片
    print("\n" + "-"*60)
    print("示例 1: 检查单张图片的重复检测")
    print("-"*60)
    
    test_img = JPG_paths[0]
    print(f"检查图片: {os.path.basename(test_img)}")
    
    dup_info = check_duplicate_detections(
        test_img, 
        model, 
        iou_threshold=0.5,  # IOU 阈值，超过此值认为是同一对象
        conf_threshold=0.1  # 置信度阈值
    )
    
    print(f"  总检测数: {dup_info['total_detections']}")
    print(f"  发现重复: {dup_info['duplicates_found']}")
    
    if dup_info['duplicates_found']:
        print("\n  重复检测详情:")
        for pair in dup_info['duplicate_pairs']:
            print(f"    - 框 #{pair['box1_idx']} (置信度: {pair['box1_conf']:.3f}) vs "
                  f"框 #{pair['box2_idx']} (置信度: {pair['box2_conf']:.3f})")
            print(f"      IOU: {pair['iou']:.3f} (重叠程度)")
    
    # 示例 2: 可视化检测结果
    print("\n" + "-"*60)
    print("示例 2: 可视化检测结果")
    print("-"*60)
    print("生成可视化图像，绿色框=正常检测，红色框=重复检测")
    
    # 查找有重复检测的图片进行可视化
    for img_path in JPG_paths[:10]:  # 检查前10张
        dup_info = check_duplicate_detections(img_path, model)
        if dup_info['duplicates_found']:
            print(f"\n可视化图片: {os.path.basename(img_path)}")
            visualize_detections(img_path, model, iou_threshold=0.5)
            break
    else:
        print("前10张图片未找到重复检测，可视化第一张图片:")
        visualize_detections(JPG_paths[0], model)
    
    # 示例 3: 批量处理并去重
    print("\n" + "-"*60)
    print("示例 3: 批量处理并自动去重")
    print("-"*60)
    
    user_input = input("\n是否运行批量处理? (y/n): ")
    if user_input.lower() == 'y':
        print("\n开始批量处理...")
        meta_info, stats = process_images_with_deduplication(
            JPG_paths, 
            model, 
            DATASET_PATH,
            iou_threshold=0.5,
            conf_threshold=0.1
        )
        
        print("\n处理完成!")
        print(f"元数据已保存到: {os.path.join(DATASET_PATH, 'FIN_METAINFO.csv')}")
    else:
        print("跳过批量处理")
    
    print("\n" + "="*60)
    print("检查完成!")
    print("="*60)


if __name__ == "__main__":
    main()
