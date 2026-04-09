"""
检查和处理 YOLO 模型多次识别同一对象的问题

功能：
1. 检查检测结果中是否存在重复识别（同一对象被多次检测）
2. 提供 IOU 去重功能，保留置信度最高的检测框

When the same fin are detected multiple times, try to increase the confidence or filter with IoU.
https://github.com/ultralytics/ultralytics/issues/5811
"""

import os
import cv2
import glob
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt


def calculate_iou(box1, box2):
    """
    计算两个边界框的 IOU (Intersection Over Union)
    
    参数:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]
    
    返回:
        iou: float, IOU 值 (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # 检查是否有交集
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    # 计算交集面积
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算两个框的面积
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # 计算 IOU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


def filter_duplicate_detections(boxes, iou_threshold=0.5):
    """
    使用 NMS (Non-Maximum Suppression) 原理过滤重复的检测框
    
    参数:
        boxes:
        iou_threshold: IOU 阈值，超过此值认为是同一对象
    
    返回:
        keep_indices: 保留的检测框索引列表
    """
    if len(boxes) == 0:
        return []
    # list of [x_min, y_min, x_max, y_max]
    xyxy = boxes.xyxy.cpu().numpy()
    # list of confidence scores
    confs = boxes.conf.cpu().numpy()
    # 按置信度降序排序
    indices = np.argsort(confs)[::-1]
    
    keep_indices = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
        keep_indices.append(i)
        # 抑制与当前框 IOU 过高的其他框
        for j in indices:
            if j == i or j in suppressed:
                continue
            iou = calculate_iou(xyxy[i], xyxy[j])
            if iou > iou_threshold:
                suppressed.add(j)
    return sorted(keep_indices)


def check_duplicate_detections(image_path, model, iou_threshold=0.5, conf_threshold=0.1):
    """
    检查单张图片的检测结果是否有重复识别
    
    参数:
        image_path: 图片路径
        model: YOLO 模型
        iou_threshold: IOU 阈值
        conf_threshold: 置信度阈值，低于此值的检测将被忽略
    
    返回:
        duplicates_info: dict，包含重复检测信息
    """
    results = model(image_path)
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        return {
            'image_path': image_path,
            'total_detections': 0,
            'duplicates_found': False,
            'duplicate_pairs': []
        }
    
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    
    # 过滤低置信度检测
    valid_indices = [i for i, c in enumerate(confs) if c >= conf_threshold]
    boxes = boxes[valid_indices]
    confs = confs[valid_indices]
    
    total_detections = len(boxes)
    duplicate_pairs = []
    
    # 检查所有框对之间的 IOU
    for i in range(total_detections):
        for j in range(i + 1, total_detections):
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                duplicate_pairs.append({
                    'box1_idx': i,
                    'box2_idx': j,
                    'box1_conf': float(confs[i]),
                    'box2_conf': float(confs[j]),
                    'iou': float(iou)
                })
    
    return {
        'image_path': image_path,
        'total_detections': total_detections,
        'duplicates_found': len(duplicate_pairs) > 0,
        'duplicate_pairs': duplicate_pairs,
        'boxes': boxes,
        'confs': confs
    }


def process_images_with_deduplication(JPG_paths, model, dataset_path, 
                                       iou_threshold=0.5, conf_threshold=0.1):
    """
    处理图片并去除重复检测
    
    参数:
        JPG_paths: 图片路径列表
        model: YOLO 模型
        dataset_path: 数据集路径
        iou_threshold: IOU 阈值
        conf_threshold: 置信度阈值
    """
    fin_img_id_list = []
    path_list = []
    orignal_img_list = []
    orig_img_w_list = []
    orig_img_h_list = []
    x_min_list = []
    x_max_list = []
    y_min_list = []
    y_max_list = []
    conf_list = []
    
    duplicate_stats = {
        'total_images': 0,
        'images_with_duplicates': 0,
        'total_duplicates': 0
    }
    
    for JPG_path in JPG_paths:
        ori_img_name = os.path.basename(JPG_path)
        duplicate_stats['total_images'] += 1
        
        # 检查重复检测
        dup_info = check_duplicate_detections(JPG_path, model, iou_threshold, conf_threshold)
        
        if dup_info['duplicates_found']:
            duplicate_stats['images_with_duplicates'] += 1
            duplicate_stats['total_duplicates'] += len(dup_info['duplicate_pairs'])
            print(f"⚠️  发现重复检测: {ori_img_name}")
            for pair in dup_info['duplicate_pairs']:
                print(f"   - 框 {pair['box1_idx']}(conf={pair['box1_conf']:.3f}) vs "
                      f"框 {pair['box2_idx']}(conf={pair['box2_conf']:.3f}), "
                      f"IOU={pair['iou']:.3f}")
        
        # 运行检测
        results = model(JPG_path)
        
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            # 过滤低置信度
            valid_indices = [i for i, c in enumerate(confs) if c >= conf_threshold]
            boxes = boxes[valid_indices]
            confs = confs[valid_indices]
            
            if len(boxes) == 0:
                continue
            
            # 去重处理
            keep_indices = filter_duplicate_detections(boxes, confs, iou_threshold)
            
            if len(keep_indices) < len(boxes):
                print(f"🔄 {ori_img_name}: 从 {len(boxes)} 个检测中移除 "
                      f"{len(boxes) - len(keep_indices)} 个重复项")
            
            for new_idx, fin_idx in enumerate(keep_indices):
                x0, y0, x1, y1 = [int(i) for i in boxes[fin_idx]]
                x_min_list.append(x0)
                y_min_list.append(y0)
                x_max_list.append(x1)
                y_max_list.append(y1)
                conf = confs[fin_idx]
                conf_list.append(float(conf))
                
                # 保存裁剪图片
                cropped_img = result.orig_img[y0:y1, x0:x1, :]
                save_dir = os.path.join(dataset_path, "FIN/")
                os.makedirs(save_dir, exist_ok=True)
                img_name = ori_img_name[:-4] + f"_FIN{new_idx:02d}.JPG"
                orignal_img_list.append(ori_img_name)
                path_list.append("FIN/" + img_name)
                cv2.imwrite(os.path.join(save_dir, img_name), cropped_img)
                
                orig_img_h, orig_img_w = result.boxes.orig_shape
                orig_img_h_list.append(orig_img_h)
                orig_img_w_list.append(orig_img_w)
    
    # 保存元数据
    fin_img_id_list = range(len(path_list))
    meta_info = pd.DataFrame({
        "img_id": fin_img_id_list,
        "path": path_list,
        "x_min": x_min_list,
        "x_max": x_max_list,
        "y_min": y_min_list,
        "y_max": y_max_list,
        "orig_img": orignal_img_list,
        "crop_conf": conf_list,
        "orig_img_h": orig_img_h_list,
        "orig_img_w": orig_img_w_list
    })
    meta_info.to_csv(os.path.join(dataset_path, "FIN_METAINFO.csv"))
    
    # 打印统计信息
    print("\n" + "="*50)
    print("处理完成统计:")
    print(f"  总图片数: {duplicate_stats['total_images']}")
    print(f"  有重复检测的图片数: {duplicate_stats['images_with_duplicates']}")
    print(f"  重复检测对数: {duplicate_stats['total_duplicates']}")
    print(f"  最终保存的裁剪图数: {len(path_list)}")
    print("="*50)
    
    return meta_info, duplicate_stats


def visualize_detections(image_path, model, iou_threshold=0.5, save_path=None):
    """
    可视化检测结果，标出重复检测的框
    
    参数:
        image_path: 图片路径
        model: YOLO 模型
        iou_threshold: IOU 阈值
        save_path: 保存路径（可选）
    """
    import matplotlib.patches as patches
    
    dup_info = check_duplicate_detections(image_path, model, iou_threshold)
    
    # 绘制原图和检测框
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    
    if dup_info['total_detections'] > 0:
        boxes = dup_info['boxes']
        confs = dup_info['confs']
        
        # 标记重复对
        duplicate_indices = set()
        for pair in dup_info['duplicate_pairs']:
            duplicate_indices.add(pair['box1_idx'])
            duplicate_indices.add(pair['box2_idx'])
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(boxes)))
        
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            # 重复检测用红色，其他用绿色
            is_duplicate = i in duplicate_indices
            edgecolor = 'red' if is_duplicate else 'green'
            linewidth = 3 if is_duplicate else 2
            
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=linewidth, edgecolor=edgecolor, facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加标签
            label = f"#{i} conf={conf:.2f}"
            if is_duplicate:
                label += " (DUP)"
            ax.text(x_min, y_min - 5, label, 
                   color=edgecolor, fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_title(f"Detections: {len(boxes)}, Duplicates: {len(duplicate_indices)}")
    else:
        ax.set_title("No detections")
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()
    
    return dup_info


if __name__ == "__main__":
    # 示例用法
    print("="*50)
    print("YOLO 重复检测检查工具")
    print("="*50)
    print("\n使用方法:")
    print("1. 导入模块: from check_duplicate_detections import *")
    print("2. 加载模型: model = YOLO('models/fin_yolo_best.pt')")
    print("3. 检查单张图片:")
    print("   dup_info = check_duplicate_detections('image.jpg', model)")
    print("4. 可视化检测:")
    print("   visualize_detections('image.jpg', model)")
    print("5. 批量处理并去重:")
    print("   process_images_with_deduplication(JPG_paths, model, dataset_path)")
