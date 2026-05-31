"""
检查和处理 YOLO 模型多次识别同一对象的问题

功能：
1. 检查检测结果中是否存在重复识别（同一对象被多次检测）
2. 提供 IOU 去重功能，保留置信度最高的检测框
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
