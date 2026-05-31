#!/usr/bin/env python3
"""
找出最清晰的图像
使用拉普拉斯方差(Laplacian variance)来评估图像清晰度
"""

import os
import cv2
import numpy as np
from pathlib import Path

def calculate_sharpness(img):
    """
    计算图像清晰度（拉普拉斯方差）
    值越大表示图像越清晰
    """
    try:
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算拉普拉斯方差（清晰度指标）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 计算图像尺寸
        height, width = img.shape[:2]
        
        # 计算其他清晰度指标
        # 1. Sobel 边缘检测
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2).var()
        
        # 2. 图像熵（信息量）
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # 3. 综合清晰度评分 (拉普拉斯方差为主)
        sharpness_score = laplacian_var + 0.1 * sobel_mag + 0.01 * entropy * 100
        
        return {
            'path': str(image_path),
            'laplacian_var': laplacian_var,
            'sobel_var': sobel_mag,
            'entropy': entropy,
            'sharpness_score': sharpness_score,
            'width': width,
            'height': height,
            'file_size': os.path.getsize(image_path)
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
