# YOLO 重复检测检查与处理

## 问题描述

在使用 YOLO 模型检测海豚背鳍时，可能会遇到**同一对象被多次识别**的情况。这会导致：
- 同一张背鳍被裁剪成多个图像
- 元数据中出现重复记录
- 后续分析出现偏差

## 解决方案

本项目提供了一套完整的工具来检查和处理重复检测问题。

## 文件说明

| 文件 | 说明 |
|------|------|
| `check_duplicate_detections.py` | 核心工具模块，包含去重算法 |
| `example_check_duplicates.py` | 示例脚本，演示如何使用工具 |
| `crop_fin_deduplicated_cell.py` | 修改后的 notebook cell 代码 |

## 核心算法：IOU 去重

使用 **IOU (Intersection Over Union)** 计算两个检测框的重叠程度：

```
IOU = 交集面积 / 并集面积
```

- IOU = 0：两个框完全不重叠
- IOU = 1：两个框完全重叠（同一对象）
- IOU > 0.5：认为可能是同一对象（可配置）

去重策略：
1. 按置信度对检测框排序
2. 保留置信度最高的检测框
3. 移除与该框 IOU 超过阈值的其他框
4. 重复直到处理完所有框

## 使用方法

### 方法 1：使用提供的工具模块

```python
from check_duplicate_detections import (
    check_duplicate_detections,
    visualize_detections,
    process_images_with_deduplication
)
from ultralytics import YOLO

# 加载模型
model = YOLO("models/fin_yolo_best.pt")

# 1. 检查单张图片
result = check_duplicate_detections('image.jpg', model, iou_threshold=0.5)
print(f"发现重复: {result['duplicates_found']}")

# 2. 可视化检测（绿色=正常，红色=重复）
visualize_detections('image.jpg', model)

# 3. 批量处理并自动去重
JPG_paths = [...]  # 图片路径列表
dataset_path = '/path/to/dataset'
meta_info, stats = process_images_with_deduplication(
    JPG_paths, model, dataset_path,
    iou_threshold=0.5,
    conf_threshold=0.1
)
```

### 方法 2：运行示例脚本

```bash
python example_check_duplicates.py
```

### 方法 3：修改 Notebook

将 `crop_fin.ipynb` 中的处理 cell 替换为 `crop_fin_deduplicated_cell.py` 中的代码。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iou_threshold` | 0.5 | IOU 阈值，超过此值认为是同一对象 |
| `conf_threshold` | 0.1 | 置信度阈值，低于此值的检测将被忽略 |

### 参数调整建议

- **如果去重太严格**（误删不同对象的检测）：提高 `iou_threshold` 到 0.7-0.8
- **如果去重不彻底**（仍有多余检测）：降低 `iou_threshold` 到 0.3-0.4
- **如果低置信度检测太多**：提高 `conf_threshold` 到 0.3-0.5

## 检测结果示例

### 正常情况
```
处理完成统计:
  总图片数: 1000
  有重复检测的图片数: 23
  重复检测对数: 31
  最终保存的裁剪图数: 987
```

### 重复检测警告
```
⚠️  发现重复检测: 0123_20240824JM01ZRA10123.JPG
   - 框 0(conf=0.823) vs 框 1(conf=0.756), IOU=0.891
🔄 0123_20240824JM01ZRA10123.JPG: 从 2 个检测中移除 1 个重复项
```

## 注意事项

1. **YOLO 内置 NMS**：YOLO 模型本身包含 NMS，但在某些情况下可能不够严格
2. **置信度调整**：如 notebook 注释所述，提高置信度阈值也可以减少重复检测
3. **可视化验证**：建议先可视化几张图片，确认去重参数设置合理

## 参考链接

- [Ultralytics Issue #5811 - Duplicate Detections](https://github.com/ultralytics/ultralytics/issues/5811)
- [IOU (Intersection Over Union) 说明](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
