"""
这是修改后的 crop_fin.ipynb Cell 代码，添加了重复检测过滤功能

将此代码替换 notebook 中的处理循环 cell
"""

import numpy as np


def calculate_iou(box1, box2):
    """计算两个边界框的 IOU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def filter_duplicate_detections(boxes, confs, iou_threshold=0.5):
    """
    过滤重复的检测框，保留置信度最高的
    
    参数:
        boxes: list of [x_min, y_min, x_max, y_max]
        confs: list of confidence scores
        iou_threshold: IOU 阈值
    
    返回:
        keep_indices: 保留的检测框索引列表
    """
    if len(boxes) == 0:
        return []
    
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
            
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                suppressed.add(j)
    
    return sorted(keep_indices)


# 主处理循环（带去重）
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

# 去重参数
IOU_THRESHOLD = 0.5  # IOU 超过此值认为是同一对象
CONF_THRESHOLD = 0.1  # 置信度阈值

print("开始处理图片，启用重复检测过滤...")
duplicate_count = 0

for JPG_path in JPG_paths:
    ori_img_name = os.path.basename(JPG_path)
    results = fin_detector(JPG_path)
    
    for result in results:
        boxes_data = result.boxes
        
        if boxes_data is None or len(boxes_data) == 0:
            continue
        
        # 提取所有检测框和置信度
        all_boxes = boxes_data.xyxy.cpu().numpy()
        all_confs = boxes_data.conf.cpu().numpy()
        
        # 过滤低置信度
        valid_mask = all_confs >= CONF_THRESHOLD
        all_boxes = all_boxes[valid_mask]
        all_confs = all_confs[valid_mask]
        
        if len(all_boxes) == 0:
            continue
        
        # 去重处理：保留置信度最高的检测框
        keep_indices = filter_duplicate_detections(all_boxes, all_confs, IOU_THRESHOLD)
        
        removed_count = len(all_boxes) - len(keep_indices)
        if removed_count > 0:
            duplicate_count += removed_count
            print(f"🔄 {ori_img_name}: 移除 {removed_count} 个重复检测，保留 {len(keep_indices)} 个")
        
        # 只保存去重后的检测框
        for new_idx, fin_idx in enumerate(keep_indices):
            xyxy = all_boxes[fin_idx]
            x0, y0, x1, y1 = [int(i) for i in xyxy]
            
            x_min_list.append(x0)
            y_min_list.append(y0)
            x_max_list.append(x1)
            y_max_list.append(y1)
            
            conf = float(all_confs[fin_idx])
            conf_list.append(conf)
            
            # 裁剪并保存
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

print(f"\n处理完成! 共移除 {duplicate_count} 个重复检测")
print(f"最终保存 {len(path_list)} 个裁剪图像")

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
print(f"元数据已保存到: {os.path.join(dataset_path, 'FIN_METAINFO.csv')}")
