# coding: utf-8
"""导出 Doris 运行时推理模型为 ONNX（开发专用，需要 torch/ultralytics/timm）。

产物（models/ 下，文件名沿用权重文件、只换扩展名，跨平台通用）：
  - fin_yolo_3class_PICWD_best.onnx      YOLO 检测（动态 batch，原始预测头，NMS 在 onnx_infer 的 Python 端做）
  - fin_quality_detection_resnet101_final.onnx  模糊分类 ResNet-101（224 输入，动态 batch）
  - fin_aspect_classification_resnet101_final.onnx  部位分类 ResNet-101（224 输入，动态 batch）
  - megadescriptor_dolphin_ft.onnx       MegaDescriptor-L-384 特征（384 输入，动态 batch）

用法: python3 export_models_onnx.py
"""
import os

import torch
import timm
from ultralytics import YOLO

from resource_paths import resource_path
from fin_quality_detector_torch import BlurDetectionModel
from fin_aspect_classifier_torch import FinAspectModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_weights(model, path):
    """加载权重文件（裸 state_dict 或含 'model' 键的训练存档）。"""
    obj = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(obj)
    except (RuntimeError, TypeError):
        model.load_state_dict(obj['model'])
    return model


def export_yolo():
    print("导出 YOLO 检测模型 ...")
    model = YOLO(resource_path('models/fin_yolo_3class_PICWD_best.pt'), task='detect')
    # nms=False 动态 batch 导出（ultralytics 默认 ONNX 形态，输出原始预测头
    # [batch, 4+nc, 8400]：xywh 已解码到 640 输入空间、类别分数已 sigmoid）。
    # NMS 在 onnx_infer 的 Python 端做（复刻 conf 0.25 / 按类 iou 0.7 / top-300）：
    # nms=True 的内嵌 NMS 导出图在 dynamic batch 下 batch>1 结果错误（已实测），
    # 而固定 batch 无法适配不同显存。
    out = model.export(format='onnx', nms=False, dynamic=True, imgsz=640)
    print(" ->", out)


def _export_torch(model, onnx_path, input_size):
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    # dynamo=False 用 TorchScript 导出器：torch 2.15 的 dynamo 导出需 onnxscript，
    # legacy 导出器对这些 torchvision/timm 结构成熟可靠，且产物与推理数值一致（由验证步骤兜底）
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        dynamo=False,
    )
    print(" ->", onnx_path)


def export_blur():
    print("导出模糊分类模型 ...")
    model = _load_weights(
        BlurDetectionModel(num_classes=2, pretrained=False),
        resource_path('models/fin_quality_detection_resnet101_final.pth'))
    _export_torch(model, resource_path('models/fin_quality_detection_resnet101_final.onnx'), 224)


def export_aspect():
    print("导出部位分类模型 ...")
    model = _load_weights(
        FinAspectModel(num_classes=3, pretrained=False),
        resource_path('models/fin_aspect_classification_resnet101_final.pth'))
    _export_torch(model, resource_path('models/fin_aspect_classification_resnet101_final.onnx'), 224)


def export_megadescriptor():
    print("导出 MegaDescriptor 特征模型 ...")
    model = timm.create_model('swin_large_patch4_window12_384',
                              pretrained=False, num_classes=0)
    ckpt = resource_path('models/megadescriptor_dolphin_ft.pth')
    try:
        _load_weights(model, ckpt)
    except RuntimeError:
        # 旧版 timm Swin 存档：downsample 键移位、去掉 attn_mask 等 buffer
        from timm.models.swin_transformer import checkpoint_filter_fn
        obj = torch.load(ckpt, map_location='cpu')
        sd = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
        model.load_state_dict(checkpoint_filter_fn(sd, model))
    _export_torch(model, resource_path('models/megadescriptor_dolphin_ft.onnx'), 384)


def print_io(onnx_path):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    for item in sess.get_inputs() + sess.get_outputs():
        kind = 'input ' if item in sess.get_inputs() else 'output'
        print(f"   {kind}: {item.name} shape={item.shape} type={item.type}")


if __name__ == '__main__':
    export_yolo()
    export_blur()
    export_aspect()
    export_megadescriptor()
    print("\n各模型 I/O：")
    for name in ('fin_yolo_3class_PICWD_best.onnx',
                 'fin_quality_detection_resnet101_final.onnx',
                 'fin_aspect_classification_resnet101_final.onnx',
                 'megadescriptor_dolphin_ft.onnx'):
        path = resource_path(os.path.join('models', name))
        print(name)
        print_io(path)
