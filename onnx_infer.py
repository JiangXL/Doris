# coding: utf-8
"""ONNX Runtime 推理封装：YOLO 检测 / ResNet-101 分类 / MegaDescriptor 特征提取。

零 torch / torchvision / timm / ultralytics 依赖，供 Step2/Step3 运行时调用。
图像预处理逐像素复刻 ultralytics LetterBox 与 torchvision transforms 的语义
（letterbox/scale_boxes 公式与 ultralytics 源码逐步对齐，见各函数注释）。
"""
import os
import sys
import ast
import ctypes

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort


def _has_nvidia_driver() -> bool:
    """是否存在 NVIDIA 驱动。无 NVIDIA GPU 的机器跳过 CUDA 运行库预加载，
    避免冻结包在无 GPU 的机器上启动推理时刷 CUDA DLL 加载失败日志
    （此时本就走 CPU 回退，行为不受影响）。"""
    if sys.platform == 'win32':
        try:
            ctypes.CDLL('nvcuda.dll')  # 驱动安装到 System32，可加载即认为有 GPU
            return True
        except OSError:
            return False
    return os.path.isdir('/proc/driver/nvidia')


if _has_nvidia_driver():
    ort.preload_dlls()

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_PROVIDERS = ('CUDAExecutionProvider', 'CPUExecutionProvider')


def is_gpu_available() -> bool:
    """onnxruntime 是否可用 CUDA 执行。"""
    return 'CUDAExecutionProvider' in ort.get_available_providers()


def make_session(onnx_path, providers=None):
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_path}")
    avail = ort.get_available_providers()
    prov = [p for p in (providers or DEFAULT_PROVIDERS) if p in avail]
    if not prov:
        raise RuntimeError("onnxruntime 无可用执行后端（CUDA/CPU 均不可用）")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        return ort.InferenceSession(onnx_path, sess_options=opts, providers=prov)
    except Exception:
        # CUDA 运行库缺失等场景下回退 CPU（ORT 部分版本直接抛异常而非降级）
        if 'CPUExecutionProvider' not in prov:
            raise
        return ort.InferenceSession(onnx_path, sess_options=opts,
                                    providers=['CPUExecutionProvider'])


def read_names_from_metadata(session, fallback=None):
    props = session.get_modelmeta().custom_metadata_map
    if 'names' in props:
        return ast.literal_eval(props['names'])
    return fallback or {}


def letterbox_bgr(img, new_shape=640):
    """复刻 ultralytics LetterBox(new_shape, auto=False, scaleFill=False,
    scaleup=False, stride=32, center=True) 的图像变换（auto=False 时 padding
    不做 stride 取模）。"""
    shape = img.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)  # scaleup=False
    new_unpad = max(round(shape[1] * r), 1), max(round(shape[0] * r), 1)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2  # center=True
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def scale_boxes_to_orig(boxes, img1_shape, img0_shape):
    """复刻 ultralytics.ops.scale_boxes(padding=True, xywh=False)：
    把 letterbox 后图像坐标系下的 xyxy 还原到原图并裁剪到边界。"""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    new_h, new_w = (max(round(x * gain), 1) for x in img0_shape[:2])
    gain_y, gain_x = new_h / img0_shape[0], new_w / img0_shape[1]
    pad_x, pad_y = (round((img1_shape[1] - new_w) / 2 - 0.1),
                    round((img1_shape[0] - new_h) / 2 - 0.1))
    boxes = boxes.copy()
    boxes[..., 0] = (boxes[..., 0] - pad_x) / gain_x
    boxes[..., 1] = (boxes[..., 1] - pad_y) / gain_y
    boxes[..., 2] = (boxes[..., 2] - pad_x) / gain_x
    boxes[..., 3] = (boxes[..., 3] - pad_y) / gain_y
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, img0_shape[1])
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, img0_shape[0])
    return boxes


def _is_oom_error(exc: BaseException) -> bool:
    """判定 onnxruntime CUDA 显存分配失败（仅 ORT 抛出的内存类错误，不误伤其他异常）。"""
    if "onnxruntime" not in type(exc).__module__.lower():
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(k in text for k in ("failed to allocate", "out of memory", "cuda oom"))


def _run_with_batch_retry(run, batch):
    """以 batch 分块执行 run(batch_size)；ORT 显存不足时批大小逐级减半重试，
    适配不同显存的 GPU。batch=1 仍失败则向外抛。"""
    while True:
        try:
            return run(batch)
        except Exception as exc:
            if batch <= 1 or not _is_oom_error(exc):
                raise
            batch = max(1, batch // 2)


def _xywh_to_xyxy(boxes):
    out = boxes.copy()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def _nms_per_class(xyxy, conf, cls, iou_thres):
    """按类贪心 NMS（复刻 ultralytics 默认 agnostic=False）：
    conf 降序输入，返回保留下来的原数组下标（升序）。"""
    keep = []
    suppressed = np.zeros(len(xyxy), dtype=bool)
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for pos, i in enumerate(idx):
            if suppressed[i]:
                continue
            keep.append(i)
            x0, y0, x1, y1 = xyxy[i]
            area_i = (x1 - x0) * (y1 - y0)
            for j in idx[pos + 1:]:
                if suppressed[j]:
                    continue
                u0, v0, u1, v1 = xyxy[j]
                ix0, iy0 = max(x0, u0), max(y0, v0)
                ix1, iy1 = min(x1, u1), min(y1, v1)
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                if inter <= 0:
                    continue
                union = area_i + (u1 - u0) * (v1 - v0) - inter
                if inter / union > iou_thres:
                    suppressed[j] = True
    return keep


class _ShimBoxes:
    """模拟 ultralytics Boxes 的最小接口：xyxy/conf/cls/orig_shape + len/getitem。"""
    __slots__ = ('xyxy', 'conf', 'cls', 'orig_shape')

    def __init__(self, xyxy, conf, cls, orig_shape):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)
        self.orig_shape = orig_shape

    def __len__(self):
        return self.xyxy.shape[0]

    def __getitem__(self, idx):
        item = _ShimBoxes.__new__(_ShimBoxes)
        item.xyxy = self.xyxy[idx:idx + 1]          # (1, 4)，兼容 xyxy[0] 迭代
        item.conf = float(self.conf[idx])           # 标量，兼容 float(...)
        item.cls = int(self.cls[idx])               # 标量，兼容 int(...)
        item.orig_shape = self.orig_shape
        return item


class _ShimResult:
    """模拟 ultralytics Results 的最小接口：orig_img/names/boxes。"""
    __slots__ = ('orig_img', 'names', 'boxes')

    def __init__(self, orig_img, names, boxes):
        self.orig_img = orig_img
        self.names = names
        self.boxes = boxes


class OnnxFinDetector:
    """YOLO 背鳍检测器（ONNX 动态 batch，NMS 在 Python 端复刻 ultralytics 语义）。
    接口对齐 Step2 对 ultralytics YOLO 的用法：
    detector(photos, verbose=False) -> list[result]。

    推理批大小由 batch 控制；ORT 显存不足时自动减半重试（适配不同显存 GPU）。
    图输出为 top-300 候选 [x1,y1,x2,y2,score,cls]（640 输入空间，未做 NMS），
    后处理：conf 过滤 -> 按类 NMS -> 坐标还原。
    注意：score 列为图内综合置信度，与 ultralytics torch 管线的 conf 尺度
    略有差异（实测比值 ~1.09-1.16），但 0.25 过滤边界等价（检测计数逐图一致）。"""

    def __init__(self, onnx_path, imgsz=640, batch=16,
                 conf_thres=0.25, iou_thres=0.7, max_det=300):
        self.imgsz = imgsz
        self.batch = batch
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.session = make_session(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.names = read_names_from_metadata(self.session)

    def _run_chunked(self, letterboxed, batch):
        outputs = []
        for start in range(0, len(letterboxed), batch):
            chunk = letterboxed[start:start + batch]
            blob = np.stack(chunk)[..., ::-1].transpose(0, 3, 1, 2)
            blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
            outputs.append(self.session.run(None, {self.input_name: blob})[0])
        return np.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]

    def __call__(self, photos, verbose=False):
        letterboxed = [letterbox_bgr(p, self.imgsz) for p in photos]
        outputs = _run_with_batch_retry(lambda b: self._run_chunked(letterboxed, b),
                                        self.batch)

        results = []
        for img, det in zip(photos, outputs):
            orig_shape = img.shape[:2]
            results.append(_ShimResult(img, self.names,
                                       self._postprocess(det, orig_shape)))
        return results

    def _postprocess(self, det, orig_shape):
        det = np.asarray(det, dtype=np.float32).reshape(-1, 6)
        # 图输出为 top-300 候选（按分数取，未做 NMS，无 conf 过滤，零分填充）
        xyxy = det[:, :4]
        conf = det[:, 4]
        cls = det[:, 5]
        keep = conf >= self.conf_thres
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
        if len(conf) == 0:
            return _ShimBoxes(np.zeros((0, 4)), np.zeros(0), np.zeros(0), orig_shape)
        order = np.argsort(-conf, kind="stable")
        xyxy, conf, cls = xyxy[order], conf[order], cls[order]
        keep_idx = np.array(_nms_per_class(xyxy, conf, cls, self.iou_thres),
                            dtype=int)
        # 各类内部已按 conf 降序，类间重排后取 top-max_det
        keep_idx = keep_idx[np.argsort(-conf[keep_idx], kind="stable")][:self.max_det]
        xyxy = scale_boxes_to_orig(xyxy[keep_idx], (self.imgsz, self.imgsz), orig_shape)
        return _ShimBoxes(xyxy, conf[keep_idx], cls[keep_idx], orig_shape)


def _to_rgb(item):
    """将路径(str) / BGR ndarray / PIL Image 统一转为 PIL RGB（复刻 torch 版 _to_pil）。"""
    if isinstance(item, str):
        return Image.open(item).convert('RGB')
    if isinstance(item, np.ndarray):
        if item.ndim == 2:
            return Image.fromarray(item).convert('RGB')
        if item.ndim == 3 and item.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
        if item.ndim == 3 and item.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGRA2RGB))
        return Image.fromarray(item).convert('RGB')
    if isinstance(item, Image.Image):
        return item.convert('RGB')
    raise TypeError(f"不支持的图像输入类型: {type(item)}")


def _softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


class OnnxClassifier:
    """ResNet-101 图像分类器（ONNX）。blur/aspect 共用一套代码，
    返回格式与 torch 版 predict_batch 一致。显存不足时批大小自动减半重试。"""

    def __init__(self, onnx_path, class_names, input_size=224, batch=64):
        self.session = make_session(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = list(class_names)
        self.input_size = input_size
        self.batch = batch

    def _preprocess(self, item):
        # 复刻 torchvision: PIL 双线性 resize -> /255 -> ImageNet normalize -> CHW
        img = _to_rgb(item).resize((self.input_size, self.input_size), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = (a - IMAGENET_MEAN) / IMAGENET_STD
        return a.transpose(2, 0, 1)

    def _run_chunked(self, image_inputs, batch):
        probs_all = []
        for start in range(0, len(image_inputs), batch):
            chunk = image_inputs[start:start + batch]
            tensors = np.stack([self._preprocess(i) for i in chunk])
            logits = self.session.run(None, {self.input_name: tensors})[0]
            probs_all.append(_softmax(np.asarray(logits, dtype=np.float32)))
        return np.concatenate(probs_all, axis=0)

    def predict_batch(self, image_inputs):
        probs = _run_with_batch_retry(lambda b: self._run_chunked(image_inputs, b),
                                      self.batch)
        results = []
        for i, p in enumerate(probs):
            idx = int(np.argmax(p))
            result = {
                'class': self.class_names[idx],
                'class_idx': idx,
                'confidence': float(p[idx]),
                'probabilities': {name: float(p[j])
                                  for j, name in enumerate(self.class_names)},
            }
            if isinstance(image_inputs[i], str):
                result['path'] = image_inputs[i]
            else:
                result['index'] = i
            results.append(result)
        return results


class OnnxFeatureExtractor:
    """MegaDescriptor 深度特征提取（ONNX）。输入输出语义与 torch 版一致：
    384 输入、ImageNet normalize，返回原始 embedding ndarray [N, D]。
    显存不足时批大小自动减半重试。"""

    def __init__(self, onnx_path, input_size=384):
        self.session = make_session(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

    def _preprocess(self, item):
        img = _to_rgb(item).resize((self.input_size, self.input_size), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = (a - IMAGENET_MEAN) / IMAGENET_STD
        return a.transpose(2, 0, 1)

    def _run_chunked(self, items, batch_size):
        features = []
        for start in tqdm(range(0, len(items), batch_size), 
                          desc="Exract DeepFeatures"):
            batch = np.stack([self._preprocess(i) for i in
                              items[start:start + batch_size]])
            out = self.session.run(None, {self.input_name: batch})[0]
            features.append(np.asarray(out, dtype=np.float32).reshape(
                batch.shape[0], -1))
        return np.concatenate(features, axis=0) if features else np.zeros((0, 0), np.float32)

    def extract(self, items, batch_size=32):
        """对图像列表（路径/ndarray/PIL）提取特征，返回 [N, D] float32 ndarray。"""
        return _run_with_batch_retry(lambda b: self._run_chunked(items, b),
                                     batch_size)
