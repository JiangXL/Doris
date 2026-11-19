# ONNX 迁移总结

> Doris 推理后端从 PyTorch/ultralytics/timm 迁移到 ONNX Runtime 的完整记录（2026-11）。

## 目标与结果

把 Doris 的全部推理从 PyTorch/ultralytics/timm 迁到 ONNX Runtime，消除部署对 torch CUDA 栈的依赖。**打包体积 6.5GB → 2.1GB（–67%）**，GPU 加速保留（CUDA EP），数值一致性经验证。

## 架构

```
export_models_onnx.py（开发专用，需 torch/timm/ultralytics）
  └─ 导出 4 个 models/*.onnx（文件名沿用权重名，跨平台通用）

onnx_infer.py（运行时，零 torch 依赖）
  ├─ OnnxFinDetector    YOLO 检测：letterbox（逐像素复刻 ultralytics）
  │                     → ORT → conf 过滤 → 按类 NMS → 坐标还原
  │                     → batch 运行时旋钮 + OOM 自动减半重试
  ├─ OnnxClassifier     blur/aspect 共用（224，PIL 复刻 torchvision 预处理）
  └─ OnnxFeatureExtractor  MegaDescriptor Swin-L-384 特征（动态 batch + 降批重试）
```

**4 个模型**：YOLO26m 检测（动态 batch）、模糊 ResNet-101、部位 ResNet-101、Swin-L-384 特征。Step2/Step3/gui 全部切到 ONNX；`check_duplicate_detections` 兼容 numpy；requirements 拆分为 runtime（`onnxruntime-gpu` + `opencv-python-headless` 等）与 dev（torch 栈，见 `requirements-dev.txt`）；`Doris.spec` 剔除 torch 并 collect onnxruntime 动态库。

## 关键坑与决策（按发现顺序）

1. **torch 2.15 dynamo 导出需 onnxscript** → 改用 `dynamo=False` legacy 导出器。
2. **ultralytics 8.4 缺陷：`dynamic=True` + 内嵌 NMS 的导出图在 batch>1 时结果错误**（CPU/GPU 同现）→ 第一版改固定 batch=16；固定 batch padding 不省显存、且 64 在 11GB 卡 OOM → 最终方案：**nms=False 动态导出 + Python 端 NMS**（batch 成为运行时旋钮，显存连续自适应）。
3. **yolo26 导出模式图内做 topk-300 预选**，score 列尺度比 torch conf 高约 9~16%（IoU 类分数并入），但 0.25 过滤边界等价（实测计数逐图一致）；min 匹配 IoU 0.944（密集场景个别赢家被预选替换，裁剪用途无实际影响）。尝试提升 topk 上限无法从外部覆盖，已接受并注释在 `onnx_infer.py`。
4. **ORT CUDA 运行库**：onnxruntime-gpu 1.32 需 CUDA 13 + cuDNN 9；本机借 torch 的 nvidia pip 库 + `nvidia-curand-cu12` 解决；无库时自动回退 CPU。
5. **ORT 激活内存 > torch eager**：Swin 特征提取 batch 32→8；步骤间需释放会话（GUI 流程本就 del + gc）。
6. **Windows 部署缺 CUDA 运行库**（先报 `LoadLibrary failed for cudnn64_9.dll`，补 cuDNN 后又报 `Failed to load cudart64_13.dll`）：Linux 靠 `LD_LIBRARY_PATH` 指到 pip 的 nvidia 库，Windows 无此机制、PyInstaller 默认也不收 nvidia 包的 DLL。ORT CUDA EP 需要整套 CUDA 13（cudart/cublas/cublasLt/cufft/curand）+ cuDNN 9。解决：`Doris.spec` 在 `win32` 下收集这 5 个 pip 包的 DLL（需构建 venv 装了对应 `nvidia-*-cu13` 包）；注意这会让 Windows 安装包增大约 1.5GB（CUDA 库体积），发布需权衡。

## 验证（全部通过）

- 分类器概率差 <1e-3、类别一致；特征余弦相似度 =1.0（diff 4e-8）。
- YOLO：动态 batch 1/4/16/24 结果完全一致；与 PyTorch 对比 24 张真实图计数/类别逐图一致、IoU≥0.944。
- `_is_oom_error`/降批重试单测；真实数据 Step2→Step3 端到端（35 鳍→特征 35×1536）；torch 导入封禁下运行正常；CPU 回归通过。
- PyInstaller：dist 无 torch/timm/ultralytics，onnxruntime 含 CUDA EP 库，离屏启动正常。

## 遗留事项

- fp16 ONNX 可再省约一半模型体积（数值有微差），可作后续项。
- Step5 中 `CropConf` 现在为图内综合置信度（尺度比 torch 管线高约 10%，单调性基本一致），如影响排序可向 Step5 加归一化。
