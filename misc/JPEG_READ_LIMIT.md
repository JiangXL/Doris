每张 23MB 的 q98 JPEG 解码固需 ~660ms 核时间，16 核已达 ~85% 并行效率，16→更多线程吞吐几乎不涨（内存带宽/解码器特性所限）
 。

这套图上解码:GPU 内核 ≈ 40:1，GPU 有效占比确实只有 2~5%（前面小图上测的 55~63% 不适用）。GPU  97% 时间在等解码。

对于 20MB 左右的超大 JPEG 文件（通常对应 8K～16K 分辨率，即 3,000 万至 1.3 亿像素的高清图），

| 解码方式	| 单张 20MB 图像耗时	| 相对速度	| 核心瓶颈与前提条件
| CPU 单线程 (libjpeg-turbo / OpenCV) | 150 ms – 400 ms	| 1× (基准)	霍夫曼熵解码是串行操作，单核吃满
| nvJPEG（混合模式：消费级/老架构 GPU）如 RTX 3090 / 4090 / T4 / V100 | 60 ms – 160 ms | 约 1.5× – 3×|	90% 以上时间仍耗在 CPU 做流解析与熵解码，GPU 仅加速 IDCT 与色彩转换
| nvJPEG（硬件加速模式：数据中心 GPU）如 A100 / A30 / H100 / L40S / B200 |15 ms – 40 ms	| 约 8× – 15×|显卡内置多核专用硬件 JPEG 解码器（NVDEC/nvJPEG HW），彻底绕过 CPU 瓶颈


JPEG 解码分为两步：熵解码（Huffman Decoding） 和 反量化/IDCT/色彩空间转换。
• 没有硬件解码器的显卡（如 RTX 4090）：因为标准 JPEG 缺乏 Restart Markers 时，熵解码无法在 GPU 上高度并行，
nvJPEG 会退回“CPU 熵解码 + GPU IDCT”的混合模式。根据 Fastvideo 在 RTX 4090 上的基准测试，解码 8K～16K 超大图时，
nvJPEG 单线程吞吐量约为 3.4 – 14.9 fps（单张耗时约 67 ms – 290 ms），其中 91%–93% 的时间花在 CPU 端解析码流，
只比纯 CPU（libjpeg-turbo 单核约 185 ms）快一点。
• 带硬件 JPEG 解码器的显卡（如 A100、H100）：NVIDIA 官方技术博客指出，A100 内置 5 核专用硬件 JPEG 解码引擎，
熵解码完全卸载到硬件，速度比纯 CPU 提升最高可达 20 倍（4K 图像约 1 ms/张，20MB 级超大图通常在 15–40 ms 左右）。

[1] (https://developer.nvidia.com/blog/leveraging-hardware-jpeg-decoder-and-nvjpeg-on-a100/),
[2] (https://forums.developer.nvidia.com/t/a100-hardware-nvjpeg-batch-decoding-takes-5ms-before-decoding-and-why/228533)
[3] (https://github.com/fastvideo/jpeg-benchmark), 
[4] (https://www.reddit.com/r/CUDA/comments/1y0i3ot/exactgpujpeg_cuda_jpeg_decodeencode_thats/)


ONNX 模型推理，20MB JPEG 解码后生成的 RGB 图像在内存中会膨胀到 100MB – 400MB（例如 8192×8192×3 ≈ 201 MB）.
纯 CPU 解码花 200+ ms 解码，还要把几百 MB 的原始像素通过 PCIe 总线从内存拷到显存（耗时约 15–40 ms），并占用大量 CPU 核心。

真实大图集（12.4MP/23MB q98、含 MPO/渐进/无损混合编码）口径（`05[F]_GM`，400 张实测）：端到端被 JPEG 解码限死在 ~20 img/s
（660ms-core/张，16 核已 ~85% 并行效率，加线程无效）；此时 GPU 有效占比仅 ~2-5%（内核:解码 ≈ 1:40，即用户观测的 5% 的来源）。
磁盘侧：btrfs 无压缩，暖页缓存读取 4.6GB/s 不是瓶颈，但冷读为 SMR  HDD 顺序 ~163MB/s ≈ 7 img/s，整目录 88GB 冷扫需 ~9 分钟，
冷跑时磁盘会反限解码。TurboJPEG 对此类文件仅 +25%（渐进编码优势缩小）且色度上采样路径与 cv2 有系统像素差（2/3 通道差 >2），
未采用。GPU 分支参数据此调为 decode=全部核。
