# Doris
Dolphin ReIdentify Scripts

单独的可视化画图脚本

黑色反转？

左右背鳍，尾鳍，头部识别

only cofirm fin number larger than 5?

只能确保找到的是有的，但不能排除没找的

先有有编程经验的志愿者测试

加速 YOLO
1. batch
2. JIT
3. TRT

构造关联矩阵：
1. 图像相似度
2. 连拍序列
3. 时间距离
4. 图像距离 IOU


## TODO:
STEP2: 连拍序列里用 IOU 追踪背鳍
SETP3: 提取背鳍斑点和缺刻?
STEP5: use the clearst one as reference image? 搜索连拍图像序列中关联到的高相似度背鳍？
STEP7: 从 Blur, LOW CONFIDENCE, LOW QUALITY 里寻回
