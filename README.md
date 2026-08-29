# Doris
Dolphin ReIdentify Scripts

单独的可视化画图脚本

黑色反转？

左右背鳍，尾鳍，头部识别

only cofirm fin number larger than 5?

只能确保找到的是有的，但不能排除没找的

先有有编程经验的志愿者测试

构造关联矩阵：
1. 图像相似度
2. 连拍序列
3. 时间距离
4. 图像距离 IOU

依据相机画面晃动，计算浪涌？


## TODO:
[x]STEP2: 连拍序列里用 IOU 追踪背鳍
[x]SETP2: 计算相机抖动,如果背鳍占的比例小,则用相机抖动补偿
[]SETP3: 提取背鳍斑点和缺刻?
[]STEP5: use the clearst one as reference image? 搜索连拍图像序列中关联到的高相似度背鳍？
[]STEP7: 从 Blur, LOW CONFIDENCE, LOW QUALITY 里寻回
[]STEP5: 从高相似度里找不同的，从低相似度里找相同的

## UI:
There following planes:
1. Single full image Plane
1.1 Group by Continue shot
2. Group by Fin Aspect: dorsal left, dorsal right, tail
3. Group by Fin Blur or not: 
4. Group by Fin ID  
5. Group by Social Structure


## Tracking
Kalman filtering and Hungarian algorithm
