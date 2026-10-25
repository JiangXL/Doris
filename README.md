# Doris
Dolphin ReIdentify Scripts


## Algorithm 

identify, OrignalImageName, Path, X, Y, W, H, AssumedAspect, ConfirmedAspect, Clearness, AssumedClear, ConfirmedClear,

### Exctra Feature from Fin 
1. crop fin and classify fin aspect
2. socre fin image quality 
3. black or white fin

### 化简
1. Tracking：Kalman filtering and Hungarian algorithm

依据相机画面晃动，计算浪涌？

### Clustering
only cofirm fin number larger than 5?

只能确保找到的是有的，但不能排除没找的

构造关联矩阵：
1. 图像相似度
2. 连拍序列
3. 时间距离
4. 图像距离 IOU

### Output Structure

Group-DolphinID-FinID

如何兼容当前格式？


## GUI Layout
There following planes:
1. Group of Continue shot
2. Group of Fin Aspect: dorsal left, dorsal right, others, not dolphin
3. Group of Fin Quality: clear or blur 
4. Group of Fin ID:  Review mode, Match mode, Overview mode
5. Group of Dolphin 
6. Group of Social Structure
7. Statistics

