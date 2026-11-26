#TODO
[] SETP3: 提取背鳍斑点和缺刻?
[] STEP5: 从高相似度里找不同的，从低相似度里找相同的
[] 识别 claf 与 adult: 黑色和白色分类讨论, 先优先匹配白色的
[] use the best image to confirm the aspect
[x] add the short key and bottom to mark best in FinID tab
[] 好拍的海豚高清晰的图多？难拍的都糢糊？
[] 黑色反转？

[] 先匹配高质量背鳍，后匹配低质粒背鳍
[] 用上相机对焦区域的信息
[] 筛选yolo 结果时，偏好 box 小的
[] 从抖动补偿后的海豚运动方向，推断背鳍方向
[] Clearness hisogram 表示真阳和假
[] hightlight outlier
[] 两两映射图
[] 在 full-open window 里，右键菜单，加入 "MISC", "NOTE"
[] keep full-open window when user delete one of photo
[] 拆分 Track ，根据图像相似度？在哪里拆分？
[] 独立 Statistic 画图逻辑和主GUI。无须主 GUI 处理数据。

[x] enable user to update bonding box in full-e
[] overview mode for shot tab
[] 设计检漏页面
[x] bug: 修改 bonding box 后的图，无法定位到原始照片
[] bug: after move selected fin image instead of whole track, the BestInShot/BestInFin label may lost
[] error check: whether single FinID has multiple dolphinid?
[] error check :don't assign DphID to fin without FinID
[] assign new BestInShot for new FinID and orignal FinID when user only move selected fin images out of current FinID, 
[x] Plot: water fall traffic
[x] allow user to new fin before extract feature
[] allow user to new/delete fin before extract feature
[x] new fin
[] rename social strcture to Subgroup

[x] New dolphin 
[x] Low quality 60
[x] marker the dolhpin with NN label, assign the same font color for the same group
[x] 根据 NN 排序 DphID
[] OpenVINO for intel?

[] FinID/DolphinID relationship check
[] Reread orignal fin image instead of thumb image in dolphin overview
[] 同一 full-view 窗口样式: 右键菜单、信息条
[] 设计 doris_dataset 的成员变量, 构造 fin/dph
[] configure file
[] 后台计算过程？

11/23, 柳佳蓉
1. 记住全屏窗口的大小 [x]
2. 修复 ShotType, PhotoType 的导入类型错误，或者类型检查
3. 设计 Fin0 的候选捕捞框 [x]
4. 新建背鳍后，直接赋予FinID? 不提取 deepfeature. 直接修改或指定背鳍的 ID?[x]
5. 统一全屏窗口的标签样式。
6. 同一背鳍缩略图标签样式。
7. 更换 Dolphin Tab 的Match Mode 里的 connection matrix 为 similarity matrix
8. 导出含 FinID 的 ppt

11/24
1. rename connection matrix to incidence matrix
2. move the border label away from fin image
