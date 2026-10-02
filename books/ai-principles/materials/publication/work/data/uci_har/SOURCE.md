# UCI HAR 手机惯性时间窗：来源、划分和用途

2026-09-24 从 [UCI 官方数据页](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)下载 id 240 的外层 ZIP，URL 为 https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip 。外层 SHA-256 为 c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031，内层 UCI HAR Dataset.zip 的 SHA-256 为 2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0。[本机准备清单](manifest.json)保存两层身份、实际通道、受试者和每类窗口数。

资料由 30 名带腰部手机的志愿者做六种活动时采集。加速度计、陀螺仪以 50 Hz 记录；官方已滤噪、分离身体加速度与重力，切成 128 时刻、2.56 秒、相邻 50% 重叠的窗口。我们只读每窗六路 body_acc_x/y/z 与 body_gyro_x/y/z，组成 \((128,6)\)；不读官方预算好的 561 个时频统计特征来冒充时间序列。原始类别为 WALKING、WALKING_UPSTAIRS、WALKING_DOWNSTAIRS、SITTING、STANDING、LAYING，本机为交叉熵改成 0–5 编号。来源不是完全原始连续流，更不是说手机直接“理解活动”。

官方训练 7,352 窗与官方测试 2,947 窗按**受试者**分开。我们在官方训练受试者中用种子 20260924 固定取 3、6、11、29 号完整受试者作验证，共 1,326 窗；其余 17 位共 6,026 窗参与训练。官方测试九位受试者（2、4、9、10、12、13、18、20、24）从未参与模型轮次选择。由于同一人的相邻窗口有重叠，以窗口为单位随机分训练/验证会把高度相似的片段两边共享；本章用受试者级分离避免该具体泄漏。六路均值与标准差只由 6,026 个训练窗口计算，再应用到验证和测试。[准备程序](../../code/prepare_uci_har.py)核实际行数、通道、标签和受试者分离；保存的 [har_sequences.npz](har_sequences.npz) SHA-256 为 1f6cf2085299a919f20e5affaf1ccd6723dbcbcc394a3f32a0a3f2b0c4e7e427。

引用：Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). *Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository. DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K). UCI 页面将数据列为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；本章保留作者与原链接，只作本机可重跑研究。
