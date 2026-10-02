# Bike Sharing 日记录来源

- 来源：[UCI Machine Learning Repository，Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)，DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894)；资料提供者 Hadi Fanaee-T，页面登记于 2013-12-19。
- 原始下载地址：https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip
- 原始 ZIP SHA-256：b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401；本机取得 279,992 字节。保存在 work/data/uci_bike_sharing_original.zip。
- 从压缩包**按固定文件名读取** day.csv 和 Readme.txt，不运行包内代码。day.csv 的 SHA-256：a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be；Readme.txt 的 SHA-256：b92c8628622948bf0828c43a1b315d1517c3d7fd63654730d0dea379b8e88175。
- UCI 页面标明 CC BY 4.0；保留来源、作者与许可，并依 Readme.txt 的要求引用 Hadi Fanaee-T、João Gama 的文章 [Event labeling combining ensemble detectors and background knowledge](https://doi.org/10.1007/s13748-013-0040-3)。
- 原文说明这是美国 Washington, D.C. 的 Capital Bikeshare 2011—2012 年租车日志按日、小时聚合并附天气和日历信息。本章**另定**一天提前预测次日总租车数的任务，不把这项课堂实验称作作者原论文的事件检测复现。
- day.csv 有 731 行，从 2011-01-01 至 2012-12-31。当前任务只读 dteday、cnt，并用日期自行计算周末；不读取目标日的实际天气、湿度、风速，也不把 casual、registered 两列当输入，因为它们构成当天目标 cnt。
- UCI 网页与包内 Readme.txt 对 season 编码和小时记录数的描述存在差异；本章不依赖 season 或 hour.csv。运行程序将核对日记录自己的日期连续性、序号与 cnt=casual+registered，不以网页总实例数替代日文件实际行数。

本次只是把有许可证的历史资料用于教学与本机实验；它只能支持这一个城市、这两年的有限比较，不提供当前交通预测能力保证。
