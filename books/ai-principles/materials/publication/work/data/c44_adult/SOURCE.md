# C44成人收入资料：原始来源、筛选历史与本机使用范围

2026-09-25 从[UCI Adult数据原页](https://archive.ics.uci.edu/dataset/2/adult)指向的固定下载地址`https://archive.ics.uci.edu/static/public/2/adult.zip`取得公开压缩包620,237字节，SHA-256 `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`；原zip全员CRC通过，`work/code/fetch_c44_adult.py`只从允许名单读取`adult.data`、`adult.test`、`adult.names`三个成员，不按zip路径任意解包，逐件SHA与大小见`manifest.json`。原zip另有`Index`及`old.adult.names`，本书未将它们当训练资料。UCI页面标数据集CC BY 4.0、引用Becker与Kohavi（1996）/DOI`10.24432/C5XW20`；本书保留原UCI归属，不把此许可句当“任何用这份人口资料作实际人员决定都已获准”。本地程序不打印单人的完整原记录，也不上传记录或训练权重到服务。

`adult.names`第7、15—21行原文说明：由1994年美国Census数据库抽取，先筛`AAGE>16`、`AGI>100`、`AFNLWGT>1`、`HRSWK>0`，再由MLC++工具按约2/3与1/3**随机**分成原训练/测试文件。原`adult.data`有32,561非空行、`adult.test`有一行`|1x3 Cross validator`头再16,281条，合计UCI页面的48,842。原训练行中2,399条至少有一个`?`缺值；当前方案将它保留为类别“未知”，不默默丢掉人。原数据字段`sex`只编码`Female/Male`，`race`原列五个粗类，目标`>50K/<=50K`是当年记录的年收入分段；这些是**文件的编码**，不能当所有人的性别、种族或个人价值的天然分类，也不能把1994年收入当应得资格/能力真值。

训练阶段**只语义读取原训练文件**，未用原`adult.test`的人员字段或标签选算法。原训练32,561行中标签`>50K`7,841、`<=50K`24,720；原`sex`字段编码`Male`21,790行其中`>50K`6,662，`Female`10,771行其中`>50K`1,179。这里的比例是**被抽取文件的行比例**，没有用`fnlwgt`推美国人口分布，也没有排除历史收入、职业、采集与筛选制度造成的差别。`fnlwgt`代表原资料的抽样权重信息，本书预测器未用它作输入；因而所有本书分组指标只能说这份文件上的统计。训练/验证决定先在`work/verification/C44_pretest_decision.md`锁定后，`work/code/c44_adult_group_study.py test`才首次为本章语义解析原测试16,281行；代码只输出聚合，不保存逐人预测。运行报告在`work/runs/c44_adult_pretest/train.json`与`test.json`，完整数字和阈值边界见`work/verification/C44_adult_privacy_fairness_protocol.md`。

[Ding、Hardt、Miller、Schmidt 2021原论文](https://proceedings.neurips.cc/paper_files/paper/2021/file/32e54441e6382a7fbacbbbaf3c450059-Paper.pdf)重新追索UCI Adult的1994 Census源与加工细节，指出这份久用基准的外部有效性限制，并给更多年份/地点/任务资料。它让本章不以一份过时、筛过的美国样本宣称现在任何招聘、贷款或社会群体的真实公平。若本章训练一个收入分段预测器，其任务仍只是**复现实验资料上的条件判断及代价**，不允许它判断实际个人资格。

本章将另外区分：公开许可、隐私技术、被记录者的权利/实际法律要求、统计群体差别和模型安全都是不同对象。即使在这个公开资料上计算了一个加噪总数，也不会倒过来使原公开zip、未加噪的统计表或训练权重享有同一数学隐私保证；任何真实数据删除请求须查适用主体/系统/保存副本及当期条款，而不能从一个`CC BY`或`ε`字样机械得出处理结论。

`work/code/c44_privacy_federated_probe.py`另只读取上述训练拟合行，以UCI编码的两组假装两个本机客户；它算的是同一份公开资料的四格Laplace发布与加权梯度代数，没有跨设备通信、加密汇总或私密模型训练。`work/runs/c44_privacy_federated_first.json`属教学结果。分组图`work/runs/c44_group_figure_first/adult_coded_group_rates.png`只由锁定后首次测试的聚合报告绘成，图中`Female/Male`为原文件编码，图已目视，不展示个人行。
