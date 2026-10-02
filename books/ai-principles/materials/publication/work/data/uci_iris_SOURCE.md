# C08 候选资料：UCI Iris 原包身份与历史边界

- 官方入口：[UCI Iris](https://archive.ics.uci.edu/dataset/53/iris)，DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76)，页面标注 CC BY 4.0。UCI 页面写 1988-06-30 收录；压缩包内 iris.names 另写捐赠日期为 1988 年 7 月，二者具体日期有出入。本章若叙述时间，只说 1988 年进入 UCI，不把某一天写成确定历史事实。
- 原始下载：https://archive.ics.uci.edu/static/public/53/iris.zip 。2026-09-23 本机取得 3,738 字节，SHA-256：d11fe30213d36434a0879aab7cb00ce3c812eb7ba2495874438abff7b7b762e9，保存为 work/data/uci_iris_original.zip。ZIP 内固定成员为 Index、bezdekIris.data、iris.data、iris.names；没有运行其中内容。
- iris.data 为 4,551 字节，SHA-256：6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0。Python csv 读取后有 150 行，setosa、versicolor、virginica 各 50 行，前四列是以厘米计的萼片长宽、花瓣长宽，第五列是类别。有两组四项测量完全相同，其中一组出现三次、另一组两次；仅凭四项数相同，不能断定是同一株花重复入表，后续切分前要核查这层身份。
- [Fisher 1936 年原论文](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1469-1809.1936.tb02137.x)第 179—180 页明确感谢 **Edgar Anderson** 提供花的测量；论文先用同一群落的 setosa 与 versicolor 研究线性判别，随后在第 185—188 页讨论 virginica，并特别注明第三类不是来自前两类所在的同一自然群落。不能只把资料叫“Fisher 自己测的三类花”，也不能把他的原问题改写成今天程序库的随机三分类分数。
- UCI 页面自己指出其公开 iris.data 与 Fisher 1936 论文表格存在至少两行差异（第 35、38 个 setosa 样本）。因此本机下载的字节是 **UCI 版本**，不是原论文表格的无误复制。若第七章在该版本上运行，结果归于 UCI 版本，并在正文注明这种差别。
- 该资料在几十年的统计和机器学习教学中反复公开使用，样本仅 150 条。任何本机切分只宜用于说明方法如何组织和受什么条件限制；不能把一个已公开、已由作者读过描述的小表称为真正未知生态地区的未来盲测。

资料已下载和核对，**尚未对 C08 的具体学习任务、特征、切分、模型或评分作出运行前约定，也没有运行分类比较。**
