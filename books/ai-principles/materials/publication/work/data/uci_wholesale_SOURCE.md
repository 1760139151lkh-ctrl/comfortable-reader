# C09 真实账本资料来源：UCI Wholesale customers

- 官方入口：[UCI Wholesale customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers)，DOI [10.24432/C5030X](https://doi.org/10.24432/C5030X)。UCI 页面写由 Margarida Cardoso 提供、2014-03-30 收录，标注 CC BY 4.0。页面没有给出这些年度消费各自属于哪一历年，也没有逐客户身份或日交易顺序；本书不把它编成有时间戳的预报资料。
- 从官方地址 https://archive.ics.uci.edu/static/public/292/wholesale%2Bcustomers.zip 下载的原包保存在 work/data/uci_wholesale_original.zip。本机原包为 15,175 字节，SHA-256：647e6a61683ed23f48c7b04e1e2be78835eca8758b32267603fc42a1479dbfb8。ZIP 内只有 Wholesale customers data.csv；该文件为 15,021 字节，SHA-256：c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef。解出固定文件名，不运行包内内容。
- CSV 含表头加 440 行。列依次为 Channel、Region、Fresh、Milk、Grocery、Frozen、Detergents_Paper、Delicassen。后六列是六类商品的年度消费额，UCI 以货币单位 m.u. 记录。原表另有 Channel（Horeca／Retail）和 Region（Lisbon／Oporto／其他）编码；它们是**已有的业务属性**，不是本书想“发现”的隐藏群体真值。本章的无标签训练只取后六列，不把 Channel／Region 偷送给 PCA 或聚类，也不在结果出来后拿它们给算法发放“发现真实类型”的奖牌。
- 本机初读核得 440 行、没有八列完全重复的行。六列最小值到最大值跨度明显，例如 Fresh 为 3—112151、Detergents_Paper 为 3—40827；直接用欧氏距离会使大数值支配比较。是否取对数、怎样缩放，须在运行协议里作为**建模选择**写明，不能把数据预处理说成原始记录天然有的性质。
- 这份资料有使用许可与可核原字节，适合让读者在本机重做“无逐项目标类别时，怎样描述方向与分组”。它不能让我们知道每名客户的真实意图、因果机制、未来消费或任何商业上应采取的行动。后续若讨论隐变量，那是模型中的假设对象，不是 UCI 文件里藏着一列尚未打开的正确标签。
