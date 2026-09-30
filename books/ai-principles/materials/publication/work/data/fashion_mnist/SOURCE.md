# Fashion-MNIST 数据身份与本章用法

2026-09-24 从 [Zalando 原项目](https://github.com/zalandoresearch/fashion-mnist) 的 data/fashion 文件，经 HTTPS 原始文件地址下载。项目 [README 的四项下载与 MD5 表](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md#get-the-data)、[2017 年数据论文](https://arxiv.org/abs/1708.07747)与本地 [manifest](manifest.json)共同标识：28×28 单通道服饰产品图片，官方训练 60,000 张、测试 10,000 张，10 个类别。它不是未经标注的互联网照片，也不是本章作者采集或绘制的图片。

| 原始 gzip | 官方 MD5 核对 | 本机 SHA-256 |
|---|---|---|
| train-images-idx3-ubyte.gz | 8d4fb7e6c68d591d4c3dfef9ec88bf0d | 3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84 |
| train-labels-idx1-ubyte.gz | 25c81989df183df01b3e8a0aad5dffbe | a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845 |
| t10k-images-idx3-ubyte.gz | bef4ecab320f06d8554ea6380940ec79 | 346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073 |
| t10k-labels-idx1-ubyte.gz | bb300cfdad3c16e7a12a480ee83cd310 | 67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5 |

[prepare_fashion_mnist.py](../../code/prepare_fashion_mnist.py)在下载后核四个压缩包的官方 MD5、IDX 魔数／长度、10 类数量，按类别用种子 20260924 从官方训练部分各取 1,000 张作验证。训练 50,000、验证 10,000，官方测试保持 10,000；本章没有把测试图用于定义像素置换、选择 epoch 或设置训练参数。每类固定置换的索引与供实验用的 784 像素全局置换保存在 [chapter_split.npz](chapter_split.npz)，其 SHA-256 是 ac870595933df8a373507f0728aa2d275645e05ac53a052556995c49fe43ea9b。像素只除以 255。

项目 README 声明 [MIT 许可](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE)，本目录保留所下载的 [LICENSE](LICENSE) 原文；许可文字以“software and associated documentation”为对象，本章不由此推断所有原始商品照片在其他传播场景的单独权利。这里保存官方压缩包，仅作本地可重跑的教学实验；若另行分发图像或商业使用，应再核数据权利。读者正文引用数据原项目，不把这组服饰物图推广为自然场景、检测、图文对齐或生成任务的证据。
