# C29 无类别目标的服饰图像表示实验：实际口径

2026-09-24 本机执行。此文件是作者的实验记录，不是读者正文或用户验收。

## 原件与信息时序

- 复用 C13 的 Fashion-MNIST 官方四个 IDX gzip 包；work/data/fashion_mnist/manifest.json 逐包保存官方链接、MD5、SHA-256 与原项目许可，work/data/fashion_mnist/chapter_split.npz 固定 50,000 训练/10,000 验证。C13 建该切分时按十类分层，故只能称 C29 的**表示训练器不读类别标签**，不能声称本项目从未借类别作资料准备。
- work/code/fashion_ssl_lab.py 的 train 模式只经 load_images 打开训练图像像素及既有切分索引；训练、验证目标和选轮不调用 load_labels。它对原官方测试图像字节计算 SHA 以核源身份，但不解出图像行、不在训练或选轮用测试像素/类别。
- 冻结编码器后，probe 模式才核类别 gzip 的 SHA，再打开类别编号；每类从旧训练段抽 500 张，共 5,000 标签训练线性读出。10,000 张 C13 验证挑各支读出的 C；之后看 10,000 张官方测试。**该官方测试已在 C13 及本章早期未标准化探针出现过**，现有数值是同集复核/事后对照，不是作者盲测或独立部署资料。若重跑，不因换输出文件名而变盲。

## 同一图像资料上的三种目标

同一个两次卷积+池化后接 128 维隐藏层的编码器从随机初值训练；对比支另接 64 维归一化投影头，简化重建支接全图线性解码器。AdamW 学习率 0.001、weight decay 1e-4、批图像 256、每支 8 轮，每轮 196 批。每轮取 C13 验证前 2,048 张、固定随机视图/遮盖的**自监督损失**选最佳轮；类别标签不参与。完整实际训练与检查点 SHA 在 work/runs/c29_fashion_contrast/、c29_fashion_masked/、c29_fashion_wrong_pair/ 的 train.json 和 best.pt。

1. 正确配对对比：同一张原图独立做左右移动至多 2 像素、水平翻转、亮度扰动、轻噪声。两视图是正例，其余原图在批内作负例；2B=512 时分母含 511 项，温度 0.2。最佳第 8 轮验证目标约 1.8962；这是同原图可辨的任务成绩，不是十类准确率。
2. 错配对照：**完全相同的图像、两次视图生成和更新预算**，但把第二批视图按批内顺序循环挪一位；指定正例遂来自不同原图，真正同原图的另一视图进入负例。最佳第 4 轮验证目标约 6.23637，接近所有 511 候选均匀时 log(511)≈6.23637。该对照是有意错误的关系，不代表某原论文的训练方法。
3. 简化遮块重建：约一半 4×4 像素块置零，整个含零图经同编码器，线性解码器输出 28×28，仅在遮挡位置计原像素均方误差。最佳第 6 轮验证目标约 0.048165。MAE 原论文**只让可见 patch 入 encoder，另用轻 decoder**；本支不是 MAE 复刻。MSE也不要求补出的图像保持十类可分。选中编码器在 128 维里有 126 维对 2,048 验证图的原始标准差低于 0.01；配合下游成绩与目视重建，可报告本实现的弱/近退化表示，不能从坐标尺度独立证明所有图严格相同。

work/code/plot_c29_views.py 已生成并目视检查 work/figures/c29_same_image_two_views.png 与 c29_masked_views.png。前者同源图仍保主要形状；后者重建给出模糊、相似的衣裤鞋轮廓。这是第 6 轮所选模型的八张验证图示例，不能仅凭图证明所有预测或其它数据上的视觉质量。生成的索引与模型身份在 work/results/c29_view_figure.json。

为免把“模糊”误判成“只会输出平均图片”，work/results/c29_fashion_masked_mean_baselines.json 在**同一 2,048 验证图和固定遮盖**比较三个不训练的猜法：全零像素 MSE 0.20636、全训练集单一平均亮度 0.12515、逐位置训练平均图 0.08695。训练后的遮块支为 0.04817，确实利用了可见图像的信息；问题是这份较低重建误差没有变成好用的十类线性表示。编码器多数坐标低方差也不等于整条 128 维向量对所有图完全相同。

## 冻结编码器后的有限类别检验

读出使用 sklearn StandardScaler **只在 5,000 张有标签训练特征上拟合**，再用 LogisticRegression(lbfgs，max_iter=2000) 拟合；每支的 C 从 0.01/0.1/1/10 按验证准确率选择。冻结编码器不在这一步改权重。当前四支的迭代均达到设定容限，最大候选步数低于 2,000。旧 work/results/c29_fashion_linear_probe.json 未作特征标准化且触发收敛警告，已被下表当前 work/results/c29_fashion_linear_probe_wrong_pair.json 取代。文件 work/results/c29_fashion_linear_probe_scaled.json 则是加入错配支前的三支同口径复核，结果与下表对应三支完全相同。

| 冻结图像编码器 | 验证准确率 | 已看过的官方测试准确率 | 解释 |
|---|---:|---:|---|
| 同结构随机参数 | 79.95% | 79.76% | CNN 随机特征在简易服饰图上并非零基线 |
| 同原图正确视图对比训练 | 82.73% | 82.08% | 当前同类少标签读出上高于随机 2.32 个百分点 |
| 不同原图错当正例 | 81.37% | 81.26% | 尽管自监督辨别目标近均匀，特征读出仍非全无作用；正确配对只高 0.82 个测试百分点，单种子不能声称配对带来稳健大增益 |
| 简化遮块重建 | 51.04% | 50.01% | 重建损失可下降，但这个 encoder 的类别线性读出明显较差 |

C13 使用完整 50,000 张类别标签、不同 105,866 参数分类CNN 五轮训练，官方测试 87.79%；这里只用 5,000 张标签并冻结另一套 128维编码器，87.79% 只给读者位置感，不能拿它作这四支的公平优劣判决。所有上述数值只对 Fashion-MNIST 商品灰图、指定增强、网络、预算、单个随机种子和此线性读出有效；不能证明街景、图文语义、图像生成或用户审美。

## 另一份真实彩色场景：CIFAR-10

官方二进制归档已通过网页所列 MD5 核验，实际170052171字节；来源、技术报告引用要求、未确认的具体再分发许可及训练/测试成员SHA见 work/data/cifar10_binary/SOURCE.md、manifest.json。下载器在首次训练前为完整性已解压并计算过测试成员SHA；**训练和验证不解测试图像/类别，也不拿测试选模型**。训练文件每行的首字节是类别，work/code/cifar_ssl_lab.py 在 train_pixels 里只取后3072 RGB像素；随机、不按类别分层划45000/5000行，其索引序列SHA8b597c99...，不解释标签字节。到编码器冻结后的 probe 才读取训练/验证行的首字节；每类抽500行共5000训练线性读出，5000验证选C，全部选择结束才解官方10000测试行的图像与类别。其“首次测试”只对本章这支实验的方法选择成立。

保持 Fashion 对比支的卷积/128维编码器职能、64维投影头、AdamW学习率0.001/weight decay1e-4、批256、温度0.2；输入改三通道32×32，平移范围改至±4，其他两视图变化仍水平翻转、亮度扰动和轻噪声。实际训八轮，每轮176批；在固定2,048验证视图的无类别目标选第八轮，损失第1轮2.36801→第8轮1.93223，GPU PyTorch峰已分配971108864字节。模型和运行SHA绑定在 work/runs/c29_cifar_contrast/train.json、best.pt；同资料图片及两次增强图 work/figures/c29_cifar_same_scene_two_views.png 已目视检查，来源行号在 work/results/c29_cifar_view_figure.json，图未展示/解释标签。

冻结后 StandardScaler 仅拟合每类500训练行特征；LogisticRegression 的C从0.01/0.1/1/10在5000验证行选，两支均选0.1，候选最大迭代低于2000。work/results/c29_cifar_linear_probe.json 是这一支在本任务**首次打开官方测试内容**的结果：同结构随机编码器验证0.3578、测试0.3627；正确同图对比编码器验证0.4458、测试0.4548。测试差0.0921，即9.21个百分点，对这次同任务/同标签预算有效；不作跨域或全体自监督系统的保证。32×32像素和八轮小网络的绝对45.48%远不代表成熟自然视觉。

本机已执行：python work/code/cifar_ssl_lab.py train --epochs 8 --run-dir work/runs/c29_cifar_contrast；随后 python work/code/cifar_ssl_lab.py probe --run-dir work/runs/c29_cifar_contrast --out work/results/c29_cifar_linear_probe.json；以及 python work/code/plot_c29_cifar_views.py。重新跑时自选新目录/JSON，官方测试身份不会因重复运行重新变盲。

## 本机复核命令

在包根目录，预先存在 C13 官方数据与切分后运行。输出目录/文件不可已有内容；重跑请自选新的路径，避免覆盖原收据。

~~~powershell
$env:PYTHONUTF8='1'
python work/code/fashion_ssl_lab.py train --objective contrast --epochs 8 --out-dir work/runs/c29_my_contrast
python work/code/fashion_ssl_lab.py train --objective contrast_wrong --epochs 8 --out-dir work/runs/c29_my_wrong_pair
python work/code/fashion_ssl_lab.py train --objective masked --epochs 8 --out-dir work/runs/c29_my_masked
python work/code/fashion_ssl_lab.py masked-baseline --out work/runs/c29_my_mask_baselines.json
python work/code/fashion_ssl_lab.py probe --contrast-dir work/runs/c29_my_contrast --wrong-dir work/runs/c29_my_wrong_pair --masked-dir work/runs/c29_my_masked --out work/runs/c29_my_probe.json
~~~
