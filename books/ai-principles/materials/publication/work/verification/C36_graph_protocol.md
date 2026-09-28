# C36 真引用图、重排、平滑、瓶颈和真空间点的执行范围

## 原件与作者内部划分

LINQS官方 `cora.tgz` 168052字节SHA `0d4ed463d1627bb7f3e8420effe8f88dab44ce86cee7b26d7e8`；原README与2708行content/5429行`<被引者> <引用者>`有向引用直接读，1433匿名二值词维/7论文类别。存储与未经确认再分发许可见`work/data/cora_linqs_original/SOURCE.md`。`work/code/prepare_c36_cora.py`解析原件、核ID/二值/列数/来源边，作者选择在模型中将真实有向引用合成5278个互异无向对，并加入每节点自环。按每类20篇共140个训练标签，剩按稳定哈希取500验证、1000模型留出、1068剩余不计监督损失/报告成绩；不是原Planetoid切分。**全部2708篇词和边在训练时可见**，这是传导式节点分类而不是新论文节点归纳式测试；准备切分已读全部标签，留出只对模型权重/轮次未见。

## 同形状真实比较与首次内部测试

`work/code/c36_cora_node_models.py`三臂均两层线性1433→64→7、ReLU/Dropout0.5、AdamW同超参数、固定250轮，只在140标签算训练CE，在500标签按最小验证NLL选权重；测试1000标签在三支选好后由`work/code/evaluate_c36_cora.py`首次统一打开。MLP只用各篇匿名词；GCN各层用固定 \(S=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}\)传播，S只由真实引用边及度计算；错边GCN同代码、相同词/标签但邻接来自`prepare_c36_rewired.py`的52780次合法度保持双边交换（原5278边仅42边保留）。图真实性及度保留由`degree_rewire_manifest.json`记录。

严格的物理读取边界有一次返工：**首次三支训练代码曾载入全2708标签数组进内存，虽然唯一索引是140训练/500验证，1000测试标签从未进入损失或选轮；因此不能声称该进程从未读取测试标签字节。**发现后用`c36_separate_visible_labels.py`从已锁切分生成只有140训练+500验证标签的`train_validation_labels_only.json`，现行训练`load_data(..., include_test_labels=False)`仅读这个文件并让其余2068行标签置-1，实际统一评估才以`include_test_labels=True`载入全标签。文件SHA在`train_validation_label_view_manifest.json`。这项修订发生于原内部测试已打开后，不把后续同1000题比较叫第二份新盲测。

种子20260925验证选轮 MLP64、真GCN130、错边GCN56；首次内部1000标签 test NLL1.40241/0.73730/1.86520，准确0.529/0.772/0.278。将真GCN已选权重直接换错边不重训的事后输入干预降为0.467，它检验**依赖原连接**，不应说新训练的同度错边模型同一参数。后验用原全标签统计，真实无向边同类4275/5278=0.80997，错边932/5278=0.17658；这个事后解释统计包括测试类，不是模型输入/超参数。完整数值在`work/results/c36_cora_first_test.json`。

首次test七类数量不均；从该报告逐类准确率等权平均，MLP/真GCN/错边约0.5463/0.7846/0.2839，仍同方向。对称边双向列成10556个非自环矩阵项，加2708自环共13264非零，固定图稀疏乘法与1433→64→7线性参数均可在源码核；这不是一般大图复杂度的实测速率。

另一随机种子7三臂**在第一次测试已打开后**同样250轮方案独立重训，不能叫第二份盲测；各验证选93/98/48轮，同1000图后验复核准确0.536/0.775/0.274，真GCN固定权重换错边0.469，见`work/results/c36_cora_seed7_posthoc_recheck.json`。两个种子在这份作者划分上维持同方向，但仍不能推广到所有图、所有边定义或新论文归纳设置。

新标签加载器下，种子20260925三支重训`work/runs/c36_cora_{mlp,gcn,rewired}_visible_labels/`仍验证择64/130/56轮；相同1000题事后复核0.529/0.772/0.278、NLL与初次报告逐项同，见`work/results/c36_cora_visible_labels_posthoc_recheck.json`。种子7的test-free加载器三支`work/runs/c36_cora_{mlp,gcn,rewired}_seed7_visible_labels/`亦择93/98/48轮，同1000题后验复核0.536/0.775/0.274，见`work/results/c36_cora_seed7_visible_labels_posthoc_recheck.json`。稀疏GPU分支最佳检查点文件字节可能不同而留出结果在报告精度同，不能凭一次SHA差说语义换了。

`work/figures/c36_cora_neighborhood_and_validation_v3.png`已由作者实际打开目视：左为真实论文编号1117与十个邻接论文的局部（源引用有向，展示时合并方向），标签颜色在打开全部标签后仅作后验说明，不能冒充训练时所有邻居类已知；该局部8/10同类接近全图后验0.81，没有挑全同类的初稿节点。右图是**仅载入140训练/500验证标签文件**后重训的三臂500节点验证NLL/各自选轮，非test。初稿 `c36_cora_neighborhood_and_validation.png`选全同类局部会给过强印象；v2虽改成有异类邻域却仍读首次训练曲线，现行v3进一步绑定物理隔离标签的重训轨迹，不把旧图删掉冒充未发生。生成码和图收据在 `work/code/plot_c36_cora.py` 与 `work/results/c36_cora_figure_v3.json`。

## 结构恒等式和局限探针

- `work/code/c36_permutation_probe.py`在实际已训GCN和2708节点/5278边上同时重排X和A，输出按同一重排比较最大差1.91e-6、整图平均读出最大差1.19e-7；只重排X不重排边则均绝对logit差1.317。结果`work/results/c36_permutation_probe.json`是重命名等变的数值核验，不是训练准确率的新来源。
- `work/code/c36_smoothing_probe.py`对真实匿名词特征**不带任何可调W/ReLU**重复S传播0/1/2/4/8/16/32步，图能量`tr(Hᵀ(I−S)H)`从121.40→13.98→4.727→1.531→0.491→0.147→0.0413，随机点对特征RMS距离0.371→0.0944。Cora共有78个连通分量，最大2485点；因此并不宣称所有点同一个常量。本机谱式证明只针对固定对称S，不能偷换为“所有深层可训练GNN都必然失败”。报告`work/results/c36_smoothing_probe.json`、目视图`work/figures/c36_fixed_smoothing.png`。
- `work/code/c36_bottleneck_probe.py`的两跳星形手核/torch微分：一标量hub对m个远叶和目标根取平均时，任一叶对根的局部导数1/(m+1)；m=1/4/16/64各0.5/0.2/0.0588235/0.0153846。换sum聚合为1，说明这个数不是所有图层的万能力量界；如果目标要求识别许多叶各自的信息，固定一维瓶颈仍可能损信息。此只是过挤压机制的作者构造，而非真实Cora成绩或Alon/Yahav定理复现，`work/results/c36_bottleneck_probe.json`。
- `work/code/c36_real_geometry_equivariance.py`从C34现存ETH3D外部激光参考反投影的22485点局部PLY中随机取256真三维位置和颜色，另按当前位置自己建8近邻图；中位邻距约0.365m作固定径向尺度。整体旋转(23,-41,17)度加平移(0.7,-0.3,1.2)m后，邻居ID完全不变；基于平方距离的标量消息不变最大2.89e-15、相对位移加权向量按同R旋转最大3.16e-15。坐标乘2但径向尺度固定时标量消息平均变化0.623，不具尺度不变性。这个性质来自相对坐标/距离的手写式，**没有训练EGNN**、没有给三维语义部件。源资产受ETH3D CC BY-NC-SA4，见`work/data/eth3d_two_view/SOURCE.md`；结果`work/results/c36_real_geometry_equivariance.json`。

为了让读者练习直接可执行，后来给平滑脚本加了`--max-rounds`、几何脚本加了`--rotation-deg-xyz`/`--translation-m`/`--sigma-scale`，不改变默认公式。新码默认重新运行结果`work/results/c36_smoothing_cli_full_recheck.json`与原能量最大仅约1.49e-8浮点差、距离全同；2轮分支`c36_smoothing_two_rounds.json`精确停在2。几何默认重跑`c36_geometry_cli_default_recheck.json`与原结果同，改变角(61,5,-33)度/平移(1,-2,0.5)m仍刚体误差<5.4e-15，`c36_geometry_sigma2_variant.json`把径向尺度翻倍后仍刚体等变但对空间倍增的响应另变。旧初次运行文件均保留而不覆盖，变体是已知条件下的核验，不算新科学样本。

## 不能从这些成绩直接声称的事

Cora词维匿名、无可靠明确再分发许可，无法从特定维直接解释某论文“写了什么词”；test节点的词和边在模型训练前可见。平均邻居有用依赖图中同类连接占比，在异类关系或边方向重要的任务不自动成立。真Cora两层GCN胜MLP与错边的两个种子，是这份作者切分的经验差，不是一般图网络胜全部非图算法的证明。数学探针性质分别限于明确的对称S、均值hub或刚体变换权重；不能合并为单个深GNN普遍定理。分子性质、推荐和机器人应用未在本章由Cora实验实际评价，应各保对象/反馈与安全边界。
