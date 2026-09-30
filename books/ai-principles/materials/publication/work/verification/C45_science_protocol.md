# 第四十五章本机研究边界与执行结果

此文件记录作者实跑，不代替读者正文、独立审稿或用户掌握。2026-09-25 Windows 本机执行；Python已有NumPy、SciPy、Astropy、Matplotlib、PyTorch，热实验报告记录`device=cuda`。无外部训练权重、付费服务或真实科学新发现。历史来源实读范围见`work/research/第四十五章来源核查.md`。

## HAT-P-7：真实但经过NASA管线处理的测光

原件、预定分割和质量修订：`work/data/c45_kepler_hatp7/SOURCE.md`、`work/verification/C45_kepler_pretest_decision.md`。Q1 FITS SHA `37faa7dc82fa5d9ec6d794369d93def0d78a49c91e61cf19bb4cafe571bb8f6c`，Q2 SHA `2a10ce0fa87e8a757e6d66a740669bcf94a704eecf0646bbf898ad91b402fc08`。主训练/验证代码`work/code/c45_kepler_transit.py` SHA `8a1648a53870bf232afc021c6c98842609887c262fc42b93132243c736adcce9`；Q2质量覆盖修订脚本SHA `60033a10d05890c3a5ee3fd6a36027b1e79d004b6fbab5b6cebf42fd82c4568b`，未改原主代码或已锁模型。

按TIME/PDCSAP_FLUX/ERR有限、ERR正、质量0过滤，Q1拟合1,056行/后时验证378行。Q1拟合段中位数作Q1全部相对通量标度；常数模型训练基线0.99962954。Lomb–Scargle 1—5天搜索自行选1.10232097天，正弦振幅从其sin/cos系数算；1.102是凌日约2.205天的半谐波，说明单正弦的“最佳频率”未必是物理轨道周期。Box Least Squares先4000再局部2001周期点、四时长候选0.10/0.14/0.18/0.22天，仅按Q1拟合似然择2.20502133天、时长0.14天、起点BKJD132.38205、矩形深度0.00612763。原2008联合地面研究发表2.2047299天和0.1685天，不曾作为本机搜索标签或目标。

`work/runs/c45_kepler_q1_locked/q1_train_validation.json` SHA `97065409b1c6fb085154bfdff993a9734378a277b3c1cc59f3df5c72b6a91eaa`。Q1拟合相对通量RMSE常数/正弦/box=`0.00153391/0.00141669/0.00034946`；后时验证=`0.00135578/0.00129455/0.00039576`。验证按**拟合的box相位**划预测凌日17点、其它361点，观察均值0.9937765/1.0000144，box两类各自RMSE0.0005611/0.0003862。分母与坐标在报告，不能由一个总RMSE声称误差在所有时相均等。

取得Q2原件后，第一次按预案`test`在评分前因首个合格点后3天仅30点（少于50）退出，未写成绩。只根据TIME/QUALITY把标定扩为4天55点并明确留档，新脚本首次评分余1,982合格点，`work/runs/c45_kepler_q1_locked/q2_first_test.json` SHA `b67132b1c9280db6ef51933e22fab05cf42117de21dbfc8133a595bc28aae90d`。Q2源4,354原行/有限误差正4,070/质量非0 2,317/最终合格2,037；前4天55仅给无标签中位数1,041,075.0625电子每秒，评分从BKJD178.204至258.406。Q2相对RMSE常数/正弦/box=`0.00155197/0.00143150/0.00056464`；由Q1锁定box相位分预测凌日131点、其它1,851，实际均值0.9940624/0.9998842，box各自RMSE0.0013725/0.0004561。Q2不重训相位、周期、深度，仅零点校准；若时间/数据质量分布改变，低误差不能保证未来永续。

可视化`work/code/c45_plot_kepler.py`按两报告SHA核后画`work/runs/c45_kepler_figure_first/hat_p_7b_q1_q2_measured_and_predicted.png`，图SHA `eaecb3e4539138f13f31b32bd3c9d0f325c7018c5b3f3c993330dc2625ad6318`，**已本机目视**：Q1/Q2蓝绿实际下降大致按Q1红矩形周期排列，但红矩形边缘比观测真实渐入渐出更硬、窗口内观测底部略更低，某些预测事件附近没有点因Q2大段质量缺测。下方折相图Q1/Q2谷形而非完美平底；这直接支持“矩形是找信号的近似，提半径需完整物理拟合与其它观测”，并非测到了无误差轨道。图还有英文字标签，正文用中文逐轴解释，避免把色线当自证。

原报告可由以下命令在**新目录**重做Q1，但那时Q2已被本章打开，重新调参/重测只能称复核或敏感性分析：

~~~powershell
Set-Location '.'
python -I -X utf8 work/code/c45_fetch_kepler.py q1
python -I -X utf8 work/code/c45_kepler_transit.py train --run-dir work/runs/c45_kepler_my_run
python -I -X utf8 work/code/c45_fetch_kepler.py q2
python -I -X utf8 work/code/c45_kepler_q2_amended.py --run-dir work/runs/c45_kepler_my_run
python -I -X utf8 work/code/c45_plot_kepler.py --run-dir work/runs/c45_kepler_my_run --out-dir work/runs/c45_kepler_my_figure
~~~

## 热方程：作者精确生成的一组数学世界，明确不是天文观测

`work/code/c45_heat_science_lab.py` SHA `20eefc3076387ff8fd6f0128201cd7e3391c9dff00f9b98e14048f7d91beddd1`。固定\(u_t=0.15u_{xx}\)、两端0、单条初值\(\sin(\pi x)+0.25\sin(2\pi x)\)，解析式给任意\(x,t\)真值，因此可在**数学模型内部**检查误差。数据只给初值65点、每侧65个边界时间点；两支2241参相同初重、2000步同Adam率0.002，一支只拟这些已知值，另一支每步再从域内128随机配点自动微分算\(u_t-0.15u_{xx}\)残差。两支没有见到内部真解标签。

现行`work/runs/c45_heat_projected_current.json` SHA `f4d06fd970e9abe54796283aa0c60039958aa32c4b56437fbe37ea3a0ac4a848`：数据支初/边训练损失约1.64e−5却整域81×81网格RMSE0.279116、终点x曲线RMSE0.323799、另外60域点残差RMSE1.30623；加物理残差支整域0.009927、终点0.008479、另60点残差0.006941。时间是本次CUDA运行训练约5.92/15.63秒，不能据此与别的机器或完整原PINN论文比速度。真实另一候选**常规显式有限差分**在51空间点、dt0.001、\(r=\kappa\Delta t/\Delta x^2=0.375<1/2\)，终点RMSE `9.71e−5`，在此简单已知方程上明显更准。数值实验显示残差帮助当前小网，不证明任何PINN优于成熟数值解法、收敛到精确PDE或恢复自然真实热扩散。旧100步smoke及首次不含图的2000步报告分别保留在`c45_heat_smoke.json`/`c45_heat_primary.json`，现行数值/图以前述新版本为准；新2000步数值与前一次相同。

同一热解函数族：作者按三项正弦系数均匀生成256对初/终函数，教一个**仅三可训增益的有限谱乘子**在t0.1把\(a_n\)映到\(a_n e^{-0.15(n\pi)^2 0.1}\)。最终学得增益0.862393112/0.553122234/0.263844176对解析相同到约1e−9；64个未见系数组合在64格/把**同组**改128格RMSE约2.72e−10/2.74e−10，但若输入新增从未纳入模型的第4频率、模型丢掉它，128格RMSE0.06593。这是线性频率乘子/谱算子入口，非完整多层非线性FNO或任意网格保证；真初/终函数均由**同一解析式合成**。

再由同一解析热场在65格上投影出第1频率振幅\(a(t)\)的变化当未知，作者预置候选库`1,a,a²`与六条不同初幅轨迹。真导数数据阈值最小二乘仅留下\(a'=-1.480440660a\)，符合模型\(-0.15\pi²\)；往振幅加0.01高斯噪声后直接数值差分使点位导数RMSE0.73252，但同六轨合并的有限库系数约−1.47709。单次随机和库已含真项，不证明算法能在任意噪声/未知变量/未列候选时找到自然定律；`Θ(X)`是哪几个函数由作者先给。时间步缩小在独立读数噪声不同时会使差分噪声按\(1/\Delta t\)放大，这与简洁方程输出本身并不矛盾。

热图`work/runs/c45_heat_projected_current.png` SHA `6862935aa718b5a787e06b7f583def0c5b3c05e04453e92973e9f5602a5c238f`已目视：t0.25/1的数据支橙线过高，残差支蓝线紧贴黑解析线。它是由**数学真值**绘出的作者模拟比较，不能出现在望远镜数据段里冒充真实温度测量。可运行：

~~~powershell
Set-Location '.'
python -I -X utf8 work/code/c45_heat_science_lab.py --steps 2000 --out work/runs/c45_my_heat.json
~~~

最后证据分层：严格热方程唯一性可由差值\(w\)的能量\(d\int w²/2dt=-\kappa\int w_x²\le0\)在完整初边值下推出；这不适用于只在有限点训练到小残差的PINN。精确谱乘子对指定三模/解析数据是**模型内**计算，不是天气或生物预测。Kepler跨季验证是**观测匹配**，2008外部地面光度/径向速度支撑的是另一种行星识别层级。天气和分子真实系统还须查数据同化、外部可测量证据与实际任务结果，不能由这两条练习扩张。
