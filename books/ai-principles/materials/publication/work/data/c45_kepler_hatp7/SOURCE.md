# HAT-P-7 的 Kepler 测光：原件与本章问题

2026-09-25，从[MAST Kepler公有光变目录](https://archive.stsci.edu/pub/kepler/lightcurves/0106/010666592/)按 KIC 10666592 的九位补零号取得固定 Q1 长采样文件 `kplr010666592-2009166043257_llc.fits`。本地192,960字节/SHA-256 `37faa7dc82fa5d9ec6d794369d93def0d78a49c91e61cf19bb4cafe571bb8f6c`。文件主头标 NASA/Ames、Kepler Photometer、QUARTER=1、DATA_REL=25、FILEVER=6.1；LIGHTCURVE扩展给时间`BJD−2454833`、时间尺度TDB、相对采集日131.5—165.0、原孔径`SAP_FLUX`、经Pre-search Data Conditioning处理的`PDCSAP_FLUX`（电子/秒）、误差与`SAP_QUALITY`。共有1,639个时序行，有限TIME和PDCSAP_FLUX 1,624行、再要求质量标记0则1,434行。原始像素、天体光谱、真实行星标签不在这个FITS里。

[NASA Kepler处理流程原说明](https://keplergo.github.io/KeplerScienceWebsite/pipeline.html)第PA/PDC节说明：所谓FITS里的“raw SAP”已经经过像素校准和孔径光度提取，PDC再处理仪器/航天器系统误差；行星搜索在其后进行。每个季度航天器转90度、目标落在不同CCD，孔径定义可能改变。因此即使下载两个季度，也不能把两组原计数值当同一无漂移仪器标度；该测光已加工，不是未经人和程序处理的自然真值。

[Pál等2008原论文](https://arxiv.org/html/0803.0746)§II—IV明确 HATNet 地面监测、后续光度及 Keck 径向速度共同分析，发表周期约2.2047299天、凌日总时长约0.1685天，并检验了混合食双星等假阳性；Kepler飞船是2009年才取得本章Q1记录。故本章用这个星体作**对已经知道的天体现象重新形成和检查一个简化信号规则**，而非“AI从零发现HAT-P-7b”。2009 Borucki等Kepler相位曲线是后续研究，不应倒写成2008原发现。

预先切分与比较范围在 `work/verification/C45_kepler_pretest_decision.md`。本章仅本机按需读取公开FITS、存汇总和图；不上传。若后来拿同季/跨季数据改变模型，必须写成事后修订而非再造第一次测试。

Q1锁定后才从同一官方目录取得Q2长采样`kplr010666592-2009259160929_llc.fits`，466,560字节/SHA-256 `2a10ce0fa87e8a757e6d66a740669bcf94a704eecf0646bbf898ad91b402fc08`，头标QUARTER=2、原表4,354行。有限`TIME/PDCSAP_FLUX/ERR`且误差正有4,070行；质量标志非0有2,317行，合格后总2,037行，远比Q1缺得多。原Q2第一个3天全无质量0行，故最初预定零点窗因只有首个合格时间后三天30行而主动退出；在任何Q2得分出现前仅依据TIME/QUALITY把窗改四天55行，评分1,982行。该动作和首次成功测试分别保留在预试决定与`work/runs/c45_kepler_q1_locked/q2_first_test.json`，并不代表该星另一种资料形式也同样可靠。Q2原通量中位数约1,041,075电子/秒，Q1训练约1,034,000电子/秒，实际数见报告；原绝对流量因季度CCD/孔径不可直接连成一条无漂移物理曲线。

本目录原件可由`python -I -X utf8 work/code/c45_fetch_kepler.py q1`和`q2`分别重取；程序遇已有文件不覆盖，只查FITS起始与SHA并报告。公开可访问不等于文件头明确给了新许可；本书署名NASA/Ames/MAST并限本机教学研究，不推断其它再分发权利。Q1/Q2主图只画质量0的PDCSAP光变和事先Q1锁定模型，读者应知道原FITS既没有“是否行星”真值，也没有本机可独立判别的径向速度。
