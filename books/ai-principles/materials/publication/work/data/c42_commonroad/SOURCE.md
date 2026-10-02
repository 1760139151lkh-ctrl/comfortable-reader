# C42本机道路场景：来源、内容与不能冒称的对象

2026-09-25从`CommonRoad/commonroad-control`公开仓库固定提交`0b672f9d2164cd2b2698fbfcd5046ba492c75ccd`取得**两份原XML例子**与仓库`LICENSE`，用`work/code/fetch_c42_commonroad.py`按精确字节数和SHA-256核验并写`manifest_v2.json`。原仓库与[2017 CommonRoad论文](https://portal.fis.tum.de/en/publications/commonroad-composable-benchmarks-for-motion-planning-on-roads/)的目标是描述可复现道路规划问题；不能因为文件有真城名，就说这里有用户汽车或原始传感器驾驶记录。

| 原件 | 上游XML声明 | 大小与SHA-256 | 本机可见 |
|---|---|---|---|
| `DEU_Backnang-9_1_T-1.xml` | Scenario Factory，OpenStreetMap道路+SUMO交通，2020-08-23，步长0.1秒 | 177156字节，`f642ad66ecdb73ef4e15c57da201773fe739f6f707a3cf64c4efb5c265afc361` | 16 lanelet、9动态障碍、1规划问题；目标仅给第33步时间，未给目标车道 |
| `ITA_Foggia-6_1_T-1.xml` | Scenario Factory，OpenStreetMap道路+SUMO交通，2020-08-23，步长0.1秒 | 329353字节，`2563a7dd4eedb60ef460d37afe4c0b718510092289f81341eb967bd3f741b8b4` | 101 lanelet、10动态障碍、13交通标志、6路口、目标车道79895 |
| `LICENSE` | 上述仓库发布的BSD-3-Clause条款 | 1612字节，`60a4cd8990098ce0cca20f090ba4af5906b21d3cb43c94e3bc987a2b62d44cb6` | 仓库说明，不把它自动当OSM底层数据的唯一许可证 |

可复核固定URL：`https://github.com/CommonRoad/commonroad-control/tree/0b672f9d2164cd2b2698fbfcd5046ba492c75ccd/scenarios`，原XML在同仓库`scenarios/`，代码中的`raw.githubusercontent.com`地址包含相同提交。上游在文件头明确写了OSM和SUMO身份：[OpenStreetMap自己的许可页](https://www.openstreetmap.org/copyright)说明其源数据库为ODbL，并要求归属；这里不据仓库BSD3一句话推断派生道路几何重新发布的具体权利。本书仅在用户自己的电脑保留这两份用于研究与复现，不对外重新发放地图数据库作版权保证。

本章选Foggia作为主场景，因为它有真正的空间目标且道路、模拟车辆和起始车速可以明确读取。`work/code/c42_commonroad_scene.py`不依赖一份外部“默认导航路线”：它先找初态落入且能循lanelet后继图抵达目标79895的候选；本文件恰有唯一的可达候选82212，再沿`82212→78050→83276→79895`拼两边界的中点。该作者选路线长147.101米，初态投影在21.258米、初速度9.189米/秒。`work/runs/c42_scene_first/report.json`与`road_and_route.png`为当前作者生成结果，图已目视。这条红路线不等于上游原规划器或真实驾驶者的唯一选择。

动态障碍是XML提供的**SUMO模拟轨迹**，不是人开的车在真实道路上的同步观测；多数障碍有初态加33步（0至33，即3.3秒）可用，少数到36步。任何把这些障碍冻结、保持末速度、预测更长时刻或根据本书自车动作改它们的做法都须另标作者假设；不能悄悄把不存在的第4秒后真值补出来。场景只给二维路界、目标、模拟障碍和部分车状态；没有相机/激光原帧、轮胎附着与摩擦估计、真实制动执行、道路实测误差或车载安全证书。因此本章能实测的是**本机规定模型与已给场景记录范围内**的开环/闭环、扰动/延迟和约束检查，不是现实上路许可。

## 本书后来加入的量和文件

`work/code/c42_bicycle_control.py`里的车长宽、轴距、执行转角偏差、速度控制、纯追踪前瞻和额外停驶车均为本书作者规定；上游XML没有要求使用这辆作者自行车模型，也没有给真轮胎/刹车试验。固定原件上的路线分析见`work/runs/c42_scene_first/report.json`，车辆控制见`work/runs/c42_control_verified/report.json`及`trace.json`，看过主试验后的更大偏差/延迟见`work/runs/c42_control_sensitivity_verified/report.json`。后两份12秒路界指标仅对静态路面及作者车辆有效，原动态交通只在共同可见的0—33步抽样比较。

`work/code/c42_steering_imitation.py`在同一作者车模型上自己生成40训练/10验证/两组各10测试的初态条件，仅向作者纯追踪器索取转向标签，所有图像/人类方向盘/新道路资料均没有进入。验证后先锁模型与方案`work/verification/C42_imitation_pretest_decision.md`才打开第一次测试`work/runs/c42_imitation_pretest/test.json`；后来的第8个宽条件放大图属事后解释。`work/code/c42_lateral_state_filter.py`使用控制轨迹`closed_bias`的可知仿真真横偏，作者加σ0.4米白噪声、自己设过程噪声，现行`work/runs/c42_filter_verified/report.json`是合成噪声估计结果，绝无真实位置传感器；旧`c42_filter_first`初稿双用首条观测已修。各步范围、命令和不可跨越的证据边界详见`work/verification/C42_control_imitation_protocol.md`。
