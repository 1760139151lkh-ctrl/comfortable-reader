# 本机 CartPole-v1 实验环境

记录日期 2026-09-24。本项目没有下载个人数据或Atari ROM。为了在真实可运行的标准控制环境里核DQN与学模型规划，本机于包外工作区建立 Python 虚拟环境：

- 路径：.\work\envs\c24-gym
- 创建：Python 3.13.14，venv --system-site-packages；继承本机已有 PyTorch 2.11.0+cu128 与 NumPy 2.4.6；另安装 Gymnasium 1.3.0、cloudpickle 3.1.2、farama-notifications 0.0.6。
- Gymnasium 的 pip 元数据为 MIT License；[官方CartPole说明](https://gymnasium.farama.org/environments/classic_control/cart_pole/)列动作、四维观测、默认每步+1和500步截断。默认奖励不同于其可选的sutton_barto_reward=True（非终止0、物理失败−1）；本书本机实验用默认奖励，不冒充1983原作者的完全相同任务。
- [Gymnasium官方基本调用](https://gymnasium.farama.org/introduction/basic_usage/)用 reset(seed=...) 和 step(action) 返回 observation、reward、terminated、truncated、info。物理终止与时间截断在训练代码中分开：前者目标无后继项；后者结束采样回合，但其底层状态并非物理吸收态。
- [PyTorch官方DQN CartPole教程](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning)只用作当代API与操作对照；本书重新编写训练代码、独立留验证/测试起点，并明确与Mnih等2013/2015的Atari像素卷积实验不同。

从包根复跑当前版本时，使用虚拟环境中的 Scripts\python.exe。若环境路径丢失，可在根工作区重新建隔离环境并安装 Gymnasium 1.3.0；本章程序先核当前导入版本及现行结果，不能只凭本文件断言不同设备结果字节相同。
