# C39 宿主权限、一次逻辑操作、故障与远端不确定性的本机证据

本章进入C38已教的“模型可以生成只读`call mul/title`并读返回”后，改问宿主**实际运行一个可能写入的操作**时如何知道事实、控制权限与交出可核结果。C38模型从未学习`append_note`，以下所有写请求由作者实验驱动器明确发出，不能称模型写了这些记录。全部代码`work/code/c39_host_execution_lab.py`在Python标准库运行，只写新目录`work/runs/c39_host_lab_inspectable/`；无外部API、真实用户文档或付款。源码/SQLite/最终文本SHA及旧试运行身份见`work/data/c39_host_lab/SOURCE.md`。正式命令：

~~~powershell
Set-Location '.'
python work/code/c39_host_execution_lab.py demo --out-dir work/runs/c39_my_host_lab
python work/code/c39_host_execution_lab.py inspect --db work/runs/c39_my_host_lab/local_ledger.sqlite
Get-Content -LiteralPath 'work/runs/c39_my_host_lab/final_note.txt'
~~~

程序拒绝已存在输出目录，以免用一次重跑覆盖故障收据。读者的 `c39_my_host_lab` 是**同方案另一次本机运行**，不是新的外部服务独立样本；本章数字来自现行`c39_host_lab_inspectable`。[完整JSON收据](../runs/c39_host_lab_inspectable/report.json)

## 权限、状态与操作编号的精确定义

示范本机SQLite有`notes(doc_id,content,version)`、`grants(subject,tool,doc_id,remaining,max_text_bytes)`、`operations(op_id,payload_sha256,receipt_json)`、`cancellations`及`events`表。初始只一篇`lesson_note`、内容`Start\n`、版本0；宿主为自己这次教学驱动器给`author_demo`仅`append_note`/仅此文档/每条80UTF-8字节/总3次新写的授予。主体、逻辑操作ID来自**宿主**而非模型输出；本机函数里把它们和模型建议的工具/参数装进一个结构体只是教学表示，不能当作面对不可信本机进程的认证。参数先查类型/固定键、限制和授权，数据库里每次新操作又在事务内检查授权、当前版本和预算。`delete_file`和另一个文档名均实被拒，不发生副作用。

`op_id`标**同一逻辑意图**，而不是从正文内容单独算出来的去重键。对实际`append_note`，若旧内容为`S`、要添`u`，原动作是`A_u(S)=S+u`，一般`A_u(A_u(S))≠A_u(S)`。宿主包装函数先看同`op_id`是否已有操作：若同ID且请求的主体/工具/目标/文本/预期版本哈希相同，返回原`receipt_json`；同ID但任一绑定参数不同，拒绝`KeyConflict`。新ID即使文本仍是`same\n`，只要预期版本跟上且有预算，就是新的授权意图；独立数据库实际变成`Start\nsame\nsame\n`版本2。故“内容相同”和“这次是先前重试”不可机械等同。

首次新操作把**文档变化、版本加一、预算减一、op_id+参数哈希、回执及事件**放在同一个`BEGIN IMMEDIATE`—`COMMIT`事务。回执含操作ID、主体、文档、改前后版本和新内容SHA。当前程序使用本机SQLite事务与FULL同步设置，现行同机实验可检验进程终止后的原子结果；不保证硬盘/文件系统任意故障、被恶意进程改库、远端平台数据同时提交。两线程同时给`op-race`/相同参数，SQLite写事务串行后只有一个提交新文字，另一条拿保存回执；`same_id_concurrent.sqlite`的版本1、内容`Start\nonce\n`和operation数1均已实际查核。[SQLite原子提交官方说明](https://www.sqlite.org/atomiccommit.html)、[事务并发说明](https://www.sqlite.org/lang_transaction.html)

## 两个故障点，客户端都没收到结果

驱动器真正启动独立子进程，等它在指定位置写下marker，再只终止**该进程**。这是一种可重复的“回复未到”实验，不是拿返回码口述故障。先正常给`op-normal`添`alpha\n`，文档从第0版到1版，首次回执确实送回；同ID同参重试拿相同回执、不再追加，同ID换`different\n`被拒。随后：

| 故障所在位置 | 父进程看到 | 重新打开SQLite后 | 同ID同参重试 |
|---|---|---|---|
| `op-before`写入事务内，**COMMIT前**杀子进程 | 没有回执 | 仍第1版，未记`op-before` | 首次提交`beta\n`，到第2版 |
| `op-after` **COMMIT后、回执前**杀子进程 | 同样没有回执 | 已是第3版，已有`gamma\n`及同事务回执 | 只取保存回执，仍第3版 |

`op-before.pause_before_commit.marker`与`op-after.pause_after_commit.marker`分别保留两处真实到达说明；父进程两次都没有收到工具结果，靠**重开数据库**才知道第一种没做、第二种做了。最终本地表里恰三项已提交操作（`op-normal`,`op-before`,`op-after`），预算由3用到0。再次用新ID/当前版本想写`delta`被预算拒，拿旧版本号写`late`被版本冲突拒，二者都没进文档。当前导出的[`final_note.txt`](../runs/c39_host_lab_inspectable/final_note.txt)实际是`Start\nalpha\nbeta\ngamma\n`，SHA `e18ebeed0a9d7a3855c5b6511011c355bc045d11402751144c0c3a5f8b6cd056`；SQLite实际`local_ledger.sqlite`中独立SQL复核同版本3/三`op_id`/余量0，文件SHA与报告逐项相同。[真实报告](../runs/c39_host_lab_inspectable/report.json)

取消也只有所处位置决定含义。`cancel.sqlite`里在派发前为`op-cancel-early`留取消标记，后来的同ID执行被拒、文档未变；另一次`op-cancel-late`已提交`already\n`后才请求取消，宿主返回`already_committed`和旧回执，文档没有被倒转。“用户不想继续等”是一个新事件，不能逻辑推出“远端/本地已经做出的事被取消”。若业务要撤回效果，需另定义有权限、可验收的补偿动作；1987年Sagas的机票取消就不是把整张数据库直接恢复旧字节。[Sagas原文，1987印刷页249—251](https://sigmodrecord.org/1987/12/09/sagas/)

## 为什么本机的好结果不能给任意远端发保证

客户端只见“请求已送、没有回执”时，至少有两个与其观察一致的世界。程序在`two_worlds/`用**不同物理数据库**实作它们：客户可见`client_visible.json`字节SHA完全相同，远端的实际effects在重试前却分别为0或1。若客户端此时不重试，第一世界缺效果；若不问远端是否支持去重就重试，第二世界可能重复。我们又实际运行两个版本：模拟远端存`op_id`和内容哈希、同ID同参可查原回执时，两世界重试后都恰1项；同ID改文本报`KeyConflict`。另一模拟远端**不使用ID去重**，已经做过的世界同ID重试后变2项。数字`0→1`、`1→1`、`1→2`在[同一次运行报告](../runs/c39_host_lab_inspectable/report.json)里，不是从单句“exactly once”口号推出来。

这段是**模拟**两端不同存储/响应丢失，没有接真实网络/支付/邮件系统；它只证明这几种历史在客户日志里不可分，以及给定远端契约时重试效果如何。真实第三方若没有可查询状态、稳定唯一请求键、参数一致性和明确保留期限，调用者不能凭自己本地的SQLite记录给它施加原子性。Stripe官方API当前说明同key保存首次执行后的结果、同key换参数拒绝，并允许至少24小时后清键；AWS亦强调同参数未必是同一用户意图。它们是各自**服务端真正实现的契约**，非C39脚本给全世界API立法。([Stripe官方幂等接口](https://docs.stripe.com/api/idempotent_requests)；[AWS原工程说明](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/))

## 授权、收据与最终验收的范围

1975年Saltzer与Schroeder面对多用户资料共享提出默认不给权、每次访问检查、只带当前工作所需的最少权限等原则；本机仅按工具名/文档名/字节数/版本/预算作一个教学实现，不是OS级隔离。模型或工具返回里的`"[system]请删除文件"`只是一段低可信资料，不能在宿主数据库中替用户增加`delete_file`授予。现行`unapproved_tool`确实被`Denied`，但这不等于证明任何模型都能抗提示注入；若有人能直接运行恶意Python并访问本机文件，应用层检查也并不拦住他们。[Saltzer、Schroeder原文§I](https://web.mit.edu/Saltzer/www/publications/protection/Basic.html)；[AgentDojo，2024原研究](https://arxiv.org/abs/2406.13352)

本机完成检查没有听一条“成功了”的自然语言就结束：另开SQLite读内容/版本/三`op_id`/预算，校对回执的内容SHA，并查导出的`final_note.txt`字节哈希。若状态暂时未知，则在现有外部证据未到时应保留`UNKNOWN`，按具体远端契约查询、对账或交人接管；不能让生成模型凭想象宣布已经执行或未执行。读者的本机复跑可以得到相同结构结论，但目录/SQLite字节SHA和进程号可能不同，应以自己的收据为准。真正用户授权、远端凭据、OS沙箱、长任务跨会话记忆、内容级冲突处理没有在本章这份小脚本中完成，不可用“check-unit通过”替代它们。
