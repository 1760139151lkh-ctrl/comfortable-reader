# C39 本机宿主执行实验的对象与边界

数据全由作者于2026-09-25在此活动包 `work/runs/` 内构造，无外部账号、私有用户文档、网络请求、付费服务或其他项目文件。代码[`work/code/c39_host_execution_lab.py`](../../code/c39_host_execution_lab.py)只允许新输出目录解析后仍处于本包 `work/runs/`；示范文档是SQLite里编号`lesson_note`的一行，初始内容`Start\n`。受信任的**实验驱动器**（不是C38模型）给主体`author_demo`、逻辑操作ID和请求，宿主授予仅`append_note`、仅此文档、80字节单次上限、3次新写预算。它拒绝`delete_file`、其它文档、同ID不同参数、旧版本或超预算。这个Python应用层判断不能当作操作系统沙箱或真实用户身份鉴定：直接运行任意Python的人仍有本机账户本来的权限。

现行主运行在[`work/runs/c39_host_lab_inspectable/report.json`](../../runs/c39_host_lab_inspectable/report.json)（SHA `07ebe80b562ba66040e72ae4422ab30d97bef7d7f6685f6678918ce2ba0e56c4`），程序SHA `a767b78eef372c23600dc76b22d7055e3e0b404ef3675a58d82c72e49e62b3e5`。实际同一数据库[`local_ledger.sqlite`](../../runs/c39_host_lab_inspectable/local_ledger.sqlite) SHA `c44de5a743618ee27855ed13f01b0b54d3692b0a0cbb6659887b12ff3c77e021`，独立导出的最终[`final_note.txt`](../../runs/c39_host_lab_inspectable/final_note.txt) SHA `e18ebeed0a9d7a3855c5b6511011c355bc045d11402751144c0c3a5f8b6cd056`；报告内两个哈希与磁盘复核一致。此前 `c39_host_lab_first` / `c39_host_lab_current` / `c39_host_lab_bound_keys` 是加同文字新ID、远端参数绑定和读者可用`inspect`入口之前的独立试运行，保留而非覆盖，但读者以现行主运行解释。

本次故障不是伤害真实文档：驱动器先用**独立子进程**跑同一受限SQLite请求，等它写下到达故障点的marker后只杀这个自己启动的子进程。`op-before`在本机事务COMMIT前终止；SQLite重开后文档仍第1版、无这条operation，原ID重试才提交第2版。`op-after`在COMMIT后、调用者收到答复前终止；重开时文档已有第3版且operation/回执在表里，同ID同参数重试只取原回执、没有第四行。父进程所见都是“无结果/子进程结束”，两个真实数据库状态不同。前者展示本机回滚，后者展示本机**已提交但回执不可见**；没有测试实际断电、远端网络或任意进程威胁。

另有三个范围不同的子实验：`same_id_concurrent.sqlite`里两线程同ID并发只有一项改动且另一条读到回执；`two_distinct_intents.sqlite`允许两不同ID、相同文字在满足逐次版本/预算时真正追加两次；`cancel.sqlite`表明派发前取消可以阻止新执行，已提交后再申请取消不会把旧提交擦掉。`two_worlds/`三个**物理分开的本地SQLite模拟服务**对应“未执行”、“已执行回执丢失”和“服务端不去重”；客户可见JSON字节哈希一致，远端首次状态0/1/1，带同ID重试后分别1/1/2。这个模拟不是跨数据库原子事务或真实远端服务认证：只有模拟远端明文实施ID及参数匹配时才有其局部去重语义。

C38训练的小模型仅在乘法/标题等**只读**工具上验证过调用与返回使用，未曾在本章从写操作资料上训练，也不生成本实验的`append_note`授权。作者构造写请求以研究宿主职责，不将SQLite里的三次追加归功模型。进一步的最终产物验收须读实际文档内容、版本、三条`operations`与每条回执的内容SHA，而非接受模型说“完成”。
