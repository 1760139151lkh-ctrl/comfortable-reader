# C41真协议与受限技能/委派：执行范围

## 可复现命令

在活动包根目录用当前本机主Python运行MCP与教学技能加载器。每次换一个**尚不存在**的`work/runs/`输出目录，旧结果不覆盖：

```powershell
python work/code/c41_mcp_evidence_client.py --out-dir work/runs/c41_mcp_my_run
python work/code/c41_skill_loader.py --out-dir work/runs/c41_skill_my_run
```

第三支用本机隔离Python环境运行A2A；若环境尚未建立，从根工作区`.`新建venv，再按`work/data/c41_protocol_lab/a2a_requirements.txt`安装。这里的两个服务只听`127.0.0.1`、自动选本机空闲端口，完成或异常时都终止子进程，已用系统进程列表核无残留：

```powershell
& '.\work\envs\c41-a2a\Scripts\python.exe' work/code/c41_a2a_evidence_client.py --out-dir work/runs/c41_a2a_my_run
```

上述作者正式运行对应`work/runs/c41_mcp_second/`、`c41_skill_verified/`、`c41_a2a_second/`。三份报告都保留可读JSON，源码及原始SHA见[来源卡](../data/c41_protocol_lab/SOURCE.md)。

## 真MCP SDK2.0.0，2026-07-28版

本机`Client(stdio_client(StdioServerParameters(...)), mode="auto")`确实启动**另一Python进程**，通过真实stdin/stdout的JSON-RPC协议同官方`MCPServer`对话；协商/发现得到`protocol_version=2026-07-28`及`tools/resources/prompts`能力。列出的工具是`multiply`和`read_note`；作者客户端调用前者`12×13`得到结构结果156，传`a="twelve"`被输入模式拒绝，`isError=true`。`read_note`第一次读C39原笔记的演示拷贝，正好四行。客户端还实际列/读`evidence://c40/retrieval-summary`资源（1759段、12问题、R@5=9及报告SHA），列/取`check_note`提示模板。随后只在演示目录改笔记拷贝，第二次`read_note`得到`isError=true`和源改变消息；C39原笔记SHA未改。服务子进程在客户端离开context时结束。

这证明的是SDK在这台机器上处理2026版发现、结构参数、工具/资源/提示和错误状态；工具选择全由**作者客户端程序**固定，并未让C38小模型学习或自主挑工具。stdio子进程用同一Windows账户，`read_note`仅有指定路径与SHA保护；既无真实用户身份认证，也非OS沙箱、OAuth或全MCP协议一致性测试。当前SDK2.0.0对`Client(StdioServerParameters(...))`直接形式实测不接受，公开`stdio_client`包装形式成功；若读者更新SDK，仍以实际`client.protocol_version`和返回对象判断，不照旧教程混用`initialize`。

## 本地SKILL.md与按需加载

教学源目录`work/data/c41_skill_demo/check-book-source/`含`SKILL.md`与一份支持规则，只指导“核发布SHA、读源段、缺证时停”，不是本机Codex已安装技能。加载器复制这两个文件到新运行目录，首先只解析YAML开头的`name/description`并计算原字节SHA/长度清单，此时没有把后续步骤和支持文件文字当作已加载工作流；作者**明确选择**后才读两文件、逐件核清单哈希。故意向支持文件拷贝加字节，旧清单下再次加载报`resource changed since manifest`，原教学源文件不动。最后报告：metadata前只见简述、选后2份内容进入小程序、改字节后拒绝。

这只模拟一个文件加载器的关键字节/来源条件。Codex或MCP Skills扩展的真实宿主还要处理来源身份、按需取远端`skills/list/get`/`resources/read`、审批和注入风险；本程序没有安装技能、改变当前任务指令、运行支持脚本或授予工具。`SHA一致`指字节版本一致，不代表内容可信、事实正确或用户批准。MCP Skills Final扩展作为正在演进的另一种分发方式另由规范说明，不拿本程序冒充其互操作测试。

## A2A v1.0两端点与主作者验收

隔离环境的官方`a2a-sdk==1.0.2`真正启动两个**本机独立HTTP服务**。各自Agent Card使用v1.0的`supportedInterfaces`和`protocolVersion=1.0`，都自报同一个`c40_retrieval_count`技能。作者客户端由两张卡建立A2A客户端，对每端各送一次`SendMessage`，固定请求预算2；两项都返回`Task`，状态`TASK_STATE_COMPLETED`，都有名为`retrieval_result`的`Artifact`。这正好把“对方说已完成”与“产物是否可信”分开：

| 端点 | 卡片自报技能 | Task状态 | Artifact中的R@5 | 来源SHA | 主作者检查 |
|---|---|---|---:|---|---|
| `source_reader` | `c40_retrieval_count` | completed | 9/12 | 与真C40报告相同 | 接受这条报告内结论 |
| `unverified` | `c40_retrieval_count` | completed | 12/12 | `unverified` | 拒绝，不拼进最终回答 |

服务端的第一模式从真C40报告读取并验SHA；第二模式是作者故意给的似是而非结果，不是“另一个AI独立看错”。主作者的接受谓词为`terminal_complete AND source_sha==expected_sha AND reported_count==count_in_source`；**两任务都完成，却只有一份产物过验**。这种验收没有因多数意见或角色名而改变事实。任务由两个不同进程各自处理，仅访问同一只读报告；没有验证恶意执行者隔离、权限认证、长期任务持久化或取消后的远端副作用。客户端在固定两次调用和终态后停止，最后关闭两个子进程；它没有把一份错误报告传给另一端反复自洽。

首装默认`protobuf==7.36.2`使`a2a-sdk==1.0.2`真请求在其`FieldDescriptor.label`代码处失败；仅将隔离venv固定到`protobuf==6.33.6`后重跑成功。这是实际软件版本约束，不是协议原理失败，也不应推广为A2A v1.0不可用。另本地两端点没有训练模型，所以本章的协议成功/失败均**不抵扣**“模型能发现对方、会选择委派、能合并数学/代码研究成果”的学习能力目标。
