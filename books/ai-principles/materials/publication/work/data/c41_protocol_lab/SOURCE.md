# 第四十一章协议、技能与委派实验来源

研究和实际运行日2026-09-25。三支程序只读取本书自己的教学资料：C39笔记、C40检索报告或C41技能示例；写入只在新建的`work/runs/`教学目录。没有连接用户账号、外部MCP/A2A服务器或真实用户文档；没有派生任何AI子代理，不能称多个语言模型协作。

## 固定上游内容

- C39原笔记：`work/runs/c39_host_lab_inspectable/final_note.txt`，SHA-256 `e18ebeed0a9d7a3855c5b6511011c355bc045d11402751144c0c3a5f8b6cd056`，内容为`Start/alpha/beta/gamma`四行。MCP演示只复制它到自己新建目录，然后**故意改拷贝**来检验源变动拒绝；原件前后SHA不变。
- C40已登记的真实书段检索报告：`work/runs/c40_book_retrieval_stable_a/report.json`，SHA-256 `47a31f6459b8fc0339727210c51026a955c4cccc8b82b2bd0fbfbf881ffff161`。它来自导读+前39章1759段、作者预选12问题，指定锚点R@5为9/12。A2A可信执行者实际重读并算这个文件的SHA；另一执行者由作者故意造错误的`12/12`与`unverified`源标识，作为待拒绝产物。两个A2A服务是同一套教学代码的**确定性过程**，不读用户资料或调用LLM。

## 环境、代码和真实结果

- 本机主Python 3.13.14的`mcp==2.0.0`，官方2026-07-28规范的SDK实现。真正MCP stdio服务器/客户端代码在`work/code/c41_mcp_evidence_server.py` SHA `512ce08489b78c39b0bb62343e2ca395b2e7b8321b271816bf0ec1b0efab1be3`、`work/code/c41_mcp_evidence_client.py` SHA `1062bfb87b0849f49a31c9eac513146115d01072a57c37728b4a98d63950f67c`。完成运行`work/runs/c41_mcp_second/report.json` SHA `d78dcff96a1d7d9766eba7df956badfe0c5b6120c37c3c5b712d9b2b29ac2d8d`。Python SDK v2官网本次页面说`Client(StdioServerParameters(...))`可直用，但本机2.0.0当前类型路由不接受，第一次预运行在协议连接前报类型错；最终源码用公开的`stdio_client(spec)`交`Client`，真另进程stdio发现/调用成功。该差别属本机安装版实测，不推广到所有未来SDK版本。
- 教学用`SKILL.md`放在`work/data/c41_skill_demo/check-book-source/`，**未**安装到`~/.codex/skills`、插件或MCP扩展。`SKILL.md` SHA `43aa6e9f6119d4e13f44177506c0abeff17d54950160ad4a8cd44dfdfb623aae`，支持文件`references/source-rules.md` SHA `3a0a240d2f89e4ab65f3761672cb1a019893d3c33bb3acb41d2f4f7fbe84c576`。独立小加载器`work/code/c41_skill_loader.py` SHA `469b89f64bd2050b993f4049d955b2c7dd9a36f90f6b2054f7755b5e791236a3`；运行`work/runs/c41_skill_verified/report.json` SHA `cc26523ec9292fa5318f644ed6f88194cd8de6b5a14b0e59db87ea72f7bcbb50`。作者显式选择后才读取工作流全文/支持文件；故意改**拷贝**的支持文件，旧manifest加载被拒。初次试运行拷贝目录名与frontmatter的`name`不同而如期失败，已改成同名目录；另改frontmatter解析仅读到闭合`---`，不在metadata阶段把正文解码进内存。SHA计算仍读原始字节以建清单，但不是模型已加载指令。
- 本机隔离目录`.\work\envs\c41-a2a`：Python 3.13.14，`a2a-sdk==1.0.2`、`protobuf==6.33.6`、`uvicorn==0.54.0`、`starlette==1.7.0`、`httpx==0.28.1`。顶层重建依赖见`a2a_requirements.txt`；MCP依赖见`mcp_requirements.txt`。A2A服务/客户端源码`work/code/c41_a2a_evidence_server.py` SHA `ead35b4e56946464c78b82bacde0cddd8f158ade308623139fa451c27756d49c`、`work/code/c41_a2a_evidence_client.py` SHA `20cb992e71a9dc5b73ab91e7dfe972cb0f4fd0b36dbd76d7873d5091a2ca5d7c`；真两个本机loopback服务结果`work/runs/c41_a2a_second/report.json` SHA `14d596c117b2b541d35d03bd02d09d2afbd34e0074a2d8126d1b09f6cd2f7659`。初装默认protobuf7.36.2与官方A2A SDK1.0.2的旧`FieldDescriptor.label`调用不兼容，第一次真实HTTP请求返回服务内部错误；隔离环境固定protobuf6.33.6后完整A2A v1.0交互成功。该修复和错误有实际运行/官方protobuf弃用公告依据，未改全局Python。

MCP Skills扩展已在2026-09-13成为Final，但官网同时说SDK/宿主支持仍在实现；本机SKILL.md加载器没有宣称通过该扩展的`skills/list/get`或使用Codex内部技能路由。A2A卡片里的`AgentSkill`只是该端点自报的任务类别/示例，与可加载完整`SKILL.md`不是同一对象。三支实验的授权范围仅限本机过程所读取的指定源和新输出目录：没有OAuth、恶意本机进程隔离、远端任务持久化或用户对技能的真实批准。版本/能力和结果须按对应报告分别引用。
