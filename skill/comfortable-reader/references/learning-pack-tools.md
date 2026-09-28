# CRLearn 工具与安装版交接

本文件在明确的学习增强或阅读器升级范围内读取。普通导入不需要建立运行服务或修改应用。

`scripts/learning_pack.py` 的两个真实子命令是 `build` 和 `audit`。输入是 UTF-8 声明式 JSON；输出 `.crlearn` 是本项目约定的 JSON 数据格式，不是通用电子书标准，也不是可执行脚本。两个命令都要求实际 `--epub` 和 `--root`，派生资源另传 `--derived-root`。工具只读 EPUB 与资源，输出不自动进入正式书库。

最小规格包含 `package_id`、`revision`、`chapters`、`assets`、`href_assets`、`activities`、`source_claims`。`build` 从当前 EPUB 读取书籍标识与 SHA-256，补齐资源的字节数和 SHA-256，再执行与 `audit` 相同的检查。一个 EPUB 有多个标识时，规格须明确选择真正的 `book_uuid`，不能按标题猜书。

```json
{
  "package_id": "one-calculation",
  "revision": "1",
  "chapters": [{"id": "chapter-1", "href": "chapter.xhtml", "assets": ["program"]}],
  "assets": [{"id": "program", "relative_path": "code/example.py", "kind": "code", "title": "核对这一步计算"}],
  "href_assets": {"code.xhtml": "program"},
  "activities": [],
  "source_claims": []
}
```

`chapter.xhtml`、`code.xhtml` 是示例，必须替换为当前 EPUB 中的真实地址。活动至少使用已有的 `id`、`chapter`、`entry_asset`、`runtime`；这里的 runtime 是逻辑名称，不能夹带 shell 命令、cwd、解释器路径、环境变量或 `trusted=true`。资产路径相对于所选内容根，拒绝越界、Windows ADS/设备名、符号链接与重解析点。派生帧用 `root: "derived"`，保留原件、`derived_from`、解码时间戳与处理说明。

来源条目保留原始 HTTP(S) 链接、书中对应解释和可核对的引用范围。缺失的作者、版本或页码如实标记，不补造。书中解释与原文摘录、译文、教学构造分别标识。历史时间相邻不自动建立影响关系；学习依赖绑定正文使用点。

审计会检查 EPUB 身份、资产哈希、ZIP 路径、章与资源引用、部分媒体签名及来源协议。它不能证明数学正确、音视频已在实际窗口播放、代理转码没有教学偏差，或任意代码已获得 OS 隔离。`structurally_verified` 和 `execution_authorized: false` 应按字面理解。

安装版的 `learning-bindings.json` 属于本机宿主状态，独立于可交换的书。它把实际应用书号对应到增强包路径/哈希、允许的内容根、已检查的解释器和逐项冻结的运行配方。优先使用当前应用的安装/绑定流程；没有现成登记工具时，按程序源码的实际结构制作受审查的绑定，不猜字段。环境根只在本机保存，不写进 `.crlearn`。一份书本自己声明的字段不能授予执行权。

增强包必须先在隔离状态中通过 G0—G6，再更新生产绑定。按书籍 UUID、实际 EPUB 修订及应用记录核对，不增加同名假书、不覆盖原 EPUB 或批注。当前示例采用已有环境中的短探针；完整训练需独立的设备和时间预算批准。修改代码的高信任模式按代码哈希授权，不称安全沙箱。

交付记录分开报告资源/结构审计、真实计算、UI 自动化、隔离原生构建、正式安装版和真实用户反馈。没有正式窗口验证时保留 `awaiting_desktop_verification`。执行 `test_learning_pack.py` 之外，旧的两个保真测试仍需通过；新增测试不替代现有数学、PDF、分页或书库验收。

## 其他材料的短 Python 实验

0.5 使用书籍携带的 `run_definition`：`schemaVersion: 1`、`kind: python-cli@1`、入口与输入资产 ID、`argv`、类型化参数、超时、依赖包名称及可选适配资产。`learning_pack.py build` 编译为 `run_definition_json`。显示名称、参数标签和枚举说明也归书籍；宿主不按书名或章节猜它们。

检查原代码、适配、依赖、输入与副作用后，使用项目 `tools/bind_learning.py` 默认列计划；`--apply` 仅连接材料，明确选择活动并加 `--trust-reviewed-code` 才保存执行批准。本机绑定只保存定义/资源哈希、已有解释器和本机根；不知道依赖时不能猜成已准备。`tools/fetch_activity.py` 可以先列一个活动的必需文件，显式 `--download` 才取得材料，不会安装或运行。

已存在的旧宿主配方作为兼容数据保留，新书优先采用随书定义，不再把 `argv` 和实验参数只写进私人绑定。本机的 `reviewed_pure_local` 是审查结果，书籍不能自报授权。参数最终由宿主核对，未知参数被拒绝，不经过 shell。

运行定义的 `parameters` 由参数名索引，每项有真实 `flag`、`type`、默认值与范围。数值用 number/integer 和 min/max，枚举用 enum/choices，短文本用 string/max_bytes，开关用 boolean。界面可给同名的 `parameters` 数组，含 name/label/type/default/choiceLabels；服务端仍核对本机已批准定义的哈希。不要从命令示例猜参数，先核实际 argparse 或入口。适配资产可用 `root: profile` 和 `--profile-root`；适配源码仍须审查，不因它是增强包的一部分就取得权限。

代码在独立运行目录的冻结副本执行。`work/results`、`work/runs` 下的支持格式产物会登记，stdout/stderr 有限额并单独保存。普通短例可直接 print 计算结果；要展示图片或结构化记录，应将相应产物写入上述受管输出目录。不能把自由编辑标为沙箱：修改版按具体代码哈希取得高信任本机执行授权，具备当前用户的文件/网络能力。需要网络、长训练或外部服务的材料不归入 `reviewed_pure_local`；应实现相称的独立批准与生命周期流程。


现有个人安装的 Windows 0.4 版将学习数据放在该用户主目录下的 `.comfortable-reader`，具体仍以 `learning_runtime_info` 返回值为准。这是该安装版的兼容事实，不是新书源项目、公共仓库或其他系统的固定路径。新建可分享书籍按 [source-first-authoring.md](source-first-authoring.md) 从独立源开始；旧 EPUB 的 `.crlearn` 增强仍可按本文件办理，并把本机绑定留在本机。任何测试版继续使用独立应用状态，不进入现行个人目录。
