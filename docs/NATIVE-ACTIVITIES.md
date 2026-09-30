# 本机活动：声明随书，批准留在本机

正文与活动使用同一书籍 UUID；书内资源用相对路径和哈希关联。运行定义声明入口资源、输入资源、参数、预算、预期产物及可选适配，不携带解释器绝对路径或执行许可。

作者在学习包源规格的活动中写 `run_definition`，由 `learning_pack.py build` 序列化为 `run_definition_json`。格式版本为 `schemaVersion: 1`，当前通用类型为 `python-cli@1`。入口和输入必须引用已登记资产。`argv` 是参数数组，`parameters` 给出 flag、类型、默认值与范围；它们不经过 shell。适配代码也是单独的代码资产，须和入口及输入一起核对。

普通 Python 程序不必有适配。需要额外轨迹、原模型接口或特殊产物时，书籍可以提供适配；宿主仍只负责冻结文件、选择已批准环境、显示日志和停止。资产根通常是 `publication`；解码素材可用 `derived`，与学习包同目录的适配可用 `profile`。三者的真实路径仅在本机绑定中存在。

先在阅读器打开目标书。检查现有材料与环境：

```text
python tools/bind_learning.py --pack book.crlearn --root book-source --epub book.epub --activity my-activity
```

这一步只检测和列计划，不执行书中代码，也不安装依赖。流式书籍可以用与当前书目修订匹配的 `--manifest manifest.json` 代替完整 EPUB。若还没有材料，`fetch_activity.py --catalog <目录地址> --book <slug> --activity <id> --out <新目录>` 先列清单，显式加 `--download` 才取得所选活动的必需文件。共享分块可通过 `--cache <缓存目录>` 复用；已修改的本机文件不会被覆盖。此工具准备执行材料，正文离线保存仍由阅读器中的明确动作办理。

检查源码、适配、输入和权限后，才在绑定命令后加 `--apply --trust-reviewed-code`；只处理明确选择的活动。只加 `--apply` 则连接材料而不增加运行批准。可用 `--python <已有解释器>` 选择环境；没有依赖时先按源项目说明准备，不自动升级 Python、PyTorch 或驱动。

本机保存的是定义哈希、各输入与代码的哈希、解释器及根目录；书籍不能凭 `trusted=true` 自己获得执行权。运行时再次核对界面看到的执行修订和源码版本。修改草稿后，高信任授权按具体代码版本办理；导入草稿与下载代码都不代表批准执行。

本机原生能力目前以 Windows 为验证目标。当前短任务有时间与内存上限；完整训练另有明确设备和预算批准。停止管理进程树，但这些措施不是文件或网络沙箱。输出保存在独立任务目录，作者参考材料保持只读身份。

## 实际例子：只准备第一章

先在 Windows 阅读器中连接公共书目并打开《人工智能原理》第一章。在项目根运行以下命令查看需要的材料，再明确下载：

```text
python tools/fetch_activity.py --catalog https://1760139151lkh-ctrl.github.io/comfortable-reader/catalog.json --book ai-principles --activity perceptron --out ../my-c01
python tools/fetch_activity.py --catalog https://1760139151lkh-ctrl.github.io/comfortable-reader/catalog.json --book ai-principles --activity perceptron --out ../my-c01 --download
python tools/bind_learning.py --pack ../my-c01/profile/book.crlearn --root ../my-c01/publication --manifest ../my-c01/manifest.json --activity perceptron
```

本活动的执行材料是原感知机程序和随书适配，共 13,475 字节；此外会取得书目、清单和学习说明。它不下载其他章节、训练模型或安装环境。查看下载的两个 Python 文件、运行定义和检测结果后，可在最后一个命令加 `--apply --trust-reviewed-code`，只批准这个活动；回到阅读器重新打开随书学习，点击运行。

第一章使用 Python 标准库。第四章 PyTorch 活动以及其他实验按各自声明检测依赖，不要求读第一章的人安装它们。公共版未提供 WikiText 原文编码数组的两项训练活动，不能靠批准开关变成已就绪；完整源码与缺失材料条件在书源的 external-activities.json 中。
