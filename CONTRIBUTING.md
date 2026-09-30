# 贡献一本书，或把一处问题修好

你可以只修一个错字、补一条出处，也可以贡献整本书、改进活动或阅读器。使用文本编辑器、Codex 或其他工具都可以。阅读和本地创作不要求 GitHub 账号；向本项目提交修改时才需要自己的 GitHub 账号。

## 人工创作：先完成一章

安装 Python 3.11+。下载仓库源码 ZIP，或用 Git 克隆自己的 fork；不需要先下载 Release 的模型。创建 `my-chapters/01.md`：

```markdown
# 先提出一个可以检查的问题

写清已有的观察、现在不知道的事，以及下一步计算能回答什么。

## 一个可复核的例子

把必要条件、数据来源与计算过程放在这里。
```

在仓库根运行：

```text
python tools/new_book.py --markdown my-chapters --out ../my-books/my-book --slug my-book --title "我的书"
python tools/check_book.py ../my-books/my-book
python tools/export_epub.py ../my-books/my-book --out ../my-book.epub
```

生成器创建稳定 UUID、章节与待审核权利字段，不修改公共书目。打开 EPUB 检查实际正文；复杂公式与图片不能用当前简化 Markdown 转换器强行丢弃。它报告不支持时，保留原稿，使用技能的保真工具或 [语义 EPUB 源](docs/FORMAT.md) 流程。

网页预览需 Node.js 22+。在 `../my-books/publication-list.json` 写入：

```json
{"schemaVersion":1,"books":["my-book"],"releaseApproved":false}
```

```text
npm ci --ignore-scripts --prefix desktop
python tools/reader_build_receipt.py build
python tools/build_site.py --preview --books-dir ../my-books --reader-dist desktop/dist --out ../my-preview
python tools/serve_preview.py ../my-preview
```

输出目录需为新的空目录。移动整个 `my-book` 后再次检查，正文与资源仍应可解析。之后只修改这一份书源，提高 `revision`，重建网页和 EPUB；不分别编辑书库副本和网页副本。

## 按需要加入活动

[examples/books/measurement-lab](examples/books/measurement-lab) 展示段落、来源、数据与 `least-squares@1` 活动怎样关联；[cooling-lab](examples/books/cooling-lab) 用另一组数据复用相同能力。样例数据是合成的，不是实际测量。不要把示例正文当成自己的完整教学内容。

活动应有明确问题、章内位置、输入、可改参数、观察目标与返回位置。必需资源和可选资源分别声明；新书使用已有能力不需要修改阅读器。Python 程序和本机环境见 [本机活动指南](docs/NATIVE-ACTIVITIES.md)。虚拟环境与子进程不是完整安全沙箱；书籍声明不能给自己授予执行权。

来源卡应让未下载原件的读者也能知道题名、作者、版本、页段和用途。素材没有再分发许可时，提供来源方入口及条件，不能放虚假下载按钮。原创正文、第三方图像、字体、数据和代码分别核对；[BOOK-LICENSE](BOOK-LICENSE) 不替你取得他人的权利。

## 使用 Codex 或其他助手

见 [Codex 的安装与工作示例](docs/CODEX.md)。其他助手也可直接读取 `AGENTS.md`、`docs/FORMAT.md` 和技能流程，用相同的工具检查产物。工具选择不会让两条路线产生不同的书籍格式。

## 提交到公共书目

1. 在 GitHub 仓库点 Fork，得到自己的副本；克隆自己的副本，建立一个描述修改的分支。只改指定书或明确的阅读器功能。
2. 把经过审阅的源项目放到 `books/<slug>/`，写明作者、AI 辅助范围、引用、许可证与第三方例外。只提交这本书必要的文件，不提交笔记、聊天、凭据、个人运行记录、缓存或本机配置。
3. 在 `books/publication-list.json` 提议加入 slug。正文、资源和许可未经审核时保留 `review_required`，不要伪造通过；维护者审核后更新发行许可登记与正向文件清单。小样、测试书放在 `examples/`。
4. 用隔离目录构建，实际阅读首、中、尾和复杂内容，检查移动目录后的资源及一次更新。记录确实执行了哪些检查；没测过的安装版或平台保留未验证。
5. 将分支推送到自己的 fork，在 GitHub 点 Compare & pull request。说明具体问题、改后行为、内容与许可来源、修订号、验证结果及维护方式。维护者审查合并后，它才进入项目的发行流程。

修复现有书保留其 UUID；内容改变后提高修订号。批注和阅读位置不会只凭同名书强行迁移。新的交互类型或产品主体变化应单独提出，不让一次新书接入附带替换阅读器。

PR 的自动检查只构建惰性书籍声明与阅读源，不运行投稿者的教材程序，没有发布密钥。维护者只把明确审核的文件加入 `release-files.json`。大资源经许可、哈希核对后独立发布，不把临时链接写入书中。

## 报告问题、更正与撤回

在 [Issues](https://github.com/1760139151lkh-ctrl/comfortable-reader/issues) 写书名、公开修订、章节或段落、预期与实际结果，以及最小复现步骤。截图先移除个人信息。涉及凭据或未公开隐私时，不将原值贴进公开 issue。

权利人可以提出更正或撤下目录的请求，提供能核对归属的说明。维护者核实后更新目录、版本与处理记录；已经合法取得内容的读者和第三方 fork 不会被远程删除。已授出的开放许可按其原有条款处理，不承诺能够撤销所有历史副本。
