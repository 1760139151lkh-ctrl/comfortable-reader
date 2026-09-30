# 让自己的 Codex 参与

读书不需要 Codex。它是可选的创作与维护助手，和人工作者使用同一份书源、规范与验证工具。

先核对仓库当前可见性；访问受限时使用有权限的 GitHub 身份。技能安装不授予仓库权限，也不授权公开私人归档。已有获准取得的源码包可在本机使用。

## 安装技能

在自己的 Codex 中发送：

```text
请使用 $skill-installer，从
https://github.com/1760139151lkh-ctrl/comfortable-reader/tree/v0.5.2/skill/comfortable-reader
安装 comfortable-reader 技能。先核对来源和版本；如果已有同名技能，保留我的私人配置并比较差异，不直接覆盖。
```

安装后可输入 `$comfortable-reader`，或从技能列表选择它。没有出现时再重新启动 Codex。[官方技能说明](https://learn.chatgpt.com/docs/build-skills) 说明了仓库技能安装和发现方式。

也可以手工从已核对的 Release 源码包取出 `skill/comfortable-reader`，放入当前 Codex 的技能发现目录。当前官方文档列出用户级 `~/.agents/skills/` 与项目级 `.agents/skills/`；部分既有安装使用 `~/.codex/skills/`，应以自己的实际版本和技能列表为准，不同时复制出多个同名版本。

技能与项目工具分开：技能告诉助手怎样工作，仓库提供构建、检查和预览脚本。在 Codex 中打开克隆或解压后的项目目录，技能即可按 `AGENTS.md` 找到它们。可选的技能 `local-config.json` 只记录本机 `project_root`；不要提交它。无需 OpenAI API key，也不需要把私人书库上传给项目。

## 创作自己的书

把原稿放在自己的作者目录，然后发送：

```text
使用 $comfortable-reader，为我的这些原稿建立独立书籍源项目。
先读取仓库 AGENTS.md、技能的 source-first-authoring.md 和 docs/FORMAT.md。
保留原文与公式，使用已有阅读和交互能力；遇到不支持的结构明确处理，不能丢掉它。
生成并实际检查网页预览和 EPUB，验证移动目录后资源仍能解析。
目前只做本地创作，不加入私人书库，不公开，不安装或运行书中代码。
```

你可以把最后一句换成自己真正需要的范围。创作、入库、运行和公开是不同动作；技能不会因为生成文件就自动授权后面的动作。

## 修订已有书

```text
使用 $comfortable-reader，修订 books/<书的目录> 中的这一处内容：……
先核当前源与来源证据，保留书籍 UUID，提高修订号。
只维护一份正文，重建网页与 EPUB；说明哪些内容改变，并检查数学、链接和活动。
保护我已有的批注、草稿与运行结果，不通过删书重导完成更新。
```

## 为项目投稿

先 Fork 仓库，并在 Codex 中打开自己的 fork。可以发送：

```text
请按 CONTRIBUTING.md 把这本已审阅的书准备成一个 Pull Request。
先比较当前分支与原稿，核对来源、许可证和拟公开文件；排除私人对话、笔记、凭据与本机路径。
新增书籍复用现有阅读器，不另建前端。完成结构检查和实际预览，记录未测范围。
在我的 fork 创建分支并提交，再向上游建立 PR；不要直接推送上游主分支。
```

这段话明确授权分支、提交与 PR。尚不想发送时，把最后一句改为“只准备本地改动，先不推送或创建 PR”。贡献者也可完全手工完成同样的步骤。

## 更新技能与工具

仓库 `skill/comfortable-reader` 是维护主本。更新仓库后，助手可以用 `tools/sync_skill.py --target <技能目录> --backup <新的备份目录>` 查看差异；核对后加 `--apply`。它仅同步受管文件，保留 `vendor` 与私人配置。新技能支持什么，应以实际工具和验收结果为准，不把新增说明当成软件已经实现的功能。
