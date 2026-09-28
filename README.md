# 舒适阅读书库 · Comfortable Reader

**选一章，顺着问题读下去。** 同一本书可以翻页、多栏或连续滚动；需要时打开随书学习，查看来源、代码、声音与三维材料，再回到原段落。

[开始在线阅读](https://1760139151lkh-ctrl.github.io/comfortable-reader/) · [下载 Windows 版](https://github.com/1760139151lkh-ctrl/comfortable-reader/releases/tag/v0.5.1) · [读者指南](docs/READER.md) · [贡献一本书](CONTRIBUTING.md) · [让 Codex 加入](docs/CODEX.md)

阅读无需账号、无需克隆源码，也无需安装 Python 或 GPU 环境。本站使用桌面版的同一套阅读与学习界面。阅读方式与主题分别设置，笔记留在自己的设备；目前通过主动导入、导出迁移个人记录，没有自动上传或跨端同步。

## 正式书目

| 书籍 | 从哪里开始 |
| --- | --- |
| [人工智能原理：从学习机制到可运行系统](https://1760139151lkh-ctrl.github.io/comfortable-reader/?book=ai-principles&chapter=c01) | 导读与 47 章，从经验怎样改变规则，走向语言、视觉、行动和系统 |
| [数学定理等价性](https://1760139151lkh-ctrl.github.io/comfortable-reader/?book=mathematical-equivalence) | 公式、句子、证明、弱背景与问题归约 |
| [有限元方法：从直觉到可靠计算](https://1760139151lkh-ctrl.github.io/comfortable-reader/?book=finite-element-method) | 弱形式、基函数、矩阵组装与误差 |
| [智能优化算法简介](https://1760139151lkh-ctrl.github.io/comfortable-reader/?book=intelligent-optimization) | 表示、评价、搜索、选择和可比较的实验 |
| [模糊规划：把“差不多”变成可计算的决定](https://1760139151lkh-ctrl.github.io/comfortable-reader/?book=fuzzy-programming) | 模糊目标、软约束与满意度 |
| [综合评价方法：从指标、权重到可信结论](https://1760139151lkh-ctrl.github.io/comfortable-reader/?book=comprehensive-evaluation) | 指标、权重、AHP、TOPSIS、PCA 与 DEA |
| [运筹学分类检查：建模流程完整闭环](https://1760139151lkh-ctrl.github.io/comfortable-reader/?book=operations-research) | 从识别问题，到建模、求解、验证和反馈 |

三本短小的合成数据样书放在 [examples](examples/)，供创作和测试使用，不列入正式书目。软件快速入门、阅读器验收样章和无全文再分发授权的第三方出版物没有进入发行物。

## 阅读与实践

- 在线进入某章，只取得该章正文、必要插图和小型元数据。离线保存可选章节、活动或独立材料；完整 EPUB 另行下载。
- 学习空间保留参考结果、当前代码、运行版本与回放步骤的区别。换主题或阅读方式不会启动实验。
- Windows 支持经过本机审阅与绑定的 Python 活动。下载材料、准备环境、批准代码和开始运行是不同动作；浏览器不冒充本机 Python。
- 大型参考检查点放在 Release，以固定文件身份取得。普通阅读不下载这些模型。

第三方内容各有边界：[公开书目与材料范围](docs/PUBLIC-CONTENT.md)。例如 ETH3D 图像与三维材料遵守其非商业、同类共享许可；部分原画面和原始语料只提供合法来源入口。私人版本的材料仍可在自己的设备使用。

## 创作与贡献

人工、Codex 和其他工具从同一份独立书源开始。普通书用 Markdown 起步，复杂数学书可维护语义 XHTML / MathML；网页、EPUB 和学习包从该源生成。加入个人书库不等于公开投稿。新增书籍复用现有阅读器，不复制一套前端。

```text
python tools/new_book.py --markdown my-chapters --out my-books/my-book --slug my-book --title "书名"
python tools/check_book.py my-books/my-book
python tools/export_epub.py my-books/my-book --out my-book.epub
```

[贡献指南](CONTRIBUTING.md) 包含从第一份原稿到 Pull Request 的完整步骤；[Codex 指南](docs/CODEX.md) 提供技能安装、创作、修订与投稿提示词。`skill/comfortable-reader` 是技能维护主本，本机副本只同步受管文件，保留个人配置。

## 自行构建

Python 3.11+、Node.js 22+。仅预览自己的小书不需要公共教材的大模型；构建完整公共站点时，显式取得四份固定参考检查点：

```text
npm ci --ignore-scripts --prefix desktop
python tools/prepare_release_assets.py --download
python tools/build_site.py --release --out site-dist
python tools/serve_preview.py site-dist
```

打开 `http://127.0.0.1:4173/`。此服务只监听本机。源码、共享渲染边界、Windows 构建与测试见 [开发说明](docs/DEVELOPMENT.md)；站点、镜像与版本发布见 [发行说明](docs/RELEASE.md)。

软件和技能采用 [MIT](LICENSE)，原创教学内容采用 [CC BY 4.0](BOOK-LICENSE)，第三方组件与材料保留各自声明。作品由项目维护者使用 AI 辅助编写和整理，真实引用、编辑责任及实验条件保留在各书源中。初始维护者为 [1760139151lkh-ctrl](https://github.com/1760139151lkh-ctrl)，后续贡献在 Git 历史中记录。项目不要求使用某个 AI 厂商，也不代表任何厂商背书。
