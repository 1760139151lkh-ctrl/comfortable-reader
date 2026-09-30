# 人工智能原理：从学习机制到可运行系统

从学习规则、概率与表示，到语言、视觉、行动和可运行系统。导读与 47 章，保留真实代码、参考结果和按需实践。

维护正文：`epub/OEBPS/*.xhtml`（可编辑的语义 XHTML / MathML）。`book.json` 定义稳定身份、章节与已审核材料。构建命令在仓库根执行：

```text
python tools/check_book.py books/ai-principles
python tools/export_epub.py books/ai-principles --out output/ai-principles.epub
```

更新这一源目录并提高修订号，再重建网页与电子书。不要编辑读者书库内的生成文件，也不要在这里复制阅读器。`materials` 中带旧章名的 Markdown 是原实验所引用的历史参考快照，不是当前公开正文的第二份维护稿；它们更新时应作为独立参考材料注明修订。

公开版清理本机路径、嵌入原始输入与不具备分发条件的文件，保留数学结构与已记录实验值。来源、许可和实际限制见 [RIGHTS.md](RIGHTS.md)。
