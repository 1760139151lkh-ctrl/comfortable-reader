# 数学定理等价性

从公式与句子的区别，走向证明、理论强度、反向数学与问题归约。

维护正文：`epub/OEBPS/*.xhtml`（可编辑的语义 XHTML / MathML）。`book.json` 定义稳定身份、章节与已审核材料。构建命令在仓库根执行：

```text
python tools/check_book.py books/mathematical-equivalence
python tools/export_epub.py books/mathematical-equivalence --out output/mathematical-equivalence.epub
```

更新这一源目录并提高修订号，再重建网页与电子书。不要编辑读者书库内的生成文件，也不要在这里复制阅读器。`materials` 中带旧章名的 Markdown 是原实验所引用的历史参考快照，不是当前公开正文的第二份维护稿；它们更新时应作为独立参考材料注明修订。

公开版清理本机路径、嵌入原始输入与不具备分发条件的文件，保留数学结构与已记录实验值。来源、许可和实际限制见 [RIGHTS.md](RIGHTS.md)。
