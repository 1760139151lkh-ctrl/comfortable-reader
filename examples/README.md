# 创作示例与测试材料

这里的三本小书用于学习书源格式、复用活动和验证工具，不列入正式书目。它们使用明确标识的合成数据，不代表完整教材。可在仓库根运行：

```text
python tools/build_site.py --preview --books-dir examples/books --reader-dist desktop/dist --out ../example-preview
```

将输出设在仓库外的新空目录；仓库内可用顶层 site-dist-rNN。生产书目在 books/publication-list.json。
