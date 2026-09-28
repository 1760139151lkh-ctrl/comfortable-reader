# 《算法导论》第 4 版：私人 EPUB 存档

这里保存两份已排版的 EPUB，供仓库拥有者在私人环境中取回和用 Comfortable Reader 检查、阅读。它们不在 `books/publication-list.json` 或 `release-files.json` 中，仓库的公开书目和站点构建不会把它们当作可发行书籍。

| 文件 | 内容 | SHA-256 |
| --- | --- | --- |
| `Introduction-to-Algorithms-4e.epub` | 英文第 4 版的重排 EPUB | `f59e5f068a636c259fff9d3894e7ecd694921bd2647a390b5f32421124a2ff51` |
| `Introduction-to-Algorithms-4e-zh-CN.epub` | 依据英文版制作的中文平行译本；不是出版社官方中文版 | `4bdac75e7b55ae184b65eb4fbad85539a3891a32ce288dd82c90dd3fb92aa390` |

两份 EPUB 都内嵌同一份英文原始 PDF（SHA-256：`ae5237c0dae8e0de3f39c7af7164e58f7dd5a351bad7b22e4dffa290472b5935`）。EPUB 的 `dc:source` 元数据含制作时的本机路径。本目录仅用于私人存档；目前没有可据以公开再分发正文、译文、PDF 和插图的许可记录。Git 历史会保留已提交的文件，因此以后即使从当前分支删除 EPUB，也不能直接将此仓库改回公开。

## 取回后检查

从仓库根目录运行以下命令。它们只审计文件，不改变 EPUB 或个人书库：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'private-books/clrs-4e/Introduction-to-Algorithms-4e.epub','private-books/clrs-4e/Introduction-to-Algorithms-4e-zh-CN.epub'
python skill/comfortable-reader/scripts/persist_epub.py private-books/clrs-4e/Introduction-to-Algorithms-4e.epub --report "$env:TEMP/clrs-en-audit.json"
python skill/comfortable-reader/scripts/persist_epub.py private-books/clrs-4e/Introduction-to-Algorithms-4e-zh-CN.epub --report "$env:TEMP/clrs-zh-audit.json"
```

两份文件是现成 EPUB。需要在另一台设备加入个人书库时，按 `skill/comfortable-reader/SKILL.md` 的现成 EPUB 流程核对实际书库、内容和最终字节，再导入并在安装版打开验收。本存档不包含个人批注、阅读进度或本机运行环境。
