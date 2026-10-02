# PDF 首次重建流程

任何 PDF 输入都先读本文件。这里把“注意保真”展开成操作步骤，不依赖用户提供反面教材。

## 0. 不要从这个错误起点开始

以下结果都不是本技能要求的重排书：

| 错误做法 | 读者会看到什么 | 正确处理 |
|---|---|---|
| `extract_text()` 的每个块直接写成 `<p>` | 斜体词独占一段、同一句断成几段、跨原书页面出现空行 | 根据原页还原段落；字体变化放在同一个段落内 |
| 每页转 PNG，再塞进 EPUB | 调整字号没有正常重排；十页只是十条缩小的图片 | 正文用文字、数学用 MathML、表格用单元格 |
| 把所有公式裁成小图片拼回正文 | 白底碎片、大小不一、基线错乱、公式不能随字体自然变化 | 将公式转写为结构化数学，并与原稿逐项核对 |
| 遇到难图就删掉，或只留 OCR 出的数字 | 箭头、颜色、步骤、下划线等信息丢失 | 保留真实插图的完整视觉编码，或完整重建其语义结构 |
| 把全书放进一个 XHTML，直到软件变慢才处理 | 超长排版视图、打开和跳转缓慢、内存异常 | 提前测试分章/分节与资源加载；不用原 PDF 页当永久阅读页 |
| 只核对 `U+FFFD == 0` 或源 PDF 哈希 | 仍可能是错公式、缺下标、错表头、读序混乱 | 做源文件、内容映射、实际显示三层验证 |

## 1. 建立一次完整盘点

先保留原文件并计算 SHA-256。检查实际页数、目录、文字层、字体种类、图片及矢量对象、链接、空白或报错页。批量提取前检查可用 Python 环境及依赖；不要因一个环境缺包，就反复换转换器或重装软件。

至少实际查看这些位置：开头正文、含公式的位置、算法或表格、插图、书末。PDF 能复制文字，**不代表公式是文字**：同一行可能混合字形、行内公式图片、单独绘制的上划线。

下列任一项出现，就不能把普通抽文稿当成正式书：

- 公式、表格、算法、分栏、脚注、索引、嵌入图片或 Type3 字体。
- 提取文本的顺序与原页不一致，符号或上下标不完整。
- 原页有内容但提取为空，或有提取异常。
- 一页中的段落因换字体而被拆开，或段落在页边界中断。

建立可追溯的内容登记。可以用段落/公式/表格/图/脚注作为单位，必要时细化到 span 或字符运行段。至少记录：

```json
{
  "source_id": "稳定的原稿单位标识",
  "source_page": 1,
  "source_bbox": [0, 0, 0, 0],
  "kind": "paragraph/formula/table/figure/footnote/index",
  "output_target": "section.xhtml#element-id",
  "status": "pending/checked",
  "exception": null
}
```

图片要记录原始资源哈希和每次出现的位置。一个资源被多次引用不一定是重复错误；一次跨页绘制也不一定是两幅图。**先完整登记，再决定如何表现。**

普通纯文本 PDF 若经过核对确实没有上述结构，可直接整理抽文稿；用户明确只要文本时按该范围工作，记录被明确排除的内容。不要把所有 PDF 都强制送 OCR。

### 预检命令和输出

使用技能自带的盘点脚本，不要临时写一个只打印文字的脚本：

```powershell
python <skill-dir>\scripts\inspect_pdf.py <book.pdf> `
  --out <work-dir>\pdf-packet --render-pages "1,13,101" --refresh
```

它只会生成诊断包，不会写书库。包中包含：

- `manifest.json`：源路径、源 SHA-256、页数、目录、字体、资源、每页 unit ID 和错误。
- `pages/page-0001.json`：原页文字运行、坐标、图片出现位置、矢量对象和链接。
- `assets/`：按源资源哈希保存的图片字节。
- `previews/`：用于人工查看的页面 PNG。

盘点包出现 `errors` 时先解决错误或在报告中说明，不能把错误页默默删掉。它是输入证据，不是 EPUB；不要直接把 `pages/*.json` 中的文字拼接后入库。

## 2. 先做能暴露问题的小样

从本书中选择实际存在的结构，而不是只挑封面和最容易的几页：

1. 含行内斜体、跨原页连续句子的正文。
2. 含上下标、求和/极限、分段函数、取整符号或矩阵的公式。
3. 算法块，最好包含跨原页的情况。
4. 带表头、空单元格或多行单元格的表格。
5. 真正的图及完整图注。
6. 脚注、目录、参考文献或索引条目。

先重建这些单位，按 [math-fidelity.md](math-fidelity.md) 检查数学，再按实际阅读样式预览。**小样仍有孤立词、错符号、碎公式、空表头或一词一行时，修正方法，不要扩大到全书。**

小样通过后再运行全量识别、转换。复用一次盘点和缓存，避免每修一个公式都重新扫描整个 PDF。缓存至少关联源哈希、转换规则版本和渲染版本；相关规则改变后，旧的对照图不能继续当成当前版本通过的证据。

## 2.1 语义重建稿的固定接口

基础 agent 不需要自行设计中间格式。将盘点包旁边写成 `reconstruction.json`，最小结构如下：

```json
{
  "packet": "pdf-packet/manifest.json",
  "scope": {"kind": "sample", "pages": [13, 14]},
  "title": "Book title",
  "authors": ["Author"],
  "blocks": [
    {
      "id": "p13-preface",
      "type": "heading",
      "level": 1,
      "sources": ["p0013.text00001"],
      "runs": [{"text": "Preface"}],
      "source_reviewed": true,
      "source_evidence": [{"page": 13, "observation": "Heading matches the rendered preview."}]
    },
    {
      "id": "p13-body-1",
      "type": "paragraph",
      "sources": ["p0013.text00002", "p0013.text00003"],
      "runs": [{"text": "..."}],
      "source_reviewed": true,
      "source_evidence": [{"page": 13, "observation": "Both source lines are continuous and punctuation matches."}]
    },
    {
      "id": "p14-equation-1",
      "type": "math",
      "sources": ["p0014.image0001"],
      "latex": "T(n)=...",
      "source_reviewed": true,
      "source_evidence": [{"page": 14, "observation": "Checked exponent, limits and terminal punctuation against the preview."}]
    },
    {
      "id": "p14-table-1",
      "type": "table",
      "sources": ["p0014.image0002"],
      "header_rows": 1,
      "rows": [[{"runs":[{"text":""}]}, {"runs":[{"text":"A"}]}], [{"runs":[{"text":"row"}]}, {"runs":[{"text":""}]}]],
      "source_reviewed": true,
      "source_evidence": [{"page": 14, "observation": "Left blank cell and both columns match the source table."}]
    }
  ],
  "excluded_units": [],
  "issues": []
}
```

规则是固定的：

- `scope.kind = sample` 只能生成诊断稿；`scope.kind = full` 才可能成为正式书。`full` 需要覆盖盘点包里的每个 source unit，或在 `excluded_units` 中写允许的理由和证据。
- `sources` 必须使用盘点包中真实存在的 ID；不能为了让数量相等而编造 ID。一个可见 block 至少绑定一个源 unit。
- `runs` 只接受 `text`、`math`、`em`、`strong`、`underline`、`sub`、`sup` 和 `href` 字段，不接受未经审查的 raw HTML。普通文字 block 会自动比较其 source unit 文本；顺序或字符不一致就停止。
- `math` 的 `latex` 是人工/有证据修复后的结构，不是 OCR 原样粘贴。所有上下标、取整、上划线、行尾标点都要在 `math-fidelity.md` 的对照后标记 `source_reviewed: true`。
- `table.rows` 每行必须有明确单元格，空格也写成空 cell；复杂合并单元格使用 `{ "runs": [...], "colspan": 2, "rowspan": 2 }`，不能靠视觉猜测填充。行列覆盖不能重叠或留下未声明的格子。
- `figure` 必须写 `asset_id`、`figure_kind`、`visual_elements_checked: true`，并把图的 image occurrence unit 绑定到该 block；图注是独立的语义 block，不能只保留 OCR 描述。
- `excluded_units` 只能使用 `page_background`、`running_header`、`page_number`、`verified_duplicate` 或用户明确的 `explicit_user_exclusion`。`verified_duplicate` 必须写 `same_as` 和证据；“看起来重复”不是证据。
- 未完成的 block 不得写 `source_reviewed: true`；正式构建会阻止 pending block、缺失 unit、重复 unit 和 `issues` 非空。

可用的排版语义字段：

- `paragraph.role` 可为 `local-heading`、`caption`、`bibliography`、`index-entry`、`toc`、`colophon`；语义角色用于段距/悬挂缩进，不用到处写内联绝对字号。
- 成本分析等表格可标 `role: "algorithm-analysis"`；`header_rows` 生成真正的 `thead`，可在续栏重复，不能把表头当普通首行。
- `algorithm.lines` 中每步用 `number`、`indent`、`runs`；过程标题用 `role: "title"`，`comment_start` 是 runs 中注释开始的零基索引。PDF 的物理软换行不等于新的算法步骤。
- `algorithm.case_groups` 用 `start_row`、`end_row`（lines 数组的零基、含端点索引）及 `latex` 表示右/左括号与 case 标签；只保留必要短组不可分割，不锁住整段算法。分组不可重叠。
- 顶层 `navigation` 可显式列 `{ "level": 1, "target": "已有块ID", "label": "源目录文字" }`，保留层级及版权等非 heading 目标；不必为了进入目录把正文段落伪装成标题。`authors` 按真实作者分别列出。
- `figure.caption` 保留全部语义块；构建器用 `aria-describedby` 关联完整图注。`alt` 使用实际图号/可证实描述，不拿模型臆造的 OCR 描述代替原图注。

`source_reviewed` 不是可信的单独开关。每个 block 还必须有 `source_evidence` 数组，每项至少写 `page` 和 `observation`，例如“第 13 页首行与预览逐字一致”“第 101 页上划线覆盖变量 `A` 的补集符号”。只写“已看过”或复制同一句泛化说明不算证据。图和矢量对象必须记录观察到的箭头、颜色、线条、标签等要素；数学必须记录具体符号/范围的核对。

盘点包中的 `kind: vector` 不能丢掉，也不能随手归为“背景”。将它作为包含该线条/箭头的 `figure`、`math` 或 `table` block 的 `sources`，并在 `source_evidence` 中说明其作用。如果确实是页码、页眉或背景，才用允许的 `excluded_units` 理由和页码/坐标证据。只有图片 occurrence 才能作为 `figure.asset_id`；纯矢量图必须先重建为语义块，或制作经过核对的整幅图资源并记录其来源，不能用一串猜测的数字替代。

运行接口：

```powershell
# 先生成可检查的样章，不可入库
python <skill-dir>\scripts\build_from_ir.py <work-dir>\reconstruction.json `
  --epub <work-dir>\sample.epub --report <work-dir>\sample-coverage.json --draft

# 全书完成后，去掉 --draft；仍只生成文件并报告覆盖，不写书库
python <skill-dir>\scripts\build_from_ir.py <work-dir>\reconstruction.json `
  --epub <work-dir>\final.epub --report <work-dir>\coverage.json
```

正式构建输出的 coverage report 必须显示 `source_review_passed: true`、所有 missing/duplicate/out-of-scope/pending/issues 都为空，并绑定最终 EPUB SHA-256。只有随后用 `persist_epub.py` 和 `source_review.method = pdf_reconstruction` 才能进入 Calibre。脚本报错时保留草稿和错误清单，不能改成 `--draft` 后当成最终书。

## 3. 段落、字体与上下标怎么还原

PDF 主要记录“在某处画这些字”，不记录可靠的段落树。抽取器的 block、line、span 不是可以直接采用的语义段落。

例如原文是一句话：

```html
<p>We value <em>clarity</em> in every explanation.</p>
```

错误重建成三个 `<p>`，即使字符一个不少，也会让 `clarity` 漂浮成独立段落。

操作顺序：

1. 看原页的首行缩进、连续行基线、段间距和标题层级。
2. 将同一视觉行中的不同字体片段放回同一行，再将属于同一段的行连接起来。
3. 跨原页时看句子和缩进是否继续；不要把每个 PDF 页首强行设为新段。
4. 保留原有字词、标点及强调。不要一律删换行、一律删连字符，也不要凭模糊词典改词。
5. 如果同一个 span 内含不同基线的字符，读取字符级坐标再拆成格式运行段；例如 `dv` 可能实际是 `d` 加下标 `v`。
6. Type3 字体的粗斜体可能要看 FontDescriptor；不能只信抽取器给出的 flags。
7. 行内图片可能夹在同一个文本 span 的大边界框里。检查真实字符之间的空隙，不能因边界框重叠就把行内公式挪到别处。

输出正文使用相对字号和语义标签。不要把 PDF 的 `15pt`、绝对坐标或每行固定宽度带入重排正文。阅读器会按同屏页数调整字号，固定字号会抵消这项功能。

英文使用中文字体可能让引号、撇号和缩写显得松散。先查 DOM 空格，再查计算字体；不因字距改原文。正文优先继承阅读器字体角色，数学使用数学字体并保留 CJK 回退。新构建 EPUB 的共享样式在 `assets/epub-reading.css`，不是每本书再复制一套固定字体。检查最终计算样式和字体来回切换。

## 4. 图片、数学与数据表的分类

每个嵌入资源都要分类，尺寸和文件名不能代替观察。

- `Ω` 上方的小 `∞`、带帽字母、指向式中的箭头，都可能是数学内容，不是可以留成小贴图的“插图”。
- 独立公式应转成完整 MathML；不要逐个字符裁图拼接。
- 纯数据表、填写格、字符数组应重建单元格。保留空格、行列标题、上下标、字面字符和行列关系。
- 树、图、几何图、照片、含颜色/箭头/步骤的算法演示图通常保留完整原图。不能只留下数字、删掉箭头和着色，再声称信息齐全。
- **标了 Figure/图号不自动意味着可以保留为图片。** 编号图也可能只是一张表或一条公式。判断依据是实际内容与用户的输出要求，而不是“图号齐全”这一项计数。
- 复合插图中的图形、内嵌标签和说明须保持可理解的关系；完整图注可含多个段落、表格或公式。不能只将第一行说明与图放在一起。

常见表格错误要主动找：表头漏掉左上角空格导致整行错位；把算法阶段的着色表当成普通数表；漏掉第二张并排表；把用于填写答案的空格填成推测值；把字面 `$` 当 LaTeX 定界符删掉。

## 5. 矢量标记和源文件异常

上划线、补集横线、表格分隔线、求和竖式横线、项目符号可能是单独的绘图命令。普通文字层不包含它们。

- 结合矢量边界和相邻字形检查标记含义。缺一条补集横线会改变公式，不是装饰损失。
- 字号变化、单独基线或上划线出现时，先核对原页，再确定 `sub/sup/mover` 等结构。
- 不按“横线下方有字”就推断分数。竖式求和、矩阵分块、分数有不同语义。
- 发现页外或裁剪区外字形，不可直接删除。只有能证明它是相邻页相同字形/相同图片的重复绘制，才去重；记录原位置、对应单位、偏移和理由。偏移值由本 PDF 测得，不能套用某次案例的 648 点。
- 原 PDF 本身也可能有错误。编码/读序修复需要原字形或结构证据；仅凭数学常识“这里应该是另一个符号”，不能静默改原稿。

## 6. 算法、目录、参考文献和索引

- 算法整体保留行号、缩进、注释和分支范围。跨原页续行应回到同一个算法块；成本分析的附加列应保留成列。
- 排算法读序时按实际文字基线，不按包含高括号/图片的联合 bbox 顶边。分别检查步骤编号序列、缩进变化及 case 所覆盖的指令，字符总数相同也可能错序。
- 合并 PDF 软换行，保持 code 与右侧 comment 的关系；重复行号只有源证据确认为续行重绘时才合并，记录证据。成本表的 `cost`/`times` 可能在同一 span，按原字形位置拆列，不能把整张表误作两列伪代码。
- 长算法按完整步骤续栏，長图注按段续栏，保留语义归属；不把总高度作为整块缩小的理由。矩阵、关系表、独立公式仍需保全必要的二维关系。
- 标题不能把每个换行都变成新标题；长标题可以包含多个视觉行。题目编号、子问题标号不能被合并丢失。
- 目录要可点击。分段后重写并验证内部 `href#id`，包括图、公式、脚注、文献引用和跨章节链接。
- PDF 链接区域可能重叠；较大的矩形可覆盖较小的具体链接。检查实际锚文本和目标，不要让一整行普通文字意外跳去同一个章节。
- 参考文献按条目保留；核对编号范围与缺号，不假设任何书都从 1 连续编号。
- 索引按词条和缩进层级保留，跨页继续词条时合理连接。把几十页索引合成一个段落，是结构错误。

## 7. 分段与性能

先修正段落/表格结构，再按章节、小节及适度大小拆分 spine。不要为了分段切断图与图注、公式、表格或算法，也不要按 PDF 原页固定分页。

`build_reader.py --max-section-chars 16000` 是可试的起始参数，不是跨所有书的保证。检查实际段落长度、生成的分段数、超大原子块、打开和跳转时间、内存及跨章节滑动。阅读器应按需要呈现邻近内容，不用“懒加载整页图片”回避重排问题。

多章节书可最终采用稳定文本位置并明确标为“阅读位置”；它不是屏幕页数估算，不随栏数变化重建。若声称精确页码，则须实测布局。应用问题先区分 CSS/结构与程序；是否改程序按用户授权范围决定。

## 8. 全书验收与失败处理

全量登记必须能解释每个输入单位如何进入输出，或为何是已证实的重复绘制。还需逐项处理 OCR/渲染差异、检查数学与表格语义、实际翻阅具有代表性的首中尾和边界。登记为“已分配”不能替代“已正确重建”。

如果发现一种错误模式，例如漏帽子、遗漏列、页尾字形重复：修正规则后搜索全书同类对象，不只修用户截图那一处。然后重新生成受影响的证据。

没解决的问题进入明确清单，不允许用空白、乱码、`FORMULA REVIEW REQUIRED`、原页图片或编造文字填补。只在小样、全书内容、最终 EPUB 都合格后按 [delivery-gates.md](delivery-gates.md) 入库。

## 证据的适用范围

源坐标、资源号、纠错字典和个别书的计数不作为通用转换器发布。保留可复用的结构接口、审计与语义样式；每一本新书重新取样并绑定自己的哈希。用全量自动检查、差异队列逐项处理及代表性视觉抽样，明确报告各自范围，不声称人工看过每页或每个公式。
