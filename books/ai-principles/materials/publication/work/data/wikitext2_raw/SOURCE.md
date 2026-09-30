# WikiText-2 raw：真实文本的来源与本章处理

2026-09-24 从 [Salesforce/wikitext 数据集仓库](https://huggingface.co/datasets/Salesforce/wikitext)的 wikitext-2-raw-v1 子目录、提交 b08601e 下载三份 Parquet；内容来自 Wikipedia 的 Good/Featured 文章，WikiText 数据集见 Merity、Xiong、Bradbury、Socher 的[原论文](https://arxiv.org/abs/1609.07843)。这里的 raw-v1 指没有由基准处理预先替换稀词为 UNK；原文已经有分开的标点、@-@ 等记号，**不是**直接从百科网页原始 HTML 提取的一份无处理文本。

| 作者镜像划分 | 文件 SHA-256 | 本章读出的文章数 |
|---|---|---:|
| train-00000-of-00001.parquet | e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7 | 600 |
| validation-00000-of-00001.parquet | 204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c | 60 |
| test-00000-of-00001.parquet | 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91 | 60 |

[准备程序](../../code/prepare_wikitext2.py)按两侧空行包围的顶层“= 标题 =”辨认文章。仅凭等号行就切，会把一篇冰球条目中“= Goals ; A =”的表格键错当新文章；第一次核查发现这一点，程序已按真实相邻行修正。训练／验证／测试沿用来源已有的文章分开，不把文章段落随机送去不同集合；跨集合标题与完整文章原文哈希未见相同，**未做近似重复正文的充分审计**。空行在文章内记为一枚 PARA，文章开头给两个 BOS 作两词上下文，结尾给 EOS 目标；上下文永远不跨文章。[文章索引](article_index.json)留标题、源行号、词数与原文哈希。

本章词级主实验对作者镜像中已空格分开的字段使用 Python split：它没有发明一套通用英语分词器，也不能凭词 ID 无损找回原始空白。词表只看 600 篇训练文章，固定最常见 10,000 个普通字段和 UNK/BOS/EOS/PARA 四个特殊符号，实际 [word_vocab.json](word_vocab.json) SHA-256 为 f715a5167532a6c35d66ab9f0cc9ea3c6b7db794f5d20d70d2242d3c4cc4923e。[word_documents.npz](word_documents.npz) SHA-256 为 f9810b4a338bf4b653419d1b30bfeef2815955d610bb543d24282005365e5e3e。原目标映成 UNK 的比例：训练约 9.79%、验证约 11.76%、测试约 12.57%；词级 perplexity 只对**映射后的这 10,004 类任务**有意义。

子词演示从训练集合中最多 30,000 个按频数排序的纯字母字段学 256 次词内字节对合并；合并表在 [word_internal_byte_merges.json](word_internal_byte_merges.json)。由于 256 个基础 UTF-8 字节始终保留，未见字段 Homarus 在词级为 UNK，却能用四个学习得到的片段拼回原字节。这是本章课堂版词内 byte-BPE，既不跨字段，也不声称与 GPT 的 tokenizer 配方相同。

**使用条件。**HF 仓库顶端标记 cc-by-sa-3.0 与 gfdl，数据卡许可段另写 CC BY-SA 4.0；Wikipedia 来源又有各篇贡献者归属。本地教学研究保留作者、来源和字节身份，不把这几处不一致擅自合并为“完全自由再分发”。需要把语料或衍生文本另行发布时，应核原仓库和对应文章的实际许可与归属。教材正文只摘很短的来源片段作计算追踪。
