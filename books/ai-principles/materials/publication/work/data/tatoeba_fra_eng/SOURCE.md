# 英法句对来源与本章切分

2026-09-24 从 [ManyThings.org 的双语句对页](https://www.manythings.org/anki/)下载 fra-eng.zip。包内 _about.txt 写明版本日 2026-02-13，数据来自 [Tatoeba](https://tatoeba.org/) 的句子和翻译，法英文件每行是英语、法语、逐句贡献者与 Tatoeba ID 三栏。外层 ZIP SHA-256 为 9e35994767f7307dc1fa9d5efcc36681f3756a1bdaaa02c1381b3eb11461261e；fra.txt SHA-256 为 5fb8a159269e50d5de145ace082aa89da55b988969cf0cd61a4fb7e7aae84548。来源页称原包 240,521 对；本机核到相同行数、两文件成员。

[准备程序](../../code/prepare_tatoeba_fra.py)对原句做 NFC、小写，按 Unicode 单词或单标点切单位；只保留英法双方各 2–18 单位，去规范化后完全相同的英法对。筛去长度 4,130 行，移去规范化完全重复的句对 164 行。以规范化**英语整句**的 SHA-256 分组，同一英语来源只能落一个集合，得到训练 189,037 对、验证 24,015 对、测试 23,175 对。约 77.4% 的训练对英法单位数不同；短句很多，数据不能代表各种长篇翻译。规范化法语完全相同的目标仍能对应不同英文，训练／测试之间有 3,921 个相同法语目标串；近义改写与错误翻译未充分审计。

词表只读训练句：英语、法语各最多 8,000 普通单位，另有 PAD/UNK/BOS/EOS。验证／测试未知普通单位比例，英语约 1.0%／1.1%，法语约 2.4%／2.4%。[manifest.json](manifest.json)保存原包、实际数组、词表和逐句署名文件的 SHA，模型按它复核数据身份。训练时输入英语单位加 EOS，decoder 输入 BOS 加目标前缀，训练目标是法语单位加 EOS，PAD 不计损失；未把同位英语词当法语答案。

本章画权重图的一对句子在来源文件零起第 26766 行：英语 “He wore red pants.”，法语 “Il portait un pantalon rouge.”，归属 Tatoeba [#300587（CK）](https://tatoeba.org/en/sentences/show/300587) 与 [#132782（Julien_PDC）](https://tatoeba.org/en/sentences/show/132782)，ManyThings 第三栏原样保存。它在验证集合，且在训练前按词序变化选定；模型权重不等于人工标注词对齐。

ManyThings 原包及下载页称材料沿用 [CC BY 2.0 France](https://creativecommons.org/licenses/by/2.0/fr/)，并要求引用 ManyThings.org/anki、Tatoeba.org 和用到的句子贡献者；本目录保留每行完整第三栏的训练／验证／测试署名 JSONL，正文只引用少量示例。网站同时警告原句对可能有错，筛选也不能保证全对。这个本地研究不提供“无须逐句署名即可再发布整包”的许可结论。
