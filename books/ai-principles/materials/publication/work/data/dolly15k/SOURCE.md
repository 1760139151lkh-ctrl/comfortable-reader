# Dolly 指令记录的来源与本机取舍

取得日期：2026-09-24。原始文件来自[Databricks在Hugging Face发布的 databricks-dolly-15k.jsonl](https://huggingface.co/datasets/databricks/databricks-dolly-15k)，实际下载入口为该数据集 main 分支同名文件。13,085,339 字节，SHA-256 2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec，15,011 行；字段为 instruction、context、response、category。原件保存在本目录，不改字节。

[发布者数据卡](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/README.md)注明 CC BY-SA 3.0、Databricks 员工撰写的英文指令/回复，部分 closed_qa 问答和摘要记录的参考上下文包含维基百科材料。数据卡也提示可能存在拼写、事实错误及题材/标注者偏差。本书只把它当有出处的训练示范，不把其回答当经独立核验的世界真相，或把Databricks原12B Dolly模型权重下载进主线。原版本还有不宜由此本机小模型承诺的复杂数学/代码任务。

现行派生资料为本目录 sft_v2/，由 work/code/prepare_dolly_sft.py 用第十八章当前词表 work/data/wikitext2_causal/tokenizer.json 生成。分词器SHA-256 5371bf7ee34002268dabc1676045314ef327d7d7e6c75c15a868b0a531a60cf4，仍只在 WikiText 训练文章上拟合；**没有**把Dolly测试回答偷拿去重新训练分词器。C19模型上限128个输入位置，程序不裁断上下文或回答，完整放不下的记录丢弃；去首尾空白、删除规范化后完全相同的三字段记录，并将同一规范化instruction+context的所有记录按SHA分到同一组。

作者在正式分组前曾快速看原件头三行，其中第1行按最初哈希规则落在测试组。为避免把这一已见题当成没见过的测试例，现行 sft_v2/ 明确排除原行号0、1、2及其同内容重复项；早期 sft_v1/ 仅是准备诊断，**不进入训练、验证、测试或读者成绩**。sft_v2/有训练5,482、验证678、测试691条，分别含224,839/29,295/29,566个回答或EOS目标。原始15,011条里因完整128位条件/空值过滤8,141条，规范化精确重复16条，作者先见3条。长context任务受影响尤其大：测试组摘要仅1、信息抽取8、带参考上下文的问答（closed_qa）11，分类164、短开放问答276。这种类别不均衡必须随评价报告，不能由整组NLL推出所有任务能力。

模板由本书作者选为 Instruction、可选 Context、Response 三段，不是Databricks原模型模板的复刻。分别用**同一已有分词器**编码提示与回答，再将 BOS+提示编号+回答编号+EOS 接成一个因果序列；这样回答边界是确定的编号位置。提示编号在前向中真实可读，其目标标签置−100不计损失；回答编号和EOS才参与条件负对数。因输入与loss mask各有身份，sft_v2/的 attention_mask 不等于 labels!=-100，这和第十八章普通文章目标不同。没有模型会因为读到段名 Response 就自动学会遵循指令；需要看独立生成与留出结果。

派生清单 sft_v2/manifest.json 登记原件和词表哈希、筛除、按提示分组、三份NPZ哈希、类别和目标计数。另以同程序生成 work/runs/c25_prep_rebuild/，三份NPZ逐字节相同；重建清单仅因输出目录不同而不必字节同。任何后续改模板、截断或分组都是新实验，应另存目录，不把原测试当新盲测。
