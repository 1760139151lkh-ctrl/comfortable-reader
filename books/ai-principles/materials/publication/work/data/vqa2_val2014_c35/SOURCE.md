# C35 真图问答来源

- 原件：VQA Consortium 官方 [VQA v2 下载页](https://visualqa.org/download.html) 所列 COCO val2014 的问题 ZIP 和答案 ZIP（官网转其 `cvmlp.s3.amazonaws.com`），由 `work/code/fetch_c35_vqa_v2.py`下载。归档唯一 JSON 成员名及全部 CRC 已核，字节 SHA 在本目录 `download_overlap_manifest.json`。
- 官方问答各 214354 条，其中对 C35 已取 COCO2017 val 的5000张图正好有26333问：yes/no 9943、number 3441、other 12949。这样不再下载另一份 COCO2014 照片；但 VQA v2 原发布的 **validation** 答案不能被改称官方训练/测试。
- 每题 `question` 是人写的问题；`multiple_choice_answer` 是发布者归一后的代表答案，另有10个独立答案。若本机仅以 yes/no 的代表答案作二元分类，那不是开放回答，也不是官方 VQA 评分公式。需要逐题核 10 人分歧和题型偏差。
- [VQA 官方条款](https://visualqa.org/terms.html)标问答注释 CC BY 4.0，底层 COCO 照片仍受各自 Flickr 权利约束。官方 v2 论文专门搜同题不同图却不同答案的互补例，目的是减少只看问题文字就猜的捷径；本机若只取这5000图子集，需核实际保留多少互补例，不能借论文说自身子集仍完全平衡。

获取、CRC 与交集程序：`work/code/fetch_c35_vqa_v2.py`；收据：`download_overlap_manifest.json`。
