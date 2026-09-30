# C35 真图与描述来源

- 原件：COCO 官方 2017 validation 图片 `val2017.zip`，官方 `annotations_trainval2017.zip` 中只取 `captions_val2017.json`。资料与下载入口：https://cocodataset.org/ 。原站图片主机 TLS 证书不匹配，实际取同一官方 S3 bucket 的 `https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip` 和 `https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip`，没有由第三方改造图片。
- 原件哈希：图片 ZIP SHA256 `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05`，标注 ZIP SHA256 `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`；两个归档全部成员 CRC 通过。提取程序仅写 5000 JPG 与上述描述 JSON。图片解出 814705164 字节，描述 JSON 3872473 字节、SHA256 `afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9`。25014 条描述，4987/12/1 张图分别有5/6/7句；逐图字节 SHA 在 `extraction_manifest.json`，该文件约0.76MB，不需全文打印。
- 权利：COCO 官方 [Terms of Use 原件](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)规定标注 CC BY 4.0；各张 Flickr 原照片保留各自许可或条款，不能把全部 JPG 说成 CC BY 4.0。描述/图仅供本机研究与正文少量示例；大范围再分发图片需要逐图核许可与署名条件。
- 身份：这些本来是 COCO 官方 **validation** 照片，不是官方 train2017 或隐藏测试。C35若按照片 ID 重新切训练/验证/测试，只是本书作者内部实验；因此任何成绩不能称 COCO 官方验证/测试榜成绩。人工描述也不是物体穷尽清单，未提到的物体不等于不存在。

获取及解包程序：`work/code/fetch_c35_coco_val.py`、`work/code/extract_c35_coco_val.py`。下载和解包收据：本目录 `download_manifest.json`、`extraction_manifest.json`。
