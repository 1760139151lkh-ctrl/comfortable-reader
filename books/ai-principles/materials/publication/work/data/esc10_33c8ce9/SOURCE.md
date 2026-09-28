# C32 非语音事件数据来源

- 官方项目：[Piczak ESC-50 仓库](https://github.com/karolpiczak/ESC-50)，本机固定提交 `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6` 的 GitHub 归档，645832161 字节、SHA-256 `661183a6f53ef04f12c9bd618fed0ddc1713280d6c94a5a5431e844ba6f6a21f`。`fetch_c32_esc10.py` 验证归档 CRC 与元数据后，只解出 ESC-10 的 400 段 WAV、README、LICENSE 和原始 CSV；不覆盖归档或其他工作。
- 仓库 README 明写全 ESC-50 为 50 类、每类40段，5秒、44.1 kHz、单声道，来自 Freesound 公共录音手工片段。全 ESC-50 使用 CC BY-NC 3.0；其中 CSV 的 `esc10=True` 小集 **ESC-10** 在 README 中另标 CC BY，所需逐片段署名在包内 `LICENSE`。本章只训练这个10类小集，不将全数据集许可偷换成子集许可，亦不把分类成绩等同于对自由声音场景的理解。
- 400段按上游五个 fold 各80段、10类各40段。源录音 `src_file` 的任何片段都没有跨 fold；沿用 fold 做训练/验证/首次测试，可避免同一 Freesound 原片的片段跨组。不过原论文 README 也提醒，某些片段可能已有与类别相关的带宽预处理，模型可能学到采集痕迹；这是实际限制，不会由高准确率自动消失。
- 本机 `soundfile 0.13.1` 打开样例确认5.000秒、44100 Hz、单声道有符号16位PCM。若训练程序降采样到16 kHz，需用带低通的重采样并保留原 WAV；随意每隔几格抽一项会产生混叠。
