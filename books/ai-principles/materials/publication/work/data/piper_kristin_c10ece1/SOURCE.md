# C32 本机外部语音合成对照：Piper Kristin

- 来源是 [Rhasspy Piper voices 仓库](https://huggingface.co/rhasspy/piper-voices) 固定提交 `c10ece1aade47bb51c153c893d14e5bf8e5b7117` 的 `en/en_US/kristin/medium/`；只下载 `MODEL_CARD`、ONNX权重和配置。ONNX为63531379字节，SHA-256 `5849957f929cbf720c258f8458692d6103fff2f0e3d3b19c8259474bb06a18d4`，与Hub该提交的LFS SHA相同；本机逐件SHA和22,050Hz配置见 `manifest.json`。
- 该提交的模型卡说这是英语、单说话者、中等质量的**预训练外部模型**，训练者用约11.5小时LibriVox公版朗读从随机权重训练，2000轮；这些是公开模型卡的描述，**不是本书亲自取得该训练资料或复现2000轮的收据**。不能将这份外部权重的自然发声归功于本书8分钟无文字条件GRU，也不能把它当LibriSpeech CTC转写器的反向函数。
- 本机推理工具按 [Open Home Foundation Piper官方CLI/Python文档](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md) 使用`piper-tts 1.4.2`和现有`onnxruntime 1.26.0`（CPU），离线运行；Piper软件仓库为GPL-3.0，语音数据/权重的声明按固定模型卡另存，不把这几个对象叫同一种许可。Piper先用eSpeak-ng给文字音素形式，再把文字条件送入已训神经语音模型，输出22.05kHz波形。这里对软件与模型的来源作身份记录，不给超出这些文本的许可法律判断。
- 只作本地教学对照，不上载用户资料、训练原音、文字指令或输出。合成输出属于外部预训练分支的**推理**，本书没有对这份权重执行反传或优化器更新；读者可听与本书本机短历史波形生成器的差距，再回看两者训练信息的差别。
- 当前Windows Piper wheel 的 eSpeak 桥在中文路径下首次尝试回落到构建机 `D:/a/...` 并因找不到`phontab`而停止，未产生声音；程序随后把 wheel 自带的约18MB `espeak-ng-data`复制到当前用户ASCII临时目录，核`phontab`/`en_dict` SHA相同，再将该目录显式交给`PiperVoice.load`。二次调用返回0，输出两段22.05kHz单声道WAV；这是**运行路径修复**，没有换模型、外部合成资料或手写拼接声音。
