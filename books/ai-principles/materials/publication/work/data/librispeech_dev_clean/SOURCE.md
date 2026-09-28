# C32 真实英语朗读语音来源

- 官方原件：[OpenSLR SLR12 LibriSpeech](https://www.openslr.org/12/) 的 `dev-clean.tar.gz`，本机原归档 `dev-clean.tar.gz`，337926286 字节。官方 `md5sum.txt` 对该文件给 `42e2234ba48799c1f50f24a7926300a1`，下载后本机重算一致；本机 SHA-256 为 `76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3`，详见 `manifest.json`。
- 包内 `LibriSpeech/LICENSE.TXT` 写明 CC BY 4.0，`README.TXT` 说这是基于 LibriVox 公版有声读物的**朗读语音**；`dev-clean` 是官方**开发集**，不是官方训练集。`clean` 是粗略自动选择，不保证每段完全无噪声，也不代表自然对话、其他语言或所有口音。
- 安全解包只允许 `LibriSpeech/dev-clean/` 下的普通文件和根部五份元数据，拒绝符号链接与越界路径。实际 2703 段 FLAC、97 份转录文本、40 位朗读者；`soundfile 0.13.1` 逐文件头核实均为 16 kHz 单声道。归档及解包后文件均保留，脚本不覆盖已核原件。
- 本书另用 `prepare_c32_librispeech.py` 将这份**开发集内部**按朗读者切出训练/验证/首次留出测试，明确是作者自定义切分，成绩不能与官方 train-clean-100 或 test-clean 基准互换。准备脚本会读音频头和全部配对文本建清单，训练脚本只用其中 train/validation；测试内容仅在模型与选择规则固定后打开并计分。全程记录说话者、片段、时长、文本与归档身份。
- `soundfile 0.13.1` 仅装在本包的 `work/envs/c32-audio/`（继承本机原有 PyTorch/NumPy），用于读取 FLAC；不使用外部预训练模型或外部转录 API。后续训练路线若改变，在实验收据里另记。
