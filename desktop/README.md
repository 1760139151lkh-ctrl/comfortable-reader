# 舒适阅读书库 0.6

在既有 Tauri 2 / WebView2 / epub.js 阅读器中整合连续滚动、按章获取和随书学习。Calibre 仍可提供本机书籍；新作品从独立源项目构建，网页复用相同阅读界面。EPUB 字节不因活动或批注而改动，正式应用标识仍为 `com.comfortablereader.desktop`。

## 阅读与学习

书页默认保留正文与顶部的简短入口。点「阅读工具」或按 F8 唤出覆盖工具栏，点击正文或 Escape 收起；目录、搜索和设置的开关不改变正文尺寸。打开“随书学习”或正文中的已登记链接，可以查看实际源码、来源、图像、PDF、声音、视频和独立学习记录。宽内容使用可返回的学习舞台。退出暂停媒体，保存代码和位置，不自动执行或上传。

- Auto 根据窗口选择栏数，拖动时缩放，稳定后按语义位置重排。固定 1–10 栏在窗口改变时只缩放已有画布；主动换栏数或字体才重排。
- 单箭头移动一栏，双箭头或 PageUp/PageDown 移动一屏。F11 全屏。多章节长书显示“阅读位置”；增强书另区分正文位置和资料，不把两者当 PDF 原页。
- 可打开最多 12 本不同书籍，空间足够时同时阅读。分割布局支持上下、左右、交换和拖动比例；临时专注与窄窗切换不改写原排列。每本保存自己的位置、栏数、阅读方式、字号、字体、行距与批注。暖纸、明亮、夜读、高对比主题保留正文/数学/代码字体角色。
- 原文标记保存 CFI 与选中文字。手写笔迹按内容锚点和统一比例显示；旧笔迹缺少创建比例时，明确以升级时的布局作为兼容基准，不伪造历史几何。自由文字可编辑、拖动，并保持在可见画布内。
- 长算法、长表与长图注按语义行/步骤/段落续排；宽公式、表格和图片可打开详情。十栏是概览选择，并不意味着窄屏下同样适合长时间阅读。

## 运行与信任

`.crlearn` 是声明式 JSON 学习包，绑定 UUID 与 EPUB SHA-256。入口、输入、参数和适配随书声明；已批准的定义及资源哈希、解释器和实际根目录留在本机。书本不能声明 shell 命令或自行取得执行权。

已审查的短实验使用原有 Python / PyTorch 环境与 CPU。每次运行冻结独立输入、记录真实输出和代码版本；参考结果与本次运行分开。完整训练单独确认设备与时间预算。取消结束 Windows Job Object 内的进程树；恢复依靠算法检查点，不自动重跑。

**自由编辑代码采用逐版本明确授权的高信任本机运行，能使用当前 Windows 用户的文件和网络权限。** 虚拟环境、超时、4 GiB 进程组内存限制和 Job Object 都不是文件/网络沙箱。没有上传遥测或远程 AI 的默认流程。

EPUB 脚本关闭，来源内容不拥有运行桥。PDF.js 只按需绘制文档数据，不启用 PDF 脚本或表单动作。媒体不自动播放；视频逐帧使用实际解码 PNG 与时间/实验步映射，作者播放速度和原拍摄参数分别说明。

## 数据与恢复

- Calibre 现有根：由 `%APPDATA%\ComfortableReader\library-roots.json` 登记，启动/刷新执行全量扫描并报告错误。
- 旧书目、阅读位置、原文批注：`%APPDATA%\com.comfortablereader.desktop\reader-state.json`。
- 新学习包绑定：`%USERPROFILE%\.comfortable-reader\learning-bindings.json`。
- 新学习记录：`%USERPROFILE%\.comfortable-reader\records`；版本化资料：同目录 `packages`。
- 实际路径以原生 `learning_runtime_info` 返回值为准。QA 配置使用独立标识和状态，不写正式学习目录。

升级只切换软件和已核资料绑定。回退时保留期间新增的用户笔记、草稿和运行目录，不能用旧状态整份覆盖新成果。

进度记录源文件哈希与规范化成员身份。换版时建立独立位置和批注记录，旧版在“版本记录”保留；恢复原件后可找回原锚点。旧记录没有历史哈希时，只能从首次打开的当前版本建立基准，不能声称恢复了未知的旧版关系。跨端导入要求同书同版，不自动同步或上传。

## 可重复构建

需要现有 Node.js、Rust MSVC 工具链及 Windows WebView2。依赖版本由 `package-lock.json` / `Cargo.lock` 固定；不要求升级系统 Python 或显卡驱动。

```powershell
npm ci
npm run build
cargo test --manifest-path src-tauri\Cargo.toml
python ../tools/build_desktop.py
```

NSIS 安装包输出到 `src-tauri\target\release\bundle\nsis`。使用 `CARGO_TARGET_DIR` 时输出随之改变。隔离构建为 `npm run tauri -- build --config src-tauri/tauri.qa.json --no-bundle`，不可把 QA 标识用于正式安装。

所有构建都执行 TypeScript / Vite，避免打包旧前端。第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。构建成功、自动测试、隔离原生实测、正式安装版和真实用户反馈是不同证据层，最终发布记录逐项说明。

