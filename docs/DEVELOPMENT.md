# 一套阅读器，两种平台适配

`desktop/src` 是共享阅读与学习界面，包含原 EPUB 分页、滚动、批注、媒体和学习空间。`platform.ts` 选择原生 IPC 或浏览器个人存储；`portable-books.ts` 将相同书籍声明接入阅读器。`site/core.mjs` 提供内容校验和选择范围，`store.mjs` 保存可清理的内容缓存。不要为新书复制界面。

`desktop/src-tauri` 管理已登记本机文件、持久化、代码冻结、进程生命周期与权限。书籍可以提供版本化的呈现说明和 `run_definition_json`；本机只保留批准过的定义及资源哈希、解释器和根目录绑定。定义、输入、源码或环境发生变化时，运行入口重新核对。专用运行适配属于书籍，宿主验证它的哈希后才冻结执行。

准备开发依赖：

```text
npm ci --ignore-scripts --prefix desktop
python tools/prepare_release_assets.py --download
npm run build --prefix desktop
python tools/build_site.py --preview --reader-dist desktop/dist --out site-dist
python tools/serve_preview.py site-dist
```

站点只输出共享界面和纳入清单中的内容。每章 XHTML 与完整 EPUB 来自同一次导出；构建时使用与桌面相同的 epub.js 位置算法，不在网页首次阅读时遍历全部章节。内容成员的规范化哈希将按章文件与本机 EPUB 关联，每个实际取得的文件另核字节数和 SHA-256。

```text
python tools/build_site.py --preview --books-dir examples/books --reader-dist desktop/dist --out site-dist-r01
# Set SITE_DIST=site-dist-r01 for the following fixture tests.
node --test tests/core.test.mjs
python -m unittest discover -s tests -p "test_*.py"
cargo test --manifest-path desktop/src-tauri/Cargo.toml --lib
```

`SITE_DIST` 可以指向另一个构建目录。书籍、数学和安装版的验证范围见技能契约；静态测试不能代替真实窗口中的位置、媒体、原生运行与重开检查。`tauri.qa.json` 使用独立状态；正式安装使用稳定的生产标识，不能换标识来规避旧数据兼容。

Windows 发行构建使用 `python tools/build_desktop.py`，隔离构建加 `--qa --no-bundle`。该入口仅为本次构建设置源码路径重映射和 PDB 路径，不改全局 Rust 配置。安装包在 Cargo target 的 `release/bundle/nsis` 下，默认当前用户安装。没有代码签名的构建须如实标明。

本机执行使用 Windows Job Objects，未测试系统保留未验证状态。公开前另用 `tools/audit_release.py` 检查允许发布的文件和资产，保存哈希。不能把一台开发电脑构建成功当作全平台安装通过。

旧候选的独立滚动页面、嵌入式 `public/open` 书目窗口及其构建脚本已退出活动构建。`site` 只保留共用获取、存储、内置计算与 service worker。新入口接续旧来源数据库中的笔记、草稿、收据；跨来源仍须在原地址导出，不能声称能读取其他网站的私有存储。
