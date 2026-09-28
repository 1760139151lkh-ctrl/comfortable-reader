use serde::{Deserialize, Serialize};
mod learning;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    env,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tauri::{ipc::Response, AppHandle, Manager};
use walkdir::WalkDir;
use zip::ZipArchive;

const STATE_FILE: &str = "reader-state.json";
const SHARED_ROOTS_DIR: &str = "ComfortableReader";
const SHARED_ROOTS_FILE: &str = "library-roots.json";
const MAX_SCAN_ISSUES: usize = 50;
#[derive(Default)]
struct ReaderWriteLock(std::sync::Mutex<()>);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BookRecord {
    #[serde(default)]
    modified_at: Option<u64>,
    #[serde(default)]
    available: Option<bool>,
    #[serde(default)]
    book_uuid: Option<String>,
    #[serde(default)]
    catalog_source: Option<serde_json::Value>,
    id: String,
    title: String,
    author: String,
    path: String,
    added_at: u64,
    /// Page count for visual/page-stream EPUBs. Older state files omit this.
    #[serde(default)]
    lazy_pages: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BookProgress {
    book_uuid: Option<String>,
    content_digest: Option<String>,
    reading_mode: Option<String>,
    /// The visible measurement is separate from legacy whole-file locations.
    reading_metric: Option<serde_json::Value>,
    /// Current source bytes. A changed edition keeps old annotations on hold.
    source_sha256: Option<String>,
    cfi: Option<String>,
    page: u32,
    total_pages: u32,
    percent: f64,
    updated_at: u64,
    /// 0 means automatic; 1..=10 pins that many visible pages for this book.
    page_mode: u8,
    /// Flexible, versioned reader annotations owned by the frontend. Keeping
    /// them with progress makes every pane and future session see the same notes.
    annotations: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReaderSession {
    #[serde(default)]
    line_height: Option<f64>,
    #[serde(default)]
    content_width: Option<u16>,
    pane_count: u8,
    pane_book_ids: Vec<Option<String>>,
    active_pane: usize,
    theme: String,
    font_scale: u16,
    #[serde(default = "default_reader_font")]
    reader_font: String,
}

fn default_reader_font() -> String {
    "serif".to_string()
}

impl Default for ReaderSession {
    fn default() -> Self {
        Self {
            line_height: None,
            content_width: None,
            pane_count: 1,
            pane_book_ids: vec![None, None, None, None],
            active_pane: 0,
            theme: "paper".to_string(),
            font_scale: 100,
            reader_font: default_reader_font(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct PersistedState {
    books: Vec<BookRecord>,
    progress: HashMap<String, BookProgress>,
    session: ReaderSession,
    library_roots: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSnapshot {
    books: Vec<BookRecord>,
    progress: HashMap<String, BookProgress>,
    session: ReaderSession,
    library_roots: Vec<String>,
    storage_path: String,
    scan: ScanReport,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanIssue {
    path: String,
    message: String,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanReport {
    roots_scanned: usize,
    missing_roots: usize,
    files_seen: usize,
    epub_candidates: usize,
    loaded_candidates: usize,
    books_loaded: usize,
    added: usize,
    updated: usize,
    duplicates: usize,
    unreadable: usize,
    other_book_files: usize,
    ignored_trees: usize,
    issues: Vec<ScanIssue>,
}

#[derive(Default)]
struct ScanContext {
    seen_files: HashSet<String>,
    report: ScanReport,
}

enum AddOutcome {
    Added,
    Updated,
    Unreadable(String),
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn storage_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let path = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("无法确定应用数据目录：{error}"))?;
    fs::create_dir_all(&path).map_err(|error| format!("无法创建应用数据目录：{error}"))?;
    Ok(path)
}

fn state_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(storage_dir(app)?.join(STATE_FILE))
}

fn shared_roots_path(app: &AppHandle) -> Result<PathBuf, String> {
    if app.config().identifier != "com.comfortablereader.desktop" {
        return Ok(storage_dir(app)?.join(SHARED_ROOTS_FILE));
    }
    let base = if let Some(path) = env::var_os("APPDATA") {
        PathBuf::from(path)
    } else {
        let app_config = app
            .path()
            .app_config_dir()
            .map_err(|error| format!("无法确定共享配置目录：{error}"))?;
        app_config
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or(app_config)
    };
    let directory = base.join(SHARED_ROOTS_DIR);
    fs::create_dir_all(&directory).map_err(|error| format!("无法创建共享书库目录：{error}"))?;
    Ok(directory.join(SHARED_ROOTS_FILE))
}

fn load_shared_roots(app: &AppHandle) -> Result<Vec<String>, String> {
    let path = shared_roots_path(app)?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw =
        fs::read_to_string(&path).map_err(|error| format!("无法读取共享书库清单：{error}"))?;
    serde_json::from_str(&raw).map_err(|error| format!("共享书库清单损坏，未自动覆盖：{error}"))
}

fn save_shared_roots(app: &AppHandle, roots: &[String]) -> Result<(), String> {
    let path = shared_roots_path(app)?;
    let raw = serde_json::to_vec_pretty(roots)
        .map_err(|error| format!("无法序列化共享书库清单：{error}"))?;
    fs::write(path, raw).map_err(|error| format!("无法保存共享书库清单：{error}"))
}

fn load_state(app: &AppHandle) -> Result<PersistedState, String> {
    let path = state_path(app)?;
    if !path.exists() {
        return Ok(PersistedState::default());
    }
    let raw = fs::read_to_string(&path).map_err(|error| format!("无法读取书库状态：{error}"))?;
    serde_json::from_str(&raw).map_err(|error| format!("书库状态文件损坏，未自动覆盖：{error}"))
}

fn save_state(app: &AppHandle, state: &PersistedState) -> Result<(), String> {
    let path = state_path(app)?;
    let raw =
        serde_json::to_vec_pretty(state).map_err(|error| format!("无法序列化书库状态：{error}"))?;
    let temporary = path.with_extension("writing");
    {
        use std::io::Write;
        let mut file =
            File::create(&temporary).map_err(|error| format!("无法准备书库状态：{error}"))?;
        file.write_all(&raw)
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("无法完整保存书库状态：{error}"))?;
    }
    fs::rename(temporary, path).map_err(|error| format!("无法切换书库状态：{error}"))
}

fn snapshot(
    app: &AppHandle,
    state: &PersistedState,
    scan: ScanReport,
) -> Result<AppSnapshot, String> {
    Ok(AppSnapshot {
        books: state.books.clone(),
        progress: state.progress.clone(),
        session: state.session.clone(),
        library_roots: state.library_roots.clone(),
        storage_path: storage_dir(app)?.to_string_lossy().to_string(),
        scan,
    })
}

fn normalized_path(path: &Path) -> Result<PathBuf, String> {
    fs::canonicalize(path).map_err(|error| format!("无法访问 {}：{error}", path.display()))
}

fn read_zip_text(archive: &mut ZipArchive<File>, name: &str) -> Option<String> {
    let mut entry = archive.by_name(name).ok()?;
    if entry.size()>4*1024*1024 {return None;}
    let mut raw = Vec::new();
    entry.read_to_end(&mut raw).ok()?;
    String::from_utf8(raw).ok()
}

fn opf_metadata(opf: &str) -> (Option<String>, Option<String>, Option<u32>) {
    let Ok(document) = roxmltree::Document::parse(opf) else {
        return (None, None, None);
    };
    let text_for = |name: &str| {
        document
            .descendants()
            .find(|node| node.is_element() && node.tag_name().name().eq_ignore_ascii_case(name))
            .and_then(|node| node.text())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let lazy_pages = document
        .descendants()
        .find(|node| {
            node.is_element()
                && node.tag_name().name().eq_ignore_ascii_case("meta")
                && node
                    .attribute("name")
                    .is_some_and(|name| name.eq_ignore_ascii_case("comfortable-reader-page-count"))
        })
        .and_then(|node| node.attribute("content"))
        .and_then(|content| content.parse::<u32>().ok())
        .filter(|count| *count > 0);
    let creators: Vec<_> = document
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name().eq_ignore_ascii_case("creator"))
        .filter_map(|node| node.text())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect();
    let author = (!creators.is_empty()).then(|| creators.join("; "));
    (text_for("title"), author, lazy_pages)
}

fn epub_metadata(path: &Path) -> Result<(Option<String>, Option<String>, Option<u32>), String> {
    let file = File::open(path).map_err(|error| format!("无法打开 EPUB 文件：{error}"))?;
    let mut archive =
        ZipArchive::new(file).map_err(|error| format!("不是可读取的 EPUB/ZIP：{error}"))?;

    let opf_path = read_zip_text(&mut archive, "META-INF/container.xml")
        .and_then(|container| {
            roxmltree::Document::parse(&container).ok().and_then(|doc| {
                doc.descendants()
                    .find(|node| node.is_element() && node.tag_name().name() == "rootfile")
                    .and_then(|node| node.attribute("full-path"))
                    .map(str::to_string)
            })
        })
        .or_else(|| {
            (0..archive.len()).find_map(|index| {
                let name = archive.by_index(index).ok()?.name().to_string();
                name.to_ascii_lowercase().ends_with(".opf").then_some(name)
            })
        });

    let opf_path = opf_path.ok_or_else(|| "EPUB 缺少 OPF 包文档".to_string())?;
    Ok(read_zip_text(&mut archive, &opf_path)
        .map(|opf| opf_metadata(&opf))
        .unwrap_or((None, None, None)))
}

fn book_from_path(path: &Path) -> Result<BookRecord, String> {
    let path = normalized_path(path)?;
    if path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        != Some("epub".to_string())
    {
        return Err(format!("只支持 EPUB：{}", path.display()));
    }
    let (title, author, lazy_pages) = epub_metadata(&path)?;
    let fallback_title = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("未命名书籍")
        .to_string();
    let book_uuid=epub_identifier(&path);
    let id=if let Some(uuid)=&book_uuid {hex::encode(Sha256::digest(format!("epub-identifier-v1:{}",uuid.to_lowercase()).as_bytes()))[..24].to_string()}
        else {let mut file=File::open(&path).map_err(|e|e.to_string())?;let mut hash=Sha256::new();let mut buffer=[0u8;65536];loop{let count=file.read(&mut buffer).map_err(|e|e.to_string())?;if count==0{break;}hash.update(&buffer[..count]);}hex::encode(hash.finalize())[..24].to_string()};
    Ok(BookRecord {
        modified_at:fs::metadata(&path).ok().and_then(|m|m.modified().ok()).and_then(|t|t.duration_since(UNIX_EPOCH).ok()).map(|d|d.as_millis() as u64),
        available:Some(true),
        book_uuid,
        catalog_source: None,
        id,
        title: title.unwrap_or(fallback_title),
        author: author.unwrap_or_else(|| "未知作者".to_string()),
        path: path.to_string_lossy().to_string(),
        added_at: now_secs(),
        lazy_pages,
    })
}

fn add_epub_file_outcome(state: &mut PersistedState, path: &Path, prefer_imported:bool) -> AddOutcome {
    let book = match book_from_path(path) {
        Ok(book) => book,
        Err(error) => return AddOutcome::Unreadable(error),
    };
    if let Some(existing) = state
        .books
        .iter_mut()
        .find(|existing| existing.path.eq_ignore_ascii_case(&book.path)||existing.id == book.id||existing.book_uuid.is_some()&&existing.book_uuid==book.book_uuid&&!Path::new(&existing.path).is_file())
    {
        if !prefer_imported&&!existing.path.eq_ignore_ascii_case(&book.path)&&(existing.path.starts_with("https://")||existing.path.starts_with("http://")||Path::new(&existing.path).is_file()){return AddOutcome::Updated;}
        // A previous build could have persisted metadata after it was decoded by
        // the wrong Windows code page. Refreshing the library must repair that
        // stale record from the UTF-8 OPF instead of permanently keeping it.
        existing.title = book.title;
        existing.author = book.author;
        existing.path = book.path;
        existing.lazy_pages = book.lazy_pages;
        existing.book_uuid = book.book_uuid;
        existing.modified_at=book.modified_at;existing.available=Some(true);
        return AddOutcome::Updated;
    }
    state.books.push(book);
    AddOutcome::Added
}

#[cfg(test)]
fn add_epub_file(state: &mut PersistedState, path: &Path) -> bool {
    matches!(add_epub_file_outcome(state, path, true), AddOutcome::Added)
}

fn push_issue(report: &mut ScanReport, path: &Path, message: impl Into<String>) {
    if report.issues.len() < MAX_SCAN_ISSUES {
        report.issues.push(ScanIssue {
            path: path.to_string_lossy().to_string(),
            message: message.into(),
        });
    }
}

fn is_ignored_directory_name(name: &std::ffi::OsStr) -> bool {
    let name = name.to_string_lossy();
    name.eq_ignore_ascii_case(".caltrash")
        || name.eq_ignore_ascii_case("$recycle.bin")
        || name.eq_ignore_ascii_case("recycler")
}

fn is_ignored_book_path(path: &Path) -> bool {
    path.components().any(|component| match component {
        std::path::Component::Normal(name) => is_ignored_directory_name(name),
        _ => false,
    })
}

fn is_other_book_file(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|value| value.to_str()) else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "azw"
            | "azw3"
            | "cb7"
            | "cbr"
            | "cbz"
            | "djvu"
            | "docx"
            | "fb2"
            | "html"
            | "htm"
            | "md"
            | "mobi"
            | "pdf"
            | "rtf"
            | "txt"
    )
}

fn scan_file(state: &mut PersistedState, path: &Path, context: &mut ScanContext) {
    context.report.files_seen += 1;
    let is_epub = path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("epub"));
    if !is_epub {
        if is_other_book_file(path) {
            context.report.other_book_files += 1;
        }
        return;
    }

    let normalized = match normalized_path(path) {
        Ok(path) => path,
        Err(error) => {
            context.report.epub_candidates += 1;
            context.report.unreadable += 1;
            push_issue(&mut context.report, path, error);
            return;
        }
    };
    let key = normalized
        .to_string_lossy()
        .replace('\\', "/")
        .to_lowercase();
    if !context.seen_files.insert(key) {
        context.report.duplicates += 1;
        return;
    }

    context.report.epub_candidates += 1;
    match add_epub_file_outcome(state, &normalized, false) {
        AddOutcome::Added => {
            context.report.added += 1;
            context.report.loaded_candidates += 1;
        }
        AddOutcome::Updated => {
            context.report.updated += 1;
            context.report.loaded_candidates += 1;
        }
        AddOutcome::Unreadable(error) => {
            context.report.unreadable += 1;
            push_issue(&mut context.report, &normalized, error);
        }
    }
}

fn scan_root(state: &mut PersistedState, root: &Path, context: &mut ScanContext) {
    if root.is_file() {
        scan_file(state, root, context);
        return;
    }
    if !root.is_dir() {
        context.report.missing_roots += 1;
        push_issue(&mut context.report, root, "书库目录当前不可用");
        return;
    }
    context.report.roots_scanned += 1;
    let mut walker = WalkDir::new(root).follow_links(true).into_iter();
    while let Some(result) = walker.next() {
        match result {
            Ok(entry) => {
                if entry.depth() > 0
                    && entry.file_type().is_dir()
                    && is_ignored_directory_name(entry.file_name())
                {
                    context.report.ignored_trees += 1;
                    walker.skip_current_dir();
                    continue;
                }
                if entry.file_type().is_file() {
                    scan_file(state, entry.path(), context);
                }
            }
            Err(error) => {
                let path = error.path().unwrap_or(root);
                push_issue(&mut context.report, path, format!("扫描失败：{error}"));
            }
        }
    }
}

fn default_library_roots() -> Vec<PathBuf> { Vec::new() }

fn refresh_state(app: &AppHandle, state: &mut PersistedState) -> Result<ScanReport, String> {
    for book in &mut state.books{
        if book.path.starts_with("https://")||book.path.starts_with("http://"){continue;}
        if let Ok(current)=book_from_path(Path::new(&book.path)){book.title=current.title;book.author=current.author;book.book_uuid=current.book_uuid;book.modified_at=current.modified_at;book.available=Some(true);book.lazy_pages=current.lazy_pages;}
        else{book.available=Some(false);}
    }
    state.books.retain(|book| {
        if book.catalog_source.is_some() { return true; }
        let path = Path::new(&book.path);
        !is_ignored_book_path(path)
    });

    let shared_roots = load_shared_roots(app)?;
    let mut registered_roots: Vec<PathBuf> =
        state.library_roots.iter().map(PathBuf::from).collect();
    registered_roots.extend(shared_roots.iter().map(PathBuf::from));
    let mut roots = registered_roots.clone();
    roots.extend(
        default_library_roots()
            .into_iter()
            .filter(|root| root.is_dir()),
    );
    let mut seen = HashSet::new();
    let mut context = ScanContext::default();
    for root in roots {
        if !root.is_dir() {
            if registered_roots
                .iter()
                .any(|registered| registered == &root)
            {
                context.report.missing_roots += 1;
                push_issue(&mut context.report, &root, "已登记的书库目录当前不可用");
            }
            continue;
        }
        let Ok(root) = normalized_path(&root) else {
            continue;
        };
        let key = root.to_string_lossy().to_lowercase();
        if seen.insert(key) {
            scan_root(state, &root, &mut context);
            let root_string = root.to_string_lossy().to_string();
            if !state
                .library_roots
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(&root_string))
            {
                state.library_roots.push(root_string);
            }
        }
    }

    state
        .books
        .sort_by(|left, right| left.title.cmp(&right.title));
    state.library_roots.sort_by_key(|left| left.to_lowercase());
    state
        .library_roots
        .dedup_by(|left, right| left.eq_ignore_ascii_case(right));
    save_shared_roots(app, &state.library_roots)?;
    context.report.books_loaded = state.books.len();
    Ok(context.report)
}

#[tauri::command]
fn bootstrap(app: AppHandle) -> Result<AppSnapshot, String> {
    let lock = app.state::<ReaderWriteLock>();
    let _guard = lock.0.lock().map_err(|_| "书库写入锁不可用")?;
    let mut state = load_state(&app)?;
    let scan = refresh_state(&app, &mut state)?;
    save_state(&app, &state)?;
    snapshot(&app, &state, scan)
}

#[tauri::command]
fn import_paths(app: AppHandle, paths: Vec<String>) -> Result<AppSnapshot, String> {
    let lock = app.state::<ReaderWriteLock>();
    let _guard = lock.0.lock().map_err(|_| "书库写入锁不可用")?;
    let mut state = load_state(&app)?;
    for raw in paths {
        let path = normalized_path(Path::new(&raw))?;
        if path.is_dir() {
            let root = path.to_string_lossy().to_string();
            if !state
                .library_roots
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(&root))
            {
                state.library_roots.push(root);
            }
        } else {
            match add_epub_file_outcome(&mut state, &path, true) {
                AddOutcome::Added | AddOutcome::Updated => {}
                AddOutcome::Unreadable(error) => {
                    return Err(format!("无法把 {} 加入书库：{error}", path.display()));
                }
            }
        }
    }
    let scan = refresh_state(&app, &mut state)?;
    save_state(&app, &state)?;
    snapshot(&app, &state, scan)
}

#[tauri::command]
fn refresh_library(app: AppHandle) -> Result<AppSnapshot, String> {
    let lock = app.state::<ReaderWriteLock>();
    let _guard = lock.0.lock().map_err(|_| "书库写入锁不可用")?;
    let mut state = load_state(&app)?;
    let scan = refresh_state(&app, &mut state)?;
    save_state(&app, &state)?;
    snapshot(&app, &state, scan)
}

fn epub_identifier(path: &Path) -> Option<String> {
    let file=File::open(path).ok()?;
    let mut archive=ZipArchive::new(file).ok()?;
    let container=read_zip_text(&mut archive,"META-INF/container.xml")?;
    let xml=roxmltree::Document::parse(&container).ok()?;
    let name=xml.descendants().find(|n|n.has_tag_name("rootfile"))?.attribute("full-path")?;
    let opf=read_zip_text(&mut archive,name)?;
    let doc=roxmltree::Document::parse(&opf).ok()?;
    let wanted=doc.root_element().attribute("unique-identifier");
    doc.descendants().find(|n|n.is_element()&&n.tag_name().name()=="identifier"&&wanted.map_or(true,|id|n.attribute("id")==Some(id)))?.text().map(|s|s.trim().to_lowercase())
}

#[tauri::command]
fn register_catalog_book(app:AppHandle,mut record:BookRecord,replace_existing:Option<bool>)->Result<BookRecord,String>{
    let source=record.catalog_source.as_ref().ok_or("书目来源未登记")?;
    let url=tauri::Url::parse(source["url"].as_str().ok_or("书目地址无效")?).map_err(|_|"书目地址无效")?;
    let local=url.scheme()=="http"&&matches!(url.host_str(),Some("127.0.0.1"|"localhost"));
    if (url.scheme()!="https"&&!local)||!url.username().is_empty()||url.password().is_some(){return Err("只允许无凭据的 HTTPS 书目或本机预览".into());}
    let uuid=record.book_uuid.as_deref().ok_or("书籍身份缺失")?;
    if !uuid.starts_with("urn:uuid:")||uuid.len()!=45||!uuid[9..].chars().all(|c|c.is_ascii_hexdigit()||c=='-'){return Err("书籍 UUID 无效".into());}
    let hash=source["sha256"].as_str().ok_or("书目校验身份缺失")?;
    if hash.len()!=64||!hash.chars().all(|c|c.is_ascii_hexdigit())||source["bytes"].as_u64().filter(|n|*n<=2*1024*1024).is_none(){return Err("书目大小或哈希无效".into());}
    if record.title.len()>2000||record.author.len()>2000{return Err("书目信息过长".into());}
    record.id=hex::encode(Sha256::digest(format!("epub-identifier-v1:{}",uuid.to_lowercase()).as_bytes()))[..24].to_string();
    record.path=url.to_string();record.lazy_pages=None;record.added_at=now_secs();
    let lock=app.state::<ReaderWriteLock>();let _guard=lock.0.lock().map_err(|_|"书库写入锁不可用")?;
    let mut state=load_state(&app)?;
    // Existing local books retain their installed identity, bytes and learning binding.
    if let Some(existing)=state.books.iter_mut().find(|b|b.book_uuid==record.book_uuid||b.id==record.id){
        if replace_existing==Some(true)||existing.path.starts_with("https://")||existing.path.starts_with("http://"){record.id=existing.id.clone();record.added_at=existing.added_at;*existing=record.clone();}else{
            let declared=record.catalog_source.as_ref().and_then(|s|s["epub"]["sha256"].as_str());
            if let Some(expected)=declared{if let Ok(bytes)=fs::read(&existing.path){if hex::encode(Sha256::digest(&bytes))==expected{existing.catalog_source=record.catalog_source.clone();record=existing.clone();}else{return Ok(existing.clone());}}else{return Ok(existing.clone());}}else{return Ok(existing.clone());}
        }
    }else{state.books.push(record.clone());}
    save_state(&app,&state)?;Ok(record)
}

#[tauri::command]
fn load_book_bytes(app: AppHandle, book_id: String) -> Result<Response, String> {
    let state = load_state(&app)?;
    let book = state
        .books
        .iter()
        .find(|book| book.id == book_id)
        .ok_or_else(|| "书籍不在本地书库中".to_string())?;
    let bytes = fs::read(&book.path).map_err(|error| format!("无法读取书籍：{error}"))?;
    Ok(Response::new(bytes))
}

/// Canonical member identities bind streamed XHTML to the exact local EPUB.
/// A publisher's claimed whole-ZIP hash alone is not sufficient for note migration.
#[tauri::command]
fn epub_content_identity(app:AppHandle,book_id:String,source_sha256:String)->Result<String,String>{
    let cache=location_index_path(&app,&source_sha256)?.with_file_name(format!("{source_sha256}-content-v1.json"));
    if let Ok(value)=fs::read_to_string(&cache){if value.len()==64&&value.chars().all(|c|c.is_ascii_hexdigit()){return Ok(value);}}
    let state=load_state(&app)?;let book=state.books.iter().find(|b|b.id==book_id&&Path::new(&b.path).is_file()).ok_or("本机书籍未登记")?;
    let bytes=fs::read(&book.path).map_err(|e|e.to_string())?;
    if hex::encode(Sha256::digest(&bytes))!=source_sha256{return Err("文件在读取期间改变，未建立跨端身份".into());}
    let mut archive=ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e|e.to_string())?;
    if archive.len()>15000{return Err("内容文件数超出核对范围".into());}
    let mut identities=Vec::new();let mut total=0u64;
    for i in 0..archive.len(){let mut entry=archive.by_index(i).map_err(|e|e.to_string())?;let name=entry.name().to_string();if entry.is_dir()||name=="mimetype"{continue;}total+=entry.size();if entry.size()>64*1024*1024||total>512*1024*1024{return Err("内容展开大小超出身份核对范围".into());}let mut hash=Sha256::new();let mut buffer=[0u8;65536];loop{let count=entry.read(&mut buffer).map_err(|e|e.to_string())?;if count==0{break;}hash.update(&buffer[..count]);}identities.push((name,entry.size(),hex::encode(hash.finalize())));}
    identities.sort_by(|a,b|a.0.cmp(&b.0));
    let canonical=serde_json::to_vec(&identities).map_err(|e|e.to_string())?;
    let digest=hex::encode(Sha256::digest(canonical));fs::create_dir_all(cache.parent().unwrap()).map_err(|e|e.to_string())?;fs::write(cache,&digest).map_err(|e|e.to_string())?;Ok(digest)
}

#[tauri::command]
fn load_book_entry(app: AppHandle, book_id: String, entry: String) -> Result<Response, String> {
    let state = load_state(&app)?;
    let book = state
        .books
        .iter()
        .find(|book| book.id == book_id)
        .ok_or_else(|| "书籍不在本地书库中".to_string())?;
    if !entry.starts_with("OEBPS/images/page-") || !entry.ends_with(".svg") || entry.contains("..")
    {
        return Err("不允许读取书籍之外的资源".to_string());
    }
    let file = File::open(&book.path).map_err(|error| format!("无法打开书籍：{error}"))?;
    let mut archive =
        ZipArchive::new(file).map_err(|error| format!("书籍 ZIP 无法读取：{error}"))?;
    let mut resource = archive
        .by_name(&entry)
        .map_err(|error| format!("书籍页面资源不存在：{error}"))?;
    if resource.size() > 32 * 1024 * 1024 {
        return Err("书籍页面资源过大，已拒绝加载".to_string());
    }
    let mut bytes = Vec::with_capacity(resource.size() as usize);
    resource
        .read_to_end(&mut bytes)
        .map_err(|error| format!("无法读取书籍页面资源：{error}"))?;
    Ok(Response::new(bytes))
}

#[tauri::command]
fn save_progress(
    app: AppHandle,
    book_id: String,
    mut progress: BookProgress,
) -> Result<(), String> {
    let lock = app.state::<ReaderWriteLock>();
    let _guard = lock.0.lock().map_err(|_| "书库写入锁不可用")?;
    let mut state = load_state(&app)?;
    if !state.books.iter().any(|book| book.id == book_id) {
        return Err("无法保存不在书库中的书籍进度".to_string());
    }
    progress.percent = progress.percent.clamp(0.0, 1.0);
    progress.page_mode = progress.page_mode.min(10);
    progress.updated_at = now_secs();
    if let Some(previous)=state.progress.get(&book_id){save_edition(&app,&book_id,previous)?;}
    save_edition(&app,&book_id,&progress)?;
    state.progress.insert(book_id, progress);
    save_state(&app, &state)
}

fn edition_root(app:&AppHandle,book_id:&str)->Result<PathBuf,String>{
    if book_id.is_empty()||book_id.len()>120||!book_id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'||b==b'_'){return Err("书籍身份无效".into());}
    Ok(storage_dir(app)?.join("editions").join(book_id))
}
fn edition_key(progress:&BookProgress)->Option<&str>{progress.content_digest.as_deref().or(progress.source_sha256.as_deref()).filter(|key|key.len()==64&&key.bytes().all(|b|b.is_ascii_hexdigit()))}
fn save_edition(app:&AppHandle,book_id:&str,progress:&BookProgress)->Result<(),String>{
    let Some(key)=edition_key(progress)else{return Ok(())};let root=edition_root(app,book_id)?;fs::create_dir_all(&root).map_err(|e|e.to_string())?;
    let file=root.join(format!("{key}.json"));let temp=root.join(format!("{key}.writing"));fs::write(&temp,serde_json::to_vec(progress).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;fs::rename(temp,file).map_err(|e|e.to_string())
}
#[tauri::command]
fn activate_progress_edition(app:AppHandle,book_id:String,source_sha256:String,content_digest:Option<String>)->Result<BookProgress,String>{
    location_index_path(&app,&source_sha256)?;if let Some(key)=&content_digest{location_index_path(&app,key)?;}
    let lock=app.state::<ReaderWriteLock>();let _guard=lock.0.lock().map_err(|_|"书库写入锁不可用")?;let mut state=load_state(&app)?;
    let book=state.books.iter().find(|b|b.id==book_id).ok_or("书籍未登记")?.clone();
    let previous=state.progress.get(&book_id).cloned().unwrap_or_default();save_edition(&app,&book_id,&previous)?;
    let root=edition_root(&app,&book_id)?;
    let mut selected=None;
    for key in content_digest.iter().chain(std::iter::once(&source_sha256)){
        if let Ok(bytes)=fs::read(root.join(format!("{key}.json"))){if bytes.len()<=16*1024*1024{if let Ok(progress)=serde_json::from_slice::<BookProgress>(&bytes){if progress.content_digest==content_digest&&content_digest.is_some()||progress.source_sha256.as_deref()==Some(source_sha256.as_str()){selected=Some(progress);break;}}}}
    }
    let mut progress=selected.unwrap_or_default();progress.page_mode=previous.page_mode;progress.reading_mode=previous.reading_mode;progress.source_sha256=Some(source_sha256);progress.content_digest=content_digest;progress.book_uuid=book.book_uuid;progress.updated_at=now_secs();
    save_edition(&app,&book_id,&progress)?;state.progress.insert(book_id,progress.clone());save_state(&app,&state)?;Ok(progress)
}
#[tauri::command]
fn progress_editions(app:AppHandle,book_id:String)->Result<Vec<serde_json::Value>,String>{
    let root=edition_root(&app,&book_id)?;let mut rows=Vec::new();if !root.exists(){return Ok(rows);}
    for entry in fs::read_dir(root).map_err(|e|e.to_string())?.filter_map(Result::ok){if entry.path().extension().and_then(|e|e.to_str())!=Some("json")||entry.metadata().map(|m|m.len()>16*1024*1024).unwrap_or(true){continue;}
      if let Ok(bytes)=fs::read(entry.path()){if let Ok(progress)=serde_json::from_slice::<BookProgress>(&bytes){if let Some(key)=edition_key(&progress){rows.push(serde_json::json!({"key":key,"progress":progress}));}}}
    }rows.sort_by_key(|r|std::cmp::Reverse(r["progress"]["updatedAt"].as_u64().unwrap_or(0)));Ok(rows)
}

fn location_index_path(app:&AppHandle,source:&str)->Result<PathBuf,String>{
    if source.len()!=64||!source.chars().all(|c|c.is_ascii_hexdigit()){return Err("内容校验身份无效".into());}
    Ok(storage_dir(app)?.join("location-index").join(format!("{}-epubjs-1000-v1.json",source.to_lowercase())))
}
#[tauri::command]
fn load_location_index(app:AppHandle,source_sha256:String)->Result<Option<String>,String>{
    let path=location_index_path(&app,&source_sha256)?;
    if !path.exists(){return Ok(None);}
    if fs::metadata(&path).map_err(|e|e.to_string())?.len()>4*1024*1024{return Ok(None);}
    Ok(Some(fs::read_to_string(path).map_err(|e|e.to_string())?))
}
#[tauri::command]
fn save_location_index(app:AppHandle,source_sha256:String,locations:String)->Result<(),String>{
    if locations.len()>4*1024*1024{return Err("阅读位置表超过范围".into());}
    let entries:Vec<String>=serde_json::from_str(&locations).map_err(|e|e.to_string())?;
    if entries.len()>100000||entries.iter().any(|c|c.len()>1000||!c.starts_with("epubcfi(")){return Err("阅读位置表无效".into());}
    let lock=app.state::<ReaderWriteLock>();let _guard=lock.0.lock().map_err(|_|"位置表写入锁不可用")?;
    let path=location_index_path(&app,&source_sha256)?;
    fs::create_dir_all(path.parent().unwrap()).map_err(|e|e.to_string())?;
    let temp=path.with_extension("writing");fs::write(&temp,locations).map_err(|e|e.to_string())?;fs::rename(temp,path).map_err(|e|e.to_string())
}

#[tauri::command]
fn save_session(app: AppHandle, mut session: ReaderSession) -> Result<(), String> {
    let lock = app.state::<ReaderWriteLock>();
    let _guard = lock.0.lock().map_err(|_| "书库写入锁不可用")?;
    session.pane_count = session.pane_count.clamp(1, 4);
    session.active_pane = session.active_pane.min(session.pane_count as usize - 1);
    session.font_scale = session.font_scale.clamp(70, 180);
    if !["serif", "sans", "publisher"].contains(&session.reader_font.as_str()) {
        session.reader_font = default_reader_font();
    }
    session.pane_book_ids.resize(4, None);
    session.pane_book_ids.truncate(4);
    let mut state = load_state(&app)?;
    state.session = session;
    save_state(&app, &state)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(ReaderWriteLock::default())
        .manage(learning::LearningManager::default())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            bootstrap,
            import_paths,
            refresh_library,
            register_catalog_book,
            load_location_index,
            save_location_index,
            load_book_bytes,
            epub_content_identity,
            load_book_entry,
            save_progress,
            activate_progress_edition,
            progress_editions,
            save_session,
            learning::learning_pack,
            learning::learning_asset,
            learning::learning_load_state,
            learning::learning_save_state,
            learning::learning_save_draft,
            learning::learning_draft,
            learning::learning_import_draft,
            learning::learning_authorize_edit,
            learning::learning_run,
            learning::learning_run_status,
            learning::learning_run_history,
            learning::learning_run_snapshot,
            learning::learning_cancel,
            learning::learning_artifact,
            learning::learning_open_source,
            learning::learning_authorize_training,
            learning::learning_draft_versions,
            learning::learning_restore_draft,
            learning::learning_export,
            learning::learning_runtime_info,
            learning::learning_resource_info
            ,learning::learning_store_builtin_run
            ,learning::learning_open_external
        ])
        .run(tauri::generate_context!())
        .expect("舒适阅读书库启动失败");
}

#[cfg(test)]
mod tests {
    use super::{add_epub_file, opf_metadata, scan_root, BookProgress, ScanContext};
    use std::path::Path;
    use std::{fs::File, io::Write};
    use tempfile::tempdir;
    use zip::{write::SimpleFileOptions, ZipWriter};

    fn create_epub(path: &Path, title: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create epub parent");
        }
        let file = File::create(path).expect("create epub");
        let mut archive = ZipWriter::new(file);
        let options = SimpleFileOptions::default();
        archive
            .start_file("META-INF/container.xml", options)
            .expect("container entry");
        archive
            .write_all(
                br#"<?xml version="1.0"?><container><rootfiles><rootfile full-path="content.opf"/></rootfiles></container>"#,
            )
            .expect("container content");
        archive
            .start_file("content.opf", options)
            .expect("opf entry");
        archive
            .write_all(
                format!(
                    r#"<?xml version="1.0"?><package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>{title}</dc:title><dc:creator>作者</dc:creator></metadata></package>"#
                )
                .as_bytes(),
            )
            .expect("opf content");
        archive.finish().expect("finish epub");
    }

    #[test]
    fn legacy_session_keeps_books_theme_and_size_when_font_role_is_added() {
        let raw = r#"{"paneCount":2,"paneBookIds":["first","second",null,null],"activePane":1,"theme":"night","fontScale":120}"#;
        let session: super::ReaderSession = serde_json::from_str(raw).expect("legacy session");
        assert_eq!(session.pane_count, 2);
        assert_eq!(session.pane_book_ids[1].as_deref(), Some("second"));
        assert_eq!(session.active_pane, 1);
        assert_eq!(session.theme, "night");
        assert_eq!(session.font_scale, 120);
        assert_eq!(session.reader_font, "serif");
        let encoded = serde_json::to_value(&session).unwrap();
        assert_eq!(encoded["readerFont"], "serif");
    }

    #[test]
    fn extracts_namespaced_epub_metadata() {
        let opf = r#"<?xml version="1.0"?>
          <package xmlns:dc="http://purl.org/dc/elements/1.1/">
            <metadata><dc:title>数学建模九步法</dc:title><dc:creator>ChatGPT</dc:creator><dc:creator> </dc:creator><dc:creator>Codex</dc:creator><meta name="comfortable-reader-page-count" content="1677"/></metadata>
          </package>"#;
        let (title, author, lazy_pages) = opf_metadata(opf);
        assert_eq!(title.as_deref(), Some("数学建模九步法"));
        assert_eq!(author.as_deref(), Some("ChatGPT; Codex"));
        assert_eq!(lazy_pages, Some(1677));
    }

    #[test]
    fn old_progress_without_page_mode_defaults_to_automatic() {
        let raw = r#"{
          "cfi": null,
          "page": 12,
          "totalPages": 120,
          "percent": 0.1,
          "updatedAt": 100
        }"#;
        let progress: BookProgress = serde_json::from_str(raw).expect("old progress should load");
        assert_eq!(progress.page_mode, 0);
        assert!(progress.annotations.is_empty());
    }

    #[test]
    fn annotations_round_trip_without_losing_style_or_points() {
        let raw = r##"{
          "cfi": "epubcfi(/6/2!/4/2/1:0)",
          "page": 3,
          "totalPages": 18,
          "percent": 0.12,
          "updatedAt": 100,
          "pageMode": 3,
          "annotations": [{
            "id": "note-1",
            "kind": "drawing",
            "color": "#d8913d",
            "strokeWidth": 3,
            "points": [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}]
          }]
        }"##;
        let progress: BookProgress = serde_json::from_str(raw).expect("annotations should load");
        assert_eq!(progress.annotations.len(), 1);
        let encoded = serde_json::to_string(&progress).expect("annotations should serialize");
        assert!(encoded.contains("\"kind\":\"drawing\""));
        assert!(encoded.contains("\"points\""));
    }

    #[test]
    fn rescanning_repairs_stale_book_metadata() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("book.epub");
        create_epub(&path, "正确中文书名");

        let mut state = super::PersistedState::default();
        assert!(add_epub_file(&mut state, &path));
        state.books[0].title = "��������".to_string();
        assert!(!add_epub_file(&mut state, &path));
        assert_eq!(state.books[0].title, "正确中文书名");
        assert_eq!(state.books[0].author, "作者");
    }

    #[test]
    fn recursively_loads_every_epub_but_not_calibre_trash() {
        let directory = tempdir().expect("temporary directory");
        create_epub(&directory.path().join("作者甲/书一.epub"), "书一");
        create_epub(&directory.path().join("作者乙/深层/书二.EPUB"), "书二");
        create_epub(
            &directory.path().join(".caltrash/b/9/已删除.epub"),
            "已删除",
        );

        let mut state = super::PersistedState::default();
        let mut context = ScanContext::default();
        scan_root(&mut state, directory.path(), &mut context);

        assert_eq!(context.report.roots_scanned, 1);
        assert_eq!(context.report.epub_candidates, 2);
        assert_eq!(context.report.loaded_candidates, 2);
        assert_eq!(context.report.unreadable, 0);
        assert_eq!(context.report.ignored_trees, 1);
        assert_eq!(state.books.len(), 2);
        assert!(state.books.iter().any(|book| book.title == "书一"));
        assert!(state.books.iter().any(|book| book.title == "书二"));
    }

    #[test]
    fn reports_unreadable_and_non_epub_books_instead_of_silently_dropping_them() {
        let directory = tempdir().expect("temporary directory");
        std::fs::write(directory.path().join("损坏.epub"), b"not an epub")
            .expect("write broken epub");
        std::fs::write(directory.path().join("尚未转换.pdf"), b"pdf fixture")
            .expect("write pdf fixture");

        let mut state = super::PersistedState::default();
        let mut context = ScanContext::default();
        scan_root(&mut state, directory.path(), &mut context);

        assert_eq!(context.report.epub_candidates, 1);
        assert_eq!(context.report.loaded_candidates, 0);
        assert_eq!(context.report.unreadable, 1);
        assert_eq!(context.report.other_book_files, 1);
        assert_eq!(context.report.issues.len(), 1);
        assert!(state.books.is_empty());
    }

    #[test]
    fn overlapping_roots_do_not_duplicate_a_book() {
        let directory = tempdir().expect("temporary directory");
        create_epub(&directory.path().join("作者/唯一.epub"), "唯一");

        let mut state = super::PersistedState::default();
        let mut context = ScanContext::default();
        scan_root(&mut state, directory.path(), &mut context);
        scan_root(&mut state, directory.path(), &mut context);

        assert_eq!(context.report.epub_candidates, 1);
        assert_eq!(context.report.loaded_candidates, 1);
        assert_eq!(context.report.duplicates, 1);
        assert_eq!(state.books.len(), 1);
    }
}
