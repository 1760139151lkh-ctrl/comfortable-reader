//! Learning assets are data; only the separately registered native recipes execute.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
#[cfg(windows)]
use std::os::windows::{fs::MetadataExt, io::AsRawHandle, process::CommandExt};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tauri::{ipc::Response, AppHandle, Manager, State};
#[cfg(windows)]
use windows_sys::Win32::{Foundation::CloseHandle, System::JobObjects::*};

const WORKER: &str = include_str!("learning_worker.py");
const MAX_ASSET: u64 = 64 * 1024 * 1024;
const MAX_LOG: usize = 2 * 1024 * 1024;
#[tauri::command]
pub fn learning_runtime_info(app: AppHandle) -> Result<Value, String> {
    let path = std::env::current_exe().map_err(|e| e.to_string())?;
    let bytes = fs::read(&path).map_err(|e| e.to_string())?;
    Ok(
        json!({"version":app.package_info().version.to_string(),"identifier":app.config().identifier,"process_id":std::process::id(),"executable_path":path.to_string_lossy(),"executable_sha256":digest(&bytes),"manifest_versions":[1],"book_scripts_enabled":false,"native_arbitrary_code_mode":"explicit_high_trust_per_revision","transport":"private Tauri IPC; no execution HTTP service","learning_data_root":state_root(&app)?.to_string_lossy(),"learning_bindings_path":binding_path(&app)?.to_string_lossy()}),
    )
}
#[derive(Default)]
pub struct LearningManager {
    inner: Mutex<HashMap<String, Arc<Mutex<Running>>>>,
}
struct Running {
    child: Child,
    job: isize,
    cancelled: bool,
    folder: PathBuf,
    report: Value,
}
struct PreparationReceipt {
    path: PathBuf,
    record: Value,
    armed: bool,
}
impl Drop for PreparationReceipt {
    fn drop(&mut self) {
        if self.armed {
            self.record["status"] = json!("failed");
            self.record["diagnostic"] =
                json!("准备或启动未完成，没有自动重试；请核对界面中的具体错误");
            let _ = atomic(&self.path, &self.record);
        }
    }
}
// HANDLE is used only under the run mutex and closed by the supervisor.
struct Context {
    binding: Value,
    pack: Value,
    state: PathBuf,
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn same_python_text(left: &[u8], right: &[u8]) -> bool {
    std::str::from_utf8(left)
        .ok()
        .zip(std::str::from_utf8(right).ok())
        .is_some_and(|(a, b)| {
            a.replace("\r\n", "\n").replace('\r', "\n")
                == b.replace("\r\n", "\n").replace('\r', "\n")
        })
}

fn text<'a>(v: &'a Value, k: &str) -> Result<&'a str, String> {
    v[k].as_str().ok_or_else(|| format!("缺少必要字段 {k}"))
}
fn token(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 160
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
    {
        Err("无效的对象标识".into())
    } else {
        Ok(())
    }
}
fn read_json(path: &Path, limit: u64) -> Result<Value, String> {
    let m = fs::metadata(path).map_err(|e| format!("资料尚未就绪：{e}"))?;
    if m.len() > limit {
        return Err("资料超过读取预算".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| format!("资料格式错误：{e}"))
}
fn atomic(path: &Path, value: &Value) -> Result<(), String> {
    let parent = path.parent().ok_or("无效保存位置")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("writing");
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    let mut file = File::create(&tmp).map_err(|e| e.to_string())?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    fs::rename(tmp, path).map_err(|e| e.to_string())
}
fn state_root(app: &AppHandle) -> Result<PathBuf, String> {
    if app.config().identifier == "com.comfortablereader.desktop" {
        Ok(learning_home(app)?.join("records"))
    } else {
        Ok(super::storage_dir(app)?.join("learning"))
    }
}
fn learning_home(app: &AppHandle) -> Result<PathBuf, String> {
    if app.config().identifier == "com.comfortablereader.desktop" {
        Ok(app
            .path()
            .home_dir()
            .map_err(|e| e.to_string())?
            .join(".comfortable-reader"))
    } else {
        super::storage_dir(app)
    }
}
fn binding_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(learning_home(app)?.join("learning-bindings.json"))
}
fn user_storage(app: &AppHandle, book_id: &str) -> Result<PathBuf, String> {
    token(book_id)?;
    Ok(state_root(app)?.join(book_id))
}
fn saved_draft_path(root: &Path, activity: &str) -> Result<PathBuf, String> {
    token(activity)?;
    Ok(root.join("drafts").join(format!("{activity}.json")))
}

fn read_draft_revision(root: &Path, activity: &str, revision: &str) -> Result<Value, String> {
    token(activity)?;
    if revision.len() != 64 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("无效的代码版本".into());
    }
    let relative = format!("drafts/{activity}/versions/{revision}.json");
    let path = safe_path(root, &relative)?;
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() > 400_000 {
        return Err("资料超过读取预算".into());
    }
    let mut bytes = Vec::new();
    std::io::Read::by_ref(&mut file)
        .take(400_001)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 400_000 {
        return Err("资料超过读取预算".into());
    }
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("资料格式错误：{e}"))?;
    let code = text(&value, "code")?;
    if code.len() > 300_000 {
        return Err("代码超过本次编辑预算".into());
    }
    let hash = digest(code.as_bytes());
    if text(&value, "sha256")? != hash || !hash.eq_ignore_ascii_case(revision) {
        return Err("代码历史版本内容与哈希不符".into());
    }
    let updated_at = value["updated_at"]
        .as_u64()
        .ok_or("代码历史版本缺少保存时间")?;
    Ok(json!({"code":code,"sha256":hash,"updated_at":updated_at}))
}

/// Reject ambiguous Windows forms and every reparse component before opening.
/// Frozen bytes are hashed AFTER opening; those exact bytes, not the path, are used.
pub fn safe_path(root: &Path, relative: &str) -> Result<PathBuf, String> {
    if relative.is_empty()
        || relative.contains('\\')
        || relative.contains(':')
        || relative.contains('%')
        || relative.starts_with('/')
        || relative.contains('\0')
    {
        return Err("拒绝不明确或越界的资源路径".into());
    }
    for part in relative.split('/') {
        let stem = part.split('.').next().unwrap_or("").to_ascii_uppercase();
        if part.ends_with('.')
            || part.ends_with(' ')
            || [
                "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
                "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8",
                "LPT9",
            ]
            .contains(&stem.as_str())
        {
            return Err("拒绝 Windows 保留设备名或不明确的文件名".into());
        }
    }
    let rel = Path::new(relative);
    if rel.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err("拒绝越界路径".into());
    }
    let base = root.canonicalize().map_err(|e| e.to_string())?;
    let mut p = base.clone();
    for part in rel.components() {
        p.push(part.as_os_str());
        let m = fs::symlink_metadata(&p).map_err(|e| format!("资源缺失：{e}"))?;
        #[cfg(windows)]
        if m.file_attributes() & 0x400 != 0 {
            return Err("资源路径包含重解析点，已停止读取".into());
        }
        if m.file_type().is_symlink() {
            return Err("资源路径包含符号链接".into());
        }
    }
    let final_path = p.canonicalize().map_err(|e| e.to_string())?;
    if !final_path.starts_with(&base) {
        return Err("资源超出登记根目录".into());
    }
    Ok(final_path)
}
fn verified(root: &Path, relative: &str, expected: &str, limit: u64) -> Result<Vec<u8>, String> {
    let p = safe_path(root, relative)?;
    let mut f = File::open(p).map_err(|e| e.to_string())?;
    if f.metadata().map_err(|e| e.to_string())?.len() > limit {
        return Err("文件超过本次读取预算".into());
    }
    let mut bytes = Vec::new();
    std::io::Read::by_ref(&mut f)
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit || digest(&bytes) != expected {
        return Err("资源已改变，须重新核对版本；保留普通阅读".into());
    }
    Ok(bytes)
}
fn context(app: &AppHandle, book_id: &str) -> Result<Context, String> {
    token(book_id)?;
    let root = state_root(app)?;
    let binding_file=binding_path(app)?;
    if !binding_file.is_file(){return Err("这本书没有登记学习增强包".into());}
    let config = read_json(&binding_file, 4 * 1024 * 1024)?;
    let binding = config["books"][book_id].clone();
    if binding.is_null() {
        return Err("这本书没有登记学习增强包".into());
    }
    let package_path = Path::new(text(&binding, "package")?);
    let bytes = fs::read(package_path).map_err(|e| format!("增强包暂不可用：{e}"))?;
    if bytes.len() > 16 * 1024 * 1024 || digest(&bytes) != text(&binding, "package_sha256")? {
        return Err("增强包版本已改变，等待重新匹配".into());
    }
    let pack: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    validate_pack(&pack)?;
    if pack["schema_version"] != 1
        || pack["book_uuid"] != binding["book_uuid"]
        || pack["book_revision_sha256"] != binding["epub_sha256"]
    {
        return Err("增强包与书籍身份不匹配".into());
    }
    let profile_root=package_path.parent().ok_or("增强包缺少所在目录")?.to_string_lossy().to_string();
    let mut binding = binding;
    binding["environments"] = config["environments"].clone();
    binding["profile_root"]=json!(profile_root);
    let approved=binding["recipes"].as_object().cloned().unwrap_or_default();
    let mut resolved=serde_json::Map::new();let mut errors=serde_json::Map::new();
    for (id,grant) in approved{
        if grant.get("definition_sha256").is_some(){match declared_recipe(&pack,&binding,&id,&grant){Ok(recipe)=>{resolved.insert(id,recipe);},Err(error)=>{errors.insert(id,json!(error));}}}
        else{resolved.insert(id,grant);}
    }
    binding["recipes"]=Value::Object(resolved);binding["recipe_errors"]=Value::Object(errors);
    Ok(Context {
        binding,
        pack,
        state: root.join(book_id),
    })
}

fn declared_recipe(pack:&Value,binding:&Value,id:&str,grant:&Value)->Result<Value,String>{
    if grant["reviewed_pure_local"]!=true{return Err("此版本尚未经本机执行审查".into());}
    let activity=pack["activities"].as_array().and_then(|rows|rows.iter().find(|a|a["id"]==id)).ok_or("活动未登记")?;
    let raw=text(activity,"run_definition_json")?;
    if raw.len()>64*1024||digest(raw.as_bytes())!=text(grant,"definition_sha256")?{return Err("活动定义已更新，原执行批准未自动转给新版".into());}
    let definition:Value=serde_json::from_str(raw).map_err(|e|e.to_string())?;
    if definition["schemaVersion"]!=1||definition["kind"]!="python-cli@1"{return Err("运行定义版本未支持".into());}
    let assets=pack["assets"].as_array().ok_or("资源表无效")?;
    let find=|id:&str|->Result<&Value,&str>{let asset=assets.iter().find(|a|a["id"]==id).ok_or("运行资源未登记")?;if grant["assets_sha256"][id]!=asset["sha256"]{return Err("代码或输入资源换版，旧执行批准没有转给新字节");}Ok(asset)};
    let entry=find(text(&definition,"entry_asset")?)?;
    if definition["entry_asset"]!=activity["entry_asset"]{return Err("界面源码与运行入口不一致".into());}
    if entry["kind"]!="code"||entry["root"].as_str().is_some_and(|r|r!="publication"){return Err("代码入口需位于书籍内容根".into());}
    let ids=definition["input_assets"].as_array().ok_or("运行输入表无效")?;if ids.len()>256{return Err("运行输入过多".into());}
    let mut inputs=Vec::new();let mut paths=std::collections::HashSet::new();paths.insert(text(entry,"relative_path")?.to_string());
    for id in ids{let a=find(id.as_str().ok_or("输入身份无效")?)?;if !paths.insert(text(a,"relative_path")?.to_string()){return Err("运行输入与源码路径冲突".into());}inputs.push(json!({"path":a["relative_path"],"sha256":a["sha256"],"root":a["root"]}));}
    let adapter=if let Some(id)=definition["adapter_asset"].as_str(){let a=find(id)?;if a["kind"]!="code"{return Err("执行适配必须为已核对的源码".into());}let root=resource_root(binding,a["root"].as_str())?;let path=safe_path(&root,text(a,"relative_path")?)?;json!({"path":path.to_string_lossy(),"sha256":a["sha256"],"reviewed":true})}else{Value::Null};
    Ok(json!({"entry":entry["relative_path"],"sha256":entry["sha256"],"inputs":inputs,"adapter":adapter,"generic_native":true,"reviewed_pure_local":true,"runtime":grant["runtime"],"editable":definition["editable"],"argv":definition["argv"],"parameters":definition["parameters"],"timeout_seconds":definition["timeout_seconds"],"requires_budget_approval":definition["requires_budget_approval"],"resume_files":definition["resume_files"],"monitor":definition["monitor"],"definition_sha256":grant["definition_sha256"]}))
}
fn resource_root(binding:&Value,kind:Option<&str>)->Result<PathBuf,String>{
    Ok(PathBuf::from(text(binding,match kind{Some("derived")=>"derived_root",Some("profile")=>"profile_root",_=>"content_root"})?))
}

fn validate_pack(pack: &Value) -> Result<(), String> {
    use std::collections::HashSet;
    let assets = pack["assets"].as_array().ok_or("增强包缺少资源表")?;
    let chapters = pack["chapters"].as_array().ok_or("增强包缺少正文表")?;
    if assets.len() > 20000 || chapters.len() > 4000 {
        return Err("增强包超过当前容量预算".into());
    }
    let mut ids = HashSet::new();
    for a in assets {
        let id = text(a, "id")?;
        token(id)?;
        if !ids.insert(id.to_owned()) {
            return Err("增强包资源身份重复".into());
        }
        if ![
            "code", "text", "image", "audio", "video", "pdf", "geometry", "data", "model",
        ]
        .contains(&text(a, "kind")?)
        {
            return Err("增强包含未知资源组件".into());
        }
        let path = text(a, "relative_path")?;
        if path.is_empty()
            || path.contains(['\\', ':', '%', '\0'])
            || path.split('/').any(|p| {
                p.is_empty() || p == "." || p == ".." || p.ends_with('.') || p.ends_with(' ')
            })
        {
            return Err("增强包含不明确或越界路径".into());
        }
        let hash = text(a, "sha256")?;
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("资源缺少有效字节身份".into());
        }
        if a.get("root").is_some() && !["publication", "derived", "profile"].contains(&text(a, "root")?) {
            return Err("资源根未登记".into());
        }
    }
    let mut chapter_ids = HashSet::new();
    for c in chapters {
        let id = text(c, "id")?;
        token(id)?;
        if !chapter_ids.insert(id.to_owned()) {
            return Err("正文章节身份重复".into());
        }
        let href = text(c, "href")?;
        if href.contains(':') || href.starts_with('/') || href.contains('\\') {
            return Err("正文地址必须属于当前 EPUB".into());
        }
    }
    if let Some(map) = pack["href_assets"].as_object() {
        for id in map.values() {
            if !id.as_str().is_some_and(|s| ids.contains(s)) {
                return Err("存在无身份的资源入口".into());
            }
        }
    }
    if let Some(activities) = pack["activities"].as_array() {
        for a in activities {
            token(text(a, "id")?)?;
            if !ids.contains(text(a, "entry_asset")?) || !chapter_ids.contains(text(a, "chapter")?)
            {
                return Err("活动未绑定正文与源码".into());
            }
            if [
                "shell",
                "command",
                "cwd",
                "python_path",
                "env",
                "trusted",
                "script",
            ]
            .iter()
            .any(|key| a.get(*key).is_some())
            {
                return Err("书籍数据不能声明执行权限".into());
            }
        }
    }
    if let Some(sources) = pack["source_claims"].as_array() {
        for s in sources {
            let url = tauri::Url::parse(text(s, "url")?).map_err(|_| "来源链接无效")?;
            if !["http", "https"].contains(&url.scheme())
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
            {
                return Err("来源不能使用本机协议或携带凭据".into());
            }
        }
    }
    Ok(())
}
fn asset<'a>(ctx: &'a Context, id: &str) -> Result<&'a Value, String> {
    ctx.pack["assets"]
        .as_array()
        .and_then(|all| all.iter().find(|a| a["id"] == id))
        .ok_or_else(|| "资料未登记".into())
}
fn asset_bytes(ctx: &Context, id: &str) -> Result<Vec<u8>, String> {
    let a = asset(ctx, id)?;
    verified(
        &resource_root(&ctx.binding,a["root"].as_str())?,
        text(a, "relative_path")?,
        text(a, "sha256")?,
        MAX_ASSET,
    )
}
fn draft_path(ctx: &Context, activity: &str) -> Result<PathBuf, String> {
    token(activity)?;
    Ok(ctx.state.join("drafts").join(format!("{activity}.json")))
}

fn ensure_book_revision(app: &AppHandle, book_id: &str, binding: &Value) -> Result<(), String> {
    let reader = super::load_state(app)?;
    let book = reader.books.iter().find(|b| b.id == book_id).ok_or("书籍不在当前书库")?;
    if book.path.starts_with("https://")||book.path.starts_with("http://"){
        if book.catalog_source.as_ref().is_some_and(|s|s["sha256"]==binding["catalog_manifest_sha256"]&&!binding["catalog_manifest_sha256"].is_null())&&book.book_uuid.as_deref()==binding["book_uuid"].as_str(){return Ok(());}
        return Err("当前书目修订与本机运行材料尚未匹配；没有启动代码".into());
    }
    let bytes = fs::read(&book.path).map_err(|e| e.to_string())?;
    if digest(&bytes) != text(binding, "epub_sha256")? {
        return Err("EPUB 已更新，增强包等待匹配；正文仍可阅读".into());
    }
    Ok(())
}

#[tauri::command]
pub async fn learning_pack(app: AppHandle, book_id: String) -> Result<Value, String> {
    let ctx = context(&app, &book_id)?;
    ensure_book_revision(&app,&book_id,&ctx.binding)?;
    let mut pack = ctx.pack;
    if let Some(activities) = pack["activities"].as_array_mut() {
        for activity in activities {
            let id = activity["id"].as_str().unwrap_or("").to_owned();
            let recipe = &ctx.binding["recipes"][&id];
            activity["registered"] = json!(!recipe.is_null());
            if !recipe.is_null(){activity["execution_revision"]=json!(execution_revision(&ctx.binding,recipe)?);}
            else{activity["registration_error"]=ctx.binding["recipe_errors"][&id].clone();}
        }
    }
    Ok(pack)
}
fn execution_revision(binding:&Value,recipe:&Value)->Result<String,String>{
    let environment=&binding["environments"][recipe["runtime"].as_str().unwrap_or("")];
    Ok(digest(&serde_json::to_vec(&json!({"recipe":recipe,"environment":environment,"package":binding["package_sha256"]})).map_err(|e|e.to_string())?))
}
#[tauri::command]
pub async fn learning_asset(
    app: AppHandle,
    book_id: String,
    asset_id: String,
) -> Result<Response, String> {
    let ctx = context(&app, &book_id)?;
    Ok(Response::new(asset_bytes(&ctx, &asset_id)?))
}
#[tauri::command]
pub fn learning_resource_info(
    app: AppHandle,
    book_id: String,
    asset_id: String,
) -> Result<Value, String> {
    let ctx = context(&app, &book_id)?;
    let asset = asset(&ctx, &asset_id)?;
    let root = resource_root(&ctx.binding,asset["root"].as_str())?;
    let path = safe_path(&root, text(asset, "relative_path")?)?;
    let _ = verified(
        &root,
        text(asset, "relative_path")?,
        text(asset, "sha256")?,
        MAX_ASSET,
    )?;
    let full = path.to_string_lossy();
    let display = if let Some(unc) = full.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{unc}")
    } else {
        full.trim_start_matches(r"\\?\").to_owned()
    };
    Ok(json!({"path":display,"sha256":asset["sha256"],"bytes":asset["bytes"]}))
}
#[tauri::command]
pub fn learning_load_state(app: AppHandle, book_id: String) -> Result<Value, String> {
    let state = user_storage(&app, &book_id)?;
    let p = state.join("study-state.json");
    if p.exists() {
        read_json(&p, 4 * 1024 * 1024)
    } else {
        Ok(json!({"version":1,"notes":[],"media":{},"drafts":{}}))
    }
}
#[tauri::command]
pub fn learning_save_state(app: AppHandle, book_id: String, value: Value) -> Result<(), String> {
    let state = user_storage(&app, &book_id)?;
    if serde_json::to_vec(&value).map_err(|e| e.to_string())?.len() > 4 * 1024 * 1024 {
        return Err("笔记超过单次保存预算".into());
    }
    atomic(&state.join("study-state.json"), &value)
}
#[tauri::command]
pub fn learning_raw_state(app: AppHandle, book_id: String) -> Result<Value, String> {
    let path = user_storage(&app, &book_id)?.join("study-state.json");
    let metadata = fs::metadata(&path).map_err(|e| format!("原始记录暂时无法读取：{e}"))?;
    if metadata.len() > 16 * 1024 * 1024 { return Err("原始记录超过备份读取范围；原文件仍保留，请先另存本机原件".into()); }
    let bytes = fs::read(&path).map_err(|e| e.to_string())?;
    Ok(json!({"name":format!("原始学习记录-{book_id}.json"),"sha256":digest(&bytes),"bytes":bytes}))
}
#[tauri::command]
pub fn learning_save_draft(
    app: AppHandle,
    book_id: String,
    activity_id: String,
    code: String,
) -> Result<Value, String> {
    let state = user_storage(&app, &book_id)?;
    token(&activity_id)?;
    if code.len() > 300_000 {
        return Err("代码超过本次编辑预算".into());
    }
    let value =
        json!({"code":code,"sha256":digest(code.as_bytes()),"updated_at":super::now_secs()});
    let version = state
        .join("drafts")
        .join(&activity_id)
        .join("versions")
        .join(format!("{}.json", digest(code.as_bytes())));
    if !version.exists() {
        atomic(&version, &value)?;
    }
    atomic(&saved_draft_path(&state, &activity_id)?, &value)?;
    Ok(value)
}
#[tauri::command]
pub fn learning_draft(
    app: AppHandle,
    book_id: String,
    activity_id: String,
) -> Result<Value, String> {
    let state = user_storage(&app, &book_id)?;
    let path = saved_draft_path(&state, &activity_id)?;
    if path.exists() {
        let mut draft = read_json(&path, 400_000)?;
        let grant = read_json(
            &state.join("grants").join(format!("{activity_id}.json")),
            16_000,
        )
        .unwrap_or(Value::Null);
        draft["authorized"] = json!(grant["code_sha256"] == draft["sha256"]);
        Ok(draft)
    } else {
        Ok(Value::Null)
    }
}
#[tauri::command]
pub fn learning_import_draft(app:AppHandle,book_id:String,activity_id:String,code:String)->Result<(),String>{
    token(&activity_id)?;if code.len()>300_000{return Err("代码超过导入范围".into());}
    let state=user_storage(&app,&book_id)?;
    let hash=digest(code.as_bytes());let value=json!({"code":code,"sha256":hash,"updated_at":super::now_secs(),"provenance":"explicit_personal_import"});
    let version=state.join("drafts").join(&activity_id).join("versions").join(format!("{hash}.json"));
    if version.exists(){read_draft_revision(&state,&activity_id,&hash)?;}else{atomic(&version,&value)?;}
    let current=saved_draft_path(&state,&activity_id)?;if !current.exists(){atomic(&current,&value)?;}
    Ok(())
}
#[tauri::command]
pub fn learning_authorize_edit(
    app: AppHandle,
    book_id: String,
    activity_id: String,
    code_sha256: String,
) -> Result<(), String> {
    let ctx = context(&app, &book_id)?;
    let draft = read_json(&draft_path(&ctx, &activity_id)?, 400_000)?;
    if draft["sha256"] != code_sha256 {
        return Err("代码已改变，请先保存当前版本".into());
    }
    atomic(
        &ctx.state.join("grants").join(format!("{activity_id}.json")),
        &json!({"code_sha256":code_sha256,"mode":"explicit_high_trust_native","created_at":super::now_secs()}),
    )
}

fn registered_arguments(
    _activity: &str,
    params: &Value,
    recipe: &Value,
) -> Result<Vec<String>, String> {
    let adapter=recipe.get("adapter").filter(|a|!a.is_null());
    if adapter.is_some() {
        if adapter.unwrap()["reviewed"]!=true {return Err("随书运行适配尚未经过本机审查".into());}
        if recipe["requires_budget_approval"]!=true&&recipe["timeout_seconds"].as_u64().unwrap_or(301)>300 {return Err("超过已登记的短任务预算".into());}
    } else if recipe["generic_native"]!=true||recipe["reviewed_pure_local"]!=true||recipe["timeout_seconds"].as_u64().unwrap_or(121)>120 {
        return Err("这项运行配置需按当前版本核对；通用短实验最长120秒，书籍声明不能自行授权".into());
    }
    let mut argv = Vec::new();
    if let Some(fixed) = recipe["argv"].as_array() {
        if fixed.len() > 32 {
            return Err("固定参数过多".into());
        }
        for item in fixed {
            let value = item.as_str().ok_or("固定参数必须为文本")?;
            if value.len() > 2048 || value.contains('\0') {
                return Err("固定参数无效".into());
            }
            argv.push(value.to_owned());
        }
    }
    let fields = recipe["parameters"].as_object();
    for key in params.as_object().ok_or("参数须为对象")?.keys() {
        if !fields.is_some_and(|f| f.contains_key(key)) {
            return Err(format!("参数 {key} 未在宿主登记"));
        }
    }
    if let Some(fields) = fields {
        if fields.len() > 24 {
            return Err("参数规格过多".into());
        }
        for (name, spec) in fields {
            let value = params.get(name).unwrap_or(&spec["default"]);
            let forward=spec["forward"]!=false;
            let flag = if forward{text(spec, "flag")?}else{"--adapter-parameter"};
            if !flag.starts_with("--")
                || flag.len() < 3
                || !flag[2..]
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            {
                return Err("参数标记格式无效".into());
            }
            let rendered = match spec["type"].as_str() {
                Some("number") | Some("integer") => {
                    let number = value.as_f64().ok_or_else(|| format!("{name} 需要数值"))?;
                    if !number.is_finite()
                        || number < spec["min"].as_f64().unwrap_or(-1e6)
                        || number > spec["max"].as_f64().unwrap_or(1e6)
                        || (spec["type"] == "integer" && number.fract() != 0.0)
                    {
                        return Err(format!("{name} 超出允许范围"));
                    }
                    number.to_string()
                }
                Some("enum") => {
                    if !spec["choices"]
                        .as_array()
                        .is_some_and(|a| a.contains(value))
                    {
                        return Err(format!("{name} 不是登记选项"));
                    }
                    value.as_str().ok_or("选项须为文本")?.to_owned()
                }
                Some("string") => {
                    let v = value.as_str().ok_or("参数须为文本")?;
                    if v.len() > spec["max_bytes"].as_u64().unwrap_or(400).min(2048) as usize
                        || v.contains('\0')
                    {
                        return Err(format!("{name} 文本过长或无效"));
                    }
                    v.to_owned()
                }
                Some("boolean") => {
                    if value.as_bool().ok_or("参数须为布尔值")? && forward {
                        argv.push(flag.to_owned());
                    }
                    continue;
                }
                _ => return Err("宿主参数类型不受支持".into()),
            };
            if forward{argv.push(format!("{flag}={rendered}"));}
        }
    }
    Ok(argv)
}

fn training_budget(params: &Value) -> Result<u64, String> {
    let budget = params["budget_seconds"]
        .as_u64()
        .ok_or("需要明确训练时长预算")?;
    if !(15..=900).contains(&budget)
        || !["cpu", "cuda"].contains(&params["device"].as_str().unwrap_or(""))
    {
        return Err("训练预算须为15—900秒，设备为CPU或CUDA".into());
    }
    if let Some(previous) = params["resume_from"].as_str() {
        if !previous.is_empty() {
            token(previous)?;
        }
    }
    Ok(budget)
}
#[tauri::command]
pub fn learning_authorize_training(
    app: AppHandle,
    book_id: String,
    request_id: String,
    params: Value,
    activity_id:String,
) -> Result<(), String> {
    token(&request_id)?;token(&activity_id)?;
    training_budget(&params)?;
    let ctx = context(&app, &book_id)?;
    if ctx.binding["recipes"][&activity_id]["requires_budget_approval"] != true {
        return Err("此书没有登记完整训练配方".into());
    }
    atomic(
        &ctx.state
            .join("training-grants")
            .join(format!("{request_id}.json")),
        &json!({"params":params,"activity_id":activity_id,"created_at":super::now_secs(),"recipe_sha256":ctx.binding["recipes"][&activity_id]["sha256"]}),
    )
}

fn draft_activity_ids(state: &Path) -> Result<Vec<String>, String> {
    let folder = state.join("drafts");
    match fs::symlink_metadata(&folder) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.to_string()),
        Ok(_) => {}
    }
    let folder = safe_path(state, "drafts")?;
    let mut ids = std::collections::BTreeSet::new();
    for item in fs::read_dir(folder).map_err(|e| e.to_string())? {
        let item = item.map_err(|e| e.to_string())?;
        let name = item.file_name();
        let Some(name) = name.to_str() else { continue; };
        let kind = item.file_type().map_err(|e| e.to_string())?;
        let id = if kind.is_dir() { name } else if let Some(id) = name.strip_suffix(".json") { id } else { continue; };
        token(id)?;
        safe_path(state, &format!("drafts/{name}"))?;
        ids.insert(id.to_owned());
        if ids.len() > 2000 { return Err("代码活动数量超过单次导出范围".into()); }
    }
    Ok(ids.into_iter().collect())
}
#[tauri::command]
pub fn learning_draft_activities(app: AppHandle, book_id: String) -> Result<Vec<String>, String> {
    draft_activity_ids(&user_storage(&app, &book_id)?)
}

#[tauri::command]
pub fn learning_draft_versions(
    app: AppHandle,
    book_id: String,
    activity_id: String,
) -> Result<Value, String> {
    token(&activity_id)?;
    let state = user_storage(&app, &book_id)?;
    let folder = state.join("drafts").join(&activity_id).join("versions");
    let mut versions = Vec::new();
    match fs::symlink_metadata(&folder) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
        Ok(_) => {
            let safe_folder = safe_path(&state, &format!("drafts/{activity_id}/versions"))?;
            for item in fs::read_dir(safe_folder).map_err(|e| e.to_string())? {
                let item = item.map_err(|e| e.to_string())?;
                let name = item.file_name();
                let Some(revision) = name.to_str().and_then(|s| s.strip_suffix(".json")) else {
                    continue;
                };
                let value = read_draft_revision(&state, &activity_id, revision)?;
                versions.push(json!({"sha256":value["sha256"],"updated_at":value["updated_at"],"bytes":text(&value,"code")?.len()}));
            }
        }
    }
    versions.sort_by_key(|v| std::cmp::Reverse(v["updated_at"].as_u64().unwrap_or(0)));
    Ok(json!(versions))
}
#[tauri::command]
pub fn learning_draft_revision(
    app: AppHandle,
    book_id: String,
    activity_id: String,
    revision: String,
) -> Result<Value, String> {
    let state = user_storage(&app, &book_id)?;
    read_draft_revision(&state, &activity_id, &revision)
}
#[tauri::command]
pub fn learning_restore_draft(
    app: AppHandle,
    book_id: String,
    activity_id: String,
    revision: String,
) -> Result<Value, String> {
    let state = user_storage(&app, &book_id)?;
    let value = read_draft_revision(&state, &activity_id, &revision)?;
    learning_save_draft(app, book_id, activity_id, text(&value, "code")?.to_owned())
}
#[tauri::command]
pub async fn learning_export(
    app: AppHandle,
    book_id: String,
    activity_id: Option<String>,
) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let state = user_storage(&app, &book_id)?;
    let (name, body) = if let Some(id) = activity_id {
        let draft = read_json(&saved_draft_path(&state, &id)?, 400_000)?;
        (
            format!("我的练习-{id}.py"),
            text(&draft, "code")?.to_owned(),
        )
    } else {
        let state = read_json(&state.join("study-state.json"), 4 * 1024 * 1024)?;
        let mut body = String::from("# 我的随书思考\n\n");
        for note in state["notes"].as_array().unwrap_or(&Vec::new()) {
            body.push_str(&format!(
                "## {}\n\n> {}\n\n{}\n\n原位：{}\n\n",
                note["chapter"].as_str().unwrap_or(""),
                note["quote"].as_str().unwrap_or("").replace('\n', "\n> "),
                note["text"].as_str().unwrap_or(""),
                note["cfi"]
                    .as_str()
                    .or_else(|| note["href"].as_str())
                    .unwrap_or("")
            ));
        }
        ("我的随书思考.md".to_owned(), body)
    };
    // The path is obtained by the native save dialog, never from book content.
    let chosen = app
        .dialog()
        .file()
        .set_file_name(&name)
        .blocking_save_file();
    if let Some(file) = chosen {
        let path = file.into_path().map_err(|e| e.to_string())?;
        fs::write(&path, body.as_bytes()).map_err(|e| format!("导出没有完成：{e}"))?;
        Ok(Some(path.to_string_lossy().to_string()))
    } else {
        Ok(None)
    }
}
fn copy_frozen(ctx: &Context, folder: &Path, relative: &str, expected: &str,source_root:Option<&str>) -> Result<(), String> {
    validate_run_path(relative)?;
    let bytes = verified(
        &resource_root(&ctx.binding,source_root)?,
        relative,
        expected,
        MAX_ASSET,
    )?;
    let out = folder.join(relative);
    fs::create_dir_all(out.parent().ok_or("无效目标")?).map_err(|e| e.to_string())?;
    fs::write(out, bytes).map_err(|e| e.to_string())
}
fn validate_run_path(relative:&str)->Result<(),String>{
    if ["worker.py","book-adapter.py","request.json","run.json","reference-source.py","environment.json","stdout.log","stderr.log"].contains(&relative.to_ascii_lowercase().as_str()){return Err("书籍输入不能覆盖任务控制文件".into());}Ok(())
}

#[cfg(windows)]
fn attach_job(child: &Child, memory_limit: usize) -> Result<isize, String> {
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return Err("无法建立任务生命周期管理；未执行代码".into());
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_JOB_MEMORY;
        info.BasicLimitInformation.ActiveProcessLimit = 16;
        info.JobMemoryLimit = memory_limit;
        if SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of_val(&info) as u32,
        ) == 0
            || AssignProcessToJobObject(job, child.as_raw_handle() as _) == 0
        {
            CloseHandle(job);
            return Err("无法关联进程树；已取消启动".into());
        }
        Ok(job as isize)
    }
}
#[cfg(not(windows))]
fn attach_job(_child: &Child, _memory_limit: usize) -> Result<isize, String> {
    Err("此发行版的原生运行需要 Windows Job Objects".into())
}
fn job_memory(job: isize) -> Value {
    #[cfg(windows)]
    unsafe {
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        if QueryInformationJobObject(job as _, JobObjectExtendedLimitInformation,
            &mut info as *mut _ as _, std::mem::size_of_val(&info) as u32,
            std::ptr::null_mut()) != 0 {
            return json!({"limit_bytes":info.JobMemoryLimit,"peak_committed_bytes":info.PeakJobMemoryUsed,
                "meaning":"Windows process-group committed memory; not dedicated GPU memory"});
        }
    }
    Value::Null
}
fn kill_run(run: &mut Running) {
    run.cancelled = true;
    #[cfg(windows)]
    unsafe {
        TerminateJobObject(run.job as _, 1);
    }
    let _ = run.child.kill();
}
fn pump(mut stream: impl Read + Send + 'static, path: PathBuf) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let Ok(mut f) = File::create(path) else {
            return;
        };
        let mut buf = [0u8; 4096];
        let mut used = 0;
        loop {
            let Ok(n) = stream.read(&mut buf) else { break };
            if n == 0 {
                break;
            }
            if used < MAX_LOG {
                let keep = n.min(MAX_LOG - used);
                let _ = f.write_all(&buf[..keep]);
                used += keep;
            }
        }
    })
}
fn abort_spawn(child: &mut Child, job: isize) {
    #[cfg(windows)]
    unsafe {
        TerminateJobObject(job as _, 1);
        CloseHandle(job as _);
    }
    let _ = child.kill();
    let _ = child.wait();
}
fn output_files(folder: &Path, inputs: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    for sub in ["work/results", "work/runs"] {
        for e in walkdir::WalkDir::new(folder.join(sub))
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !e.file_type().is_file() {
                continue;
            }
            let Ok(meta) = e.metadata() else { continue };
            if meta.len() > MAX_ASSET {
                continue;
            }
            let relative = e
                .path()
                .strip_prefix(folder)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let ext = e
                .path()
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ![
                "json", "jsonl", "csv", "txt", "png", "jpg", "wav", "mp4", "pt",
            ]
            .contains(&ext.as_str())
            {
                continue;
            }
            if safe_path(folder, &relative).is_err() {
                continue;
            }
            if let Ok(bytes) = fs::read(e.path()) {
                let hash = digest(&bytes);
                if inputs.as_array().is_some_and(|all| {
                    all.iter()
                        .any(|input| input["path"] == relative && input["sha256"] == hash)
                }) {
                    continue;
                }
                out.push(json!({"path":relative,"sha256":digest(&bytes),"bytes":bytes.len(),"extension":ext,"role":"learner_run"}));
            }
            if out.len() >= 120 {
                break;
            }
        }
    }
    out
}

#[tauri::command]
pub fn learning_run(
    app: AppHandle,
    manager: State<'_, LearningManager>,
    book_id: String,
    activity_id: String,
    request_id: String,
    params: Value,
    use_draft: bool,
    expected_recipe_revision:String,
    expected_code_sha256:String,
) -> Result<Value, String> {
    token(&request_id)?;
    token(&activity_id)?;
    let ctx = context(&app, &book_id)?;
    // Recheck at the execution boundary even if the reading view cached its pack.
    ensure_book_revision(&app, &book_id, &ctx.binding)?;
    let recipe = &ctx.binding["recipes"][&activity_id];
    if recipe.is_null() {
        return Err("此活动尚未绑定受信运行环境".into());
    }
    if execution_revision(&ctx.binding,recipe)?!=expected_recipe_revision{return Err("运行定义或环境在界面打开后改变；请重新打开活动核对，未启动新代码".into());}
    let is_training=recipe["requires_budget_approval"]==true;
    if is_training {
        training_budget(&params)?;
        if use_draft {
            return Err("完整训练只使用已核对的原配方".into());
        }
        let grant = read_json(
            &ctx.state
                .join("training-grants")
                .join(format!("{request_id}.json")),
            16000,
        )
        .map_err(|_| "完整训练尚未批准本次设备与时间预算")?;
        if grant["params"] != params || grant["recipe_sha256"] != recipe["sha256"] || grant["activity_id"] != activity_id {
            return Err("本次训练条件与批准内容不一致".into());
        }
    }
    if use_draft && recipe["editable"] != true {
        return Err("当前配方不接受自由编辑代码执行".into());
    }
    let key = format!("{book_id}-{request_id}");
    let mut map = manager.inner.lock().map_err(|_| "运行登记暂不可用")?;
    if let Some(existing) = map.get(&key) {
        return Ok(existing
            .lock()
            .map_err(|_| "运行状态不可用")?
            .report
            .clone());
    }
    if map.values().any(|r| {
        r.lock()
            .map(|r| r.report["status"] == "running")
            .unwrap_or(true)
    }) {
        return Err("已有一个本机实验在运行，请等待或停止它".into());
    }
    let folder = ctx.state.join("runs").join(&request_id);
    if folder.exists() {
        return read_json(&folder.join("run.json"), 1024 * 1024);
    }
    let argv = registered_arguments(&activity_id, &params, recipe)?;
    let env = &ctx.binding["environments"][text(recipe, "runtime")?];
    let python = Path::new(text(env, "python")?);
    let python_bytes = fs::read(python).map_err(|e| format!("Python 环境不可用：{e}"))?;
    if digest(&python_bytes) != text(env, "sha256")? {
        return Err("Python 解释器版本已改变，请重新核对环境".into());
    }
    let entry = text(recipe, "entry")?;
    validate_run_path(entry)?;
    let root = Path::new(text(&ctx.binding, "content_root")?);
    let original = verified(root, entry, text(recipe, "sha256")?, MAX_ASSET)?;
    let reference = original.clone();
    let code = if use_draft {
        let draft = read_json(&draft_path(&ctx, &activity_id)?, 400_000)?;
        let code = text(&draft, "code")?.as_bytes().to_vec();
        if !same_python_text(&code, &reference) {
            let grant = read_json(
                &ctx.state.join("grants").join(format!("{activity_id}.json")),
                16_000,
            )
            .map_err(|_| "此版本需要明确授权高信任本机执行；它拥有当前用户的文件和网络权限")?;
            if grant["code_sha256"] != digest(&code) {
                return Err("本次代码版本尚未获得高信任运行授权".into());
            }
        }
        code
    } else {
        original
    };
    if digest(&code)!=expected_code_sha256{return Err("源码或草稿已改变；本次未执行与界面不同的版本，请核对后重试".into());}
    fs::create_dir_all(folder.join("work/code")).map_err(|e| e.to_string())?;
    fs::create_dir_all(folder.join("work/results")).map_err(|e| e.to_string())?;
    fs::create_dir_all(folder.join("work/runs")).map_err(|e| e.to_string())?;
    let folder = folder.canonicalize().map_err(|e| e.to_string())?;
    let preparing = json!({"run_id":request_id,"book_id":book_id,"activity_id":activity_id,"status":"preparing","code_sha256":digest(&code),"params":params,"inputs":recipe["inputs"],"artifacts":[]});
    atomic(&folder.join("run.json"), &preparing)?;
    let mut preparation = PreparationReceipt {
        path: folder.join("run.json"),
        record: preparing,
        armed: true,
    };
    fs::create_dir_all(folder.join(entry).parent().ok_or("源码路径无效")?)
        .map_err(|e| e.to_string())?;
    fs::write(folder.join(entry), &code).map_err(|e| e.to_string())?;
    fs::write(folder.join("reference-source.py"), &reference).map_err(|e| e.to_string())?;
    for input in recipe["inputs"].as_array().ok_or("配方输入表错误")? {
        copy_frozen(&ctx, &folder, text(input, "path")?, text(input, "sha256")?,input["root"].as_str())?;
    }
    if is_training {
        if let Some(previous) = params["resume_from"].as_str().filter(|s| !s.is_empty()) {
            let old = ctx.state.join("runs").join(previous);
            let receipt = read_json(&old.join("run.json"), 1024 * 1024)?;
            if receipt["activity_id"] != activity_id
                || receipt["code_sha256"] != recipe["sha256"]
                || receipt["status"] == "running"
                || receipt["device"] != params["device"]
            {
                return Err("只能从本书已停止且代码匹配的训练恢复".into());
            }
            let files = receipt["artifacts"]
                .as_array()
                .ok_or("上次没有可恢复产物")?;
            let resume_files=recipe["resume_files"].as_array().ok_or("尚未登记可恢复产物")?;
            let latest=resume_files.first().and_then(|v|v.as_str()).ok_or("检查点入口未登记")?;
            if !files.iter().any(|a| a["path"] == latest) {
                return Err("上次还没有保存检查点；可开始一次新训练".into());
            }
            for file in resume_files {
                let relative=file.as_str().ok_or("检查点地址无效")?;
                if let Some(a) = files.iter().find(|a| a["path"] == relative) {
                    let bytes = verified(&old, relative, text(a, "sha256")?, MAX_ASSET)?;
                    fs::write(folder.join(relative), bytes).map_err(|e| e.to_string())?;
                }
            }
        }
    }
    let adapter_hash=if let Some(adapter)=recipe.get("adapter").filter(|v|!v.is_null()){
        let path=Path::new(text(adapter,"path")?);let data=fs::read(path).map_err(|e|format!("随书运行适配不可用：{e}"))?;
        if data.len()>300_000||digest(&data)!=text(adapter,"sha256")?{return Err("随书适配版本改变，尚未重新批准".into());}
        fs::write(folder.join("book-adapter.py"),&data).map_err(|e|e.to_string())?;Some(digest(&data))
    }else{None};
    fs::write(folder.join("worker.py"), WORKER).map_err(|e| e.to_string())?;
    atomic(
        &folder.join("request.json"),
        &json!({"activity_id":activity_id,"entry":entry,"argv":argv,"params":params,"code_sha256":digest(&code),"adapter_sha256":adapter_hash}),
    )?;
    let mut command = Command::new(python);
    command
        .args(["-I", "-u", "-X", "utf8", "worker.py"])
        .current_dir(&folder)
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for name in [
        "SystemRoot",
        "WINDIR",
        "TEMP",
        "TMP",
        "LOCALAPPDATA",
        "USERNAME",
        "USERPROFILE",
    ] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .env(
            "PATH",
            format!(
                "{};{}\\System32",
                python.parent().unwrap_or(Path::new("")).display(),
                std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".into())
            ),
        )
        .env("CR_RUN_ROOT", folder.as_os_str())
        .env("PYTHONUTF8", "1")
        .env("OMP_NUM_THREADS", "2")
        .env("MKL_NUM_THREADS", "2")
        .env(
            "CUDA_VISIBLE_DEVICES",
            if is_training && params["device"] == "cuda" {
                "0"
            } else {
                "-1"
            },
        );
    #[cfg(windows)]
    command.creation_flags(0x08000000);
    let mut child = command
        .spawn()
        .map_err(|e| format!("启动 Python 失败：{e}"))?;
    // CUDA's Windows allocations also require host commit. The small CPU-probe
    // budget is insufficient for the published full batch, even with free VRAM.
    let memory_limit = if is_training { 8 } else { 4 } * 1024 * 1024 * 1024usize;
    let job = match attach_job(&child, memory_limit) {
        Ok(j) => j,
        Err(e) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(e);
        }
    };
    let report = json!({"run_id":request_id,"book_id":book_id,"activity_id":activity_id,"status":"running","started_at":super::now_secs(),"code_sha256":digest(&code),"params":params,"runtime":env,"inputs":recipe["inputs"],"pid":child.id(),"identity":"learner_run","adapter_sha256":adapter_hash,"monitor":recipe["monitor"],"security_mode":if use_draft&&!same_python_text(&code,&reference){"explicit_high_trust_native"}else{"reviewed_recipe_native"},"device":if is_training{params["device"].as_str().unwrap_or("cpu")}else{"cpu"},"artifacts":[]});
    if let Err(error) = atomic(&folder.join("run.json"), &report) {
        abort_spawn(&mut child, job);
        return Err(error);
    }
    let mut pumps = Vec::new();
    if let Some(s) = child.stdout.take() {
        pumps.push(pump(s, folder.join("stdout.log")));
    }
    if let Some(s) = child.stderr.take() {
        pumps.push(pump(s, folder.join("stderr.log")));
    }
    let Some(mut input) = child.stdin.take() else {
        abort_spawn(&mut child, job);
        return Err("启动通道不可用".into());
    };
    if let Err(error) = input.write_all(b"START\n") {
        abort_spawn(&mut child, job);
        return Err(error.to_string());
    }
    drop(input);
    let timeout = if is_training {
        training_budget(&params)?
    } else {
        recipe["timeout_seconds"].as_u64().unwrap_or(30).min(300)
    };
    let running = Arc::new(Mutex::new(Running {
        child,
        job,
        cancelled: false,
        folder: folder.clone(),
        report: report.clone(),
    }));
    map.insert(key, running.clone());
    preparation.armed = false;
    std::thread::spawn(move || {
        let began = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(100));
            let Ok(mut r) = running.lock() else { break };
            r.report["job_memory"] = job_memory(r.job);
            if began.elapsed() > Duration::from_secs(timeout) && !r.cancelled {
                r.report["diagnostic"] =
                    json!("达到本次时间预算，进程树已停止；代码与已有结果保留");
                kill_run(&mut r);
            }
            match r.child.try_wait() {
                Ok(Some(exit)) => {
                    #[cfg(windows)]
                    unsafe {
                        CloseHandle(r.job as _);
                    }
                    r.job = 0;
                    for handle in pumps.drain(..) {
                        let _ = handle.join();
                    }
                    r.report["status"] = json!(if r.cancelled {
                        "cancelled"
                    } else if exit.success() {
                        "succeeded"
                    } else {
                        "failed"
                    });
                    r.report["ended_at"] = json!(super::now_secs());
                    r.report["exit_code"] = json!(exit.code());
                    r.report["artifacts"] = json!(output_files(&r.folder, &r.report["inputs"]));
                    if let Ok(environment) = read_json(&r.folder.join("environment.json"), 64000) {
                        r.report["environment"] = environment;
                    }
                    let _ = atomic(&r.folder.join("run.json"), &r.report);
                    break;
                }
                Err(e) => {
                    r.report["status"] = json!("failed");
                    r.report["diagnostic"] = json!(e.to_string());
                    kill_run(&mut r);
                    let _ = r.child.wait();
                    #[cfg(windows)]
                    unsafe {
                        CloseHandle(r.job as _);
                    }
                    r.job = 0;
                    for handle in pumps.drain(..) {
                        let _ = handle.join();
                    }
                    let _ = atomic(&r.folder.join("run.json"), &r.report);
                    break;
                }
                _ => {}
            }
        }
    });
    Ok(report)
}
#[tauri::command]
pub fn learning_run_history(app: AppHandle, book_id: String) -> Result<Value, String> {
    let state = user_storage(&app, &book_id)?;
    let root = state.join("runs");
    if !root.exists() { return Ok(json!([])); }
    let mut rows = Vec::new();
    for entry in fs::read_dir(&root).map_err(|e| e.to_string())?.filter_map(Result::ok) {
        let id = entry.file_name().to_string_lossy().to_string();
        if token(&id).is_err() { continue; }
        let Ok(path) = safe_path(&state, &format!("runs/{id}/run.json")) else { continue };
        let Ok(record) = read_json(&path, 1024 * 1024) else { continue };
        if record["book_id"] != book_id || record["run_id"] != id { continue; }
        rows.push(json!({"run_id":id,"activity_id":record["activity_id"],"status":record["status"],
            "started_at":record["started_at"],"code_sha256":record["code_sha256"],"params":record["params"],"purpose":record["purpose"]}));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r["started_at"].as_u64().unwrap_or(0)));
    Ok(json!(rows))
}
#[tauri::command]
pub fn learning_run_snapshot(app: AppHandle, book_id: String, run_id: String) -> Result<Value, String> {
    token(&run_id)?;
    let root = user_storage(&app, &book_id)?;
    let record = read_json(&safe_path(&root, &format!("runs/{run_id}/run.json"))?, 1024 * 1024)?;
    if record["book_id"] != book_id || record["run_id"] != run_id { return Err("运行身份不匹配".into()); }
    let request = read_json(&safe_path(&root, &format!("runs/{run_id}/request.json"))?, 1024 * 1024)?;
    let bytes = verified(&root, &format!("runs/{run_id}/{}", text(&request, "entry")?), text(&record,"code_sha256")?, 300_000)?;
    let code = String::from_utf8(bytes).map_err(|_| "这次代码快照不是 UTF-8")?;
    let adapter_code=if let Some(hash)=record["adapter_sha256"].as_str(){Some(String::from_utf8(verified(&root,&format!("runs/{run_id}/book-adapter.py"),hash,300_000)?).map_err(|_|"适配快照不是 UTF-8")?)}else{None};
    Ok(json!({"code":code,"code_sha256":record["code_sha256"],"run_id":run_id,"adapter_code":adapter_code,"adapter_sha256":record["adapter_sha256"]}))
}
#[tauri::command]
pub fn learning_run_status(
    app: AppHandle,
    manager: State<'_, LearningManager>,
    book_id: String,
    run_id: String,
) -> Result<Value, String> {
    token(&run_id)?;
    let folder = user_storage(&app,&book_id)?.join("runs").join(&run_id);
    let key = format!("{book_id}-{run_id}");
    let map = manager.inner.lock().map_err(|_| "状态暂不可用")?;
    let mut report = if let Some(run) = map.get(&key) {
        run.lock().map_err(|_| "状态暂不可用")?.report.clone()
    } else {
        let mut value = read_json(&folder.join("run.json"), 1024 * 1024)?;
        if value["status"] == "running" || value["status"] == "preparing" {
            value["status"] = json!("interrupted");
            value["diagnostic"] = json!("应用已重启；上次进程树已结束，没有自动重跑");
            value["artifacts"] = json!(output_files(&folder, &value["inputs"]));
            atomic(&folder.join("run.json"), &value)?
        }
        value
    };
    if let Ok(environment)=read_json(&folder.join("environment.json"),64000){report["environment"]=environment;}
    for (key, file) in [("stdout", "stdout.log"), ("stderr", "stderr.log")] {
        let bytes = fs::read(folder.join(file)).unwrap_or_default();
        report[format!("{key}_truncated")] = json!(bytes.len() >= MAX_LOG);
        report[key] = json!(String::from_utf8_lossy(&bytes).to_string());
    }
    if let Some(checkpoint_path)=report["monitor"]["checkpoint"].as_str() {
        if let Ok(checkpoint) = safe_path(&folder,checkpoint_path).and_then(|p|read_json(&p,64000)) {
            report["live_checkpoint"] = checkpoint;
        }
        if let Some(log) = report["stdout"].as_str() {
            let latest = log
                .lines()
                .rev()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .find(|event| event["event"] == "step" || event["event"] == "epoch_end");
            if let Some(event) = latest {
                report["latest_event"] = event;
            }
        }
    }
    Ok(report)
}
#[tauri::command]
pub fn learning_cancel(
    manager: State<'_, LearningManager>,
    book_id: String,
    run_id: String,
) -> Result<(), String> {
    let map = manager.inner.lock().map_err(|_| "状态暂不可用")?;
    let run = map
        .get(&format!("{book_id}-{run_id}"))
        .ok_or("此任务不在运行")?;
    let mut r = run.lock().map_err(|_| "状态暂不可用")?;
    if r.report["status"] == "running" {
        kill_run(&mut r);
    }
    Ok(())
}
#[tauri::command]
pub async fn learning_artifact(
    app: AppHandle,
    book_id: String,
    run_id: String,
    artifact_path: String,
) -> Result<Response, String> {
    token(&run_id)?;
    let folder = user_storage(&app,&book_id)?.join("runs").join(run_id);
    let report = read_json(&folder.join("run.json"), 1024 * 1024)?;
    let a = report["artifacts"]
        .as_array()
        .and_then(|a| a.iter().find(|a| a["path"] == artifact_path))
        .ok_or("产物未登记")?;
    Ok(Response::new(verified(
        &folder,
        &artifact_path,
        text(a, "sha256")?,
        MAX_ASSET,
    )?))
}

#[tauri::command]
pub fn learning_open_source(
    app: AppHandle,
    book_id: String,
    source_id: String,
) -> Result<(), String> {
    let ctx = context(&app, &book_id)?;
    let source = ctx.pack["source_claims"]
        .as_array()
        .and_then(|all| all.iter().find(|s| s["id"] == source_id))
        .ok_or("原始来源未登记")?;
    learning_open_external(text(source,"url")?.to_string())
}
#[tauri::command]
pub fn learning_open_external(url:String)->Result<(),String>{
    let url = tauri::Url::parse(&url).map_err(|_| "来源地址无效")?;
    if !["https", "http"].contains(&url.scheme())
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err("只允许打开无凭据的 HTTP(S) 原始来源".into());
    }
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::UI::Shell::ShellExecuteW;
        let target: Vec<u16> = url.as_str().encode_utf16().chain(Some(0)).collect();
        let verb: Vec<u16> = "open".encode_utf16().chain(Some(0)).collect();
        let result = ShellExecuteW(
            std::ptr::null_mut(),
            verb.as_ptr(),
            target.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            1,
        );
        if result as isize <= 32 {
            return Err("系统浏览器未能打开；请复制原始链接".into());
        }
    }
    Ok(())
}

/// Store a host-builtin worker receipt in the same personal run collection.
/// This command writes bounded data; it cannot run code or grant native access.
#[tauri::command]
pub fn learning_store_builtin_run(app:AppHandle,book_id:String,mut record:Value,result:Value,code:String)->Result<Value,String>{
    let root=user_storage(&app,&book_id)?;
    let run_id=text(&record,"run_id")?.to_string();token(&run_id)?;
    token(text(&record,"activity_id")?)?;
    if code.len()>300_000||serde_json::to_vec(&record).map_err(|e|e.to_string())?.len()>1024*1024||serde_json::to_vec(&result).map_err(|e|e.to_string())?.len()>4*1024*1024{return Err("内置结果超过保存范围".into());}
    if record["book_id"]!=book_id||record["code_sha256"]!=digest(code.as_bytes())||!matches!(record["status"].as_str(),Some("running"|"succeeded"|"failed"|"cancelled"|"interrupted")){return Err("内置计算身份无效".into());}
    let folder=root.join("runs").join(&run_id);let path=folder.join("run.json");
    if path.exists(){let old=read_json(&path,1024*1024)?;if old["identity"]!="browser_builtin"||old["code_sha256"]!=record["code_sha256"]{return Err("不能覆盖其他运行的记录".into());}}
    record["identity"]=json!("browser_builtin");record["security_mode"]=json!("trusted_builtin_worker");
    fs::create_dir_all(folder.join("work/results")).map_err(|e|e.to_string())?;
    fs::write(folder.join("builtin.mjs"),code.as_bytes()).map_err(|e|e.to_string())?;
    atomic(&folder.join("request.json"),&json!({"entry":"builtin.mjs"}))?;
    if !result.is_null(){let data=serde_json::to_vec(&result).map_err(|e|e.to_string())?;fs::write(folder.join("work/results/report.json"),&data).map_err(|e|e.to_string())?;record["artifacts"]=json!([{"path":"work/results/report.json","extension":"json","bytes":data.len(),"sha256":digest(&data),"role":"builtin_result"}]);}
    atomic(&path,&record)?;Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn traversal_rejected() {
        let d = tempfile::tempdir().unwrap();
        for p in [
            "../x",
            "a/../../x",
            "C:/x",
            "/a",
            "a\\b",
            "file:stream",
            "%2e%2e/x",
            "a.",
            "a ",
            "NUL.txt",
            "COM1",
        ] {
            assert!(safe_path(d.path(), p).is_err(), "{p}")
        }
    }
    #[test]
    fn changed_file_rejected() {
        let d = tempfile::tempdir().unwrap();
        fs::write(d.path().join("a.txt"), b"original").unwrap();
        let h = digest(b"original");
        assert_eq!(verified(d.path(), "a.txt", &h, 100).unwrap(), b"original");
        fs::write(d.path().join("a.txt"), b"changed").unwrap();
        assert!(verified(d.path(), "a.txt", &h, 100).is_err());
    }
    #[test]
    fn draft_activity_listing_keeps_retired_history_without_a_pack() {
        let d = tempfile::tempdir().unwrap();
        assert!(draft_activity_ids(d.path()).unwrap().is_empty());
        fs::create_dir_all(d.path().join("drafts/retired/versions")).unwrap();
        fs::write(d.path().join("drafts/current.json"), b"{}").unwrap();
        fs::write(d.path().join("drafts/current.writing"), b"{}").unwrap();
        fs::create_dir_all(d.path().join("drafts/current/versions")).unwrap();
        assert_eq!(draft_activity_ids(d.path()).unwrap(), vec!["current", "retired"]);
    }
    #[test]
    fn draft_revision_reads_only_matching_bounded_code() {
        let d = tempfile::tempdir().unwrap();
        let code = "print(42)\n";
        let hash = digest(code.as_bytes());
        let folder = d.path().join("drafts").join("activity").join("versions");
        fs::create_dir_all(&folder).unwrap();
        let file = folder.join(format!("{hash}.json"));
        let current = d.path().join("drafts").join("activity.json");
        fs::write(&current, b"current stays unchanged").unwrap();
        atomic(&file, &json!({"code":code,"sha256":hash,"updated_at":7})).unwrap();

        assert_eq!(
            read_draft_revision(d.path(), "activity", &hash).unwrap(),
            json!({"code":code,"sha256":hash,"updated_at":7})
        );
        assert_eq!(fs::read(&current).unwrap(), b"current stays unchanged");
        assert!(read_draft_revision(d.path(), "../activity", &hash).is_err());
        assert!(read_draft_revision(d.path(), "activity", "../version").is_err());

        atomic(&file, &json!({"code":"changed","sha256":hash,"updated_at":7})).unwrap();
        assert!(read_draft_revision(d.path(), "activity", &hash).is_err());
        atomic(&file, &json!({"code":code,"sha256":hash,"updated_at":7,"padding":"x".repeat(400_000)})).unwrap();
        assert!(read_draft_revision(d.path(), "activity", &hash).is_err());
    }
    #[test]
    fn limits_are_real() {
        assert!(training_budget(&json!({"device":"cuda","budget_seconds":901})).is_err());
        assert!(registered_arguments("unknown",&json!({}),&json!({})).is_err());
        assert!(registered_arguments("unknown",&json!({}),&json!({"adapter":{"reviewed":false}})).is_err());
    }
    #[test]
    fn generic_recipe_uses_typed_host_registration() {
        let recipe = json!({"generic_native":true,"reviewed_pure_local":true,"timeout_seconds":20,"argv":[],"parameters":{"factor":{"flag":"--factor","type":"number","default":2,"min":-10,"max":10}}});
        assert_eq!(
            registered_arguments("new-book-example", &json!({"factor":3}), &recipe).unwrap(),
            vec!["--factor=3"]
        );
        assert!(
            registered_arguments("new-book-example", &json!({"factor":"3; command"}), &recipe)
                .is_err()
        );
        assert!(
            registered_arguments("new-book-example", &json!({"shell":"anything"}), &recipe)
                .is_err()
        );
        assert!(registered_arguments("new-book-example", &json!({"factor":100}), &recipe).is_err());
        let mut unreviewed = recipe.clone();
        unreviewed["reviewed_pure_local"] = json!(false);
        assert!(registered_arguments("new-book-example", &json!({}), &unreviewed).is_err());
    }
    #[test]
    fn hostile_pack_rejected_before_rendering() {
        let mut pack = json!({"assets":[{"id":"a","kind":"code","relative_path":"code/a.py","sha256":"0".repeat(64)}],"chapters":[{"id":"c","href":"chapter.xhtml"}],"activities":[{"id":"act","chapter":"c","entry_asset":"a"}],"source_claims":[]});
        assert!(validate_pack(&pack).is_ok());
        pack["activities"][0]["command"] = json!("arbitrary shell");
        assert!(validate_pack(&pack).is_err());
        pack["activities"][0]
            .as_object_mut()
            .unwrap()
            .remove("command");
        pack["assets"][0]["relative_path"] = json!("../escape");
        assert!(validate_pack(&pack).is_err());
        pack["assets"][0]["relative_path"] = json!("code/a.py");
        pack["assets"][0]["kind"] = json!("executable-widget");
        assert!(validate_pack(&pack).is_err());
    }
}
