"""Build-time parser and validator for the public, inert book format.

Only declared files under one book root can enter the output. Book-authored
HTML and scripts are deliberately not interpreted or copied.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import re
import uuid

ID = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
VERSION = re.compile(r"^\d+\.\d+\.\d+$")
MODIFIED = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
CAPABILITY = re.compile(r"^[a-z][a-z0-9-]*@[1-9]\d*$")
DIRECTIVE = re.compile(r"(\*\*[^*\n]+\*\*|@(?:activity|resource|source|chapter) [a-z][a-z0-9-]*(?:#[a-z][a-z0-9-]*)?)")
SUPPORTED_CAPABILITIES = {"least-squares@1"}
MAX_CHAPTER_BYTES = 2 * 1024 * 1024
MAX_RESOURCE_BYTES = 128 * 1024 * 1024


class BookError(ValueError):
    pass


def valid_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise BookError(f"{what} 应使用小写字母、数字与短横线，并以字母开头")
    return value


def safe_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "%" in relative or "?" in relative or "#" in relative:
        raise BookError(f"资源地址必须是清单目录下的相对路径：{relative!r}")
    parts = relative.split("/")
    if any(not part or part in (".", "..") or part.endswith((".", " ")) for part in parts):
        raise BookError(f"资源地址含空段或上级目录：{relative!r}")
    for part in parts:
        if re.match(r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", part, re.I):
            raise BookError(f"资源地址含 Windows 设备名：{relative!r}")
    if root.is_symlink() or (hasattr(root, 'is_junction') and root.is_junction()):
        raise BookError(f"书籍根目录不能是符号链接或目录联接：{root.name}")
    base = root.resolve(strict=True)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise BookError(f"资源路径包含符号链接或重解析点：{relative!r}")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise BookError(f"资源越过书籍目录或不是文件：{relative!r}")
    return resolved


def file_identity(file: Path) -> dict:
    return {"bytes": file.stat().st_size, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}


def inline_nodes(text: str, source: str, line: int) -> list[dict]:
    out = []
    end = 0
    for match in DIRECTIVE.finditer(text):
        if match.start() > end:
            out.append({"type": "text", "text": text[end:match.start()]})
        part = match.group(0)
        if part.startswith("**"):
            out.append({"type": "strong", "text": part[2:-2]})
        else:
            kind, target = part[1:].split(" ", 1)
            key, _, anchor = target.partition("#")
            out.append({"type": kind, "id": valid_id(key, f"{source}:{line} 的 {kind} 身份"), **({"anchor": valid_id(anchor, f"{source}:{line} 的段落锚点")} if anchor else {})})
        end = match.end()
    if end < len(text):
        out.append({"type": "text", "text": text[end:]})
    if not out:
        out.append({"type": "text", "text": ""})
    if re.search(r"<\s*/?\s*[A-Za-z]|!\[[^]]*\]\(", text):
        raise BookError(f"{source}:{line} 含原始 HTML/图片语法；改用已登记的资源入口")
    if "**" in "".join(x.get("text", "") for x in out):
        raise BookError(f"{source}:{line} 有未配对的强调标记")
    return out


def parse_markdown(file: Path, root: Path) -> dict:
    if file.stat().st_size > MAX_CHAPTER_BYTES:
        raise BookError(f"章节超过 {MAX_CHAPTER_BYTES} 字节预算：{file.name}")
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise BookError(f"章节不是 UTF-8：{file.name}") from error
    nodes: list[dict] = []
    anchors: set[str] = set()
    refs: list[dict] = []
    rel = file.relative_to(root.resolve(strict=True)).as_posix()
    paragraph: list[str] = []
    paragraph_line = 1

    def append(node: dict):
        nodes.append(node)
        if len(nodes) > 8000:
            raise BookError(f"{rel} 结构超过 8000 项；按主题拆章")

    def flush():
        nonlocal paragraph
        if paragraph:
            value = " ".join(paragraph)
            body = inline_nodes(value, rel, paragraph_line)
            append({"type": "paragraph", "content": body})
            refs.extend(x for x in body if x["type"] in ("activity", "resource", "source", "chapter"))
            paragraph = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush(); i += 1; continue
        number = i + 1
        if stripped.startswith("#"):
            flush()
            match = re.fullmatch(r"(#{1,3})\s+(.+?)(?:\s+\{#([a-z][a-z0-9-]*)\})?", stripped)
            if not match:
                raise BookError(f"{rel}:{number} 标题需 #/##/### 与正文，锚点用 {{#id}}")
            level, title, anchor = len(match[1]), match[2], match[3]
            anchor = anchor or f"h-{number}"
            if anchor in anchors:
                raise BookError(f"{rel}:{number} 段落锚点重复：{anchor}")
            anchors.add(anchor)
            append({"type": "heading", "level": level, "id": anchor, "content": inline_nodes(title, rel, number)})
            i += 1; continue
        if stripped.startswith("@activity ") and stripped == line:
            flush(); key = valid_id(stripped[10:], f"{rel}:{number} 的活动")
            append({"type": "activity", "id": key}); refs.append({"type": "activity", "id": key})
            i += 1; continue
        if stripped.startswith("@image ") and stripped == line:
            flush(); key = valid_id(stripped[7:], f"{rel}:{number} 的插图")
            append({"type": "image", "id": key}); refs.append({"type": "image", "id": key})
            i += 1; continue
        if stripped.startswith("|"):
            flush()
            if i + 1 >= len(lines) or not re.fullmatch(r"\|?\s*:?[-]{3,}:?(\s*\|\s*:?[-]{3,}:?)*\s*\|?", lines[i + 1].strip()):
                raise BookError(f"{rel}:{number} 表格缺表头分隔行")
            cell = lambda value: [x.strip() for x in value.strip().strip("|").split("|")]
            header = cell(line)
            if not 1 <= len(header) <= 16:
                raise BookError(f"{rel}:{number} 表格列数应在 1—16")
            i += 2; rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                values = cell(lines[i])
                if len(values) != len(header) or len(rows) >= 1000:
                    raise BookError(f"{rel}:{i+1} 表格列数不同或行数过多")
                rows.append([inline_nodes(x, rel, i + 1) for x in values]); i += 1
            append({"type": "table", "head": [inline_nodes(x, rel, number) for x in header], "rows": rows})
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush(); marker = stripped[:3]; language = stripped[3:].strip()
            if language and not re.fullmatch(r"[a-zA-Z0-9_-]{1,24}", language):
                raise BookError(f"{rel}:{number} 代码语言标记无效")
            begin = number; i += 1; body = []
            while i < len(lines) and lines[i].strip() != marker:
                body.append(lines[i]); i += 1
            if i >= len(lines):
                raise BookError(f"{rel}:{begin} 代码块没有结束标记")
            append({"type": "code", "language": language, "text": "\n".join(body)})
            i += 1; continue
        if line.startswith("- "):
            flush(); items = []
            while i < len(lines) and lines[i].startswith("- "):
                body = inline_nodes(lines[i][2:], rel, i + 1)
                items.append(body);refs.extend(x for x in body if x["type"] in ("activity", "resource", "source", "chapter"));i += 1
            append({"type": "list", "items": items});continue
        if stripped.startswith(("<", "![", "<!--")) or re.match(r"^\d+[.)]\s", stripped):
            raise BookError(f"{rel}:{number} 含尚未验证的 Markdown/HTML 结构；请改为支持的章节元素或提交新转换能力")
        if not paragraph: paragraph_line = number
        paragraph.append(stripped)
        i += 1
    flush()
    if not nodes or nodes[0]["type"] != "heading":
        raise BookError(f"{rel}:1 章节须以标题开始")
    return {"schemaVersion": 1, "nodes": nodes, "anchors": sorted(anchors), "references": refs}


def validate_book(root: Path, expected_slug: str) -> tuple[dict, dict[str, dict]]:
    data = json.loads((root/"book.json").read_text(encoding="utf-8"))
    if data.get('sourceFormat') == 'epub-source@1':
        from rich_book import validate_rich_book
        return validate_rich_book(root, data, expected_slug)
    if data.get("schemaVersion") != 1 or data.get("slug") != expected_slug or not ID.fullmatch(expected_slug):
        raise BookError(f"{expected_slug}: schemaVersion/slug 不匹配")
    try: urn=uuid.UUID(data["id"].removeprefix("urn:uuid:"))
    except (ValueError,KeyError,TypeError) as error: raise BookError(f"{expected_slug}: 需要稳定 UUID URN") from error
    if data["id"] != f"urn:uuid:{urn}" or not VERSION.fullmatch(data.get("revision", "")):
        raise BookError(f"{expected_slug}: UUID/revision 格式不合约定")
    if not MODIFIED.fullmatch(data.get("revisionModified", "")):
        raise BookError(f"{expected_slug}: 需要该修订的 UTC revisionModified 时间")
    if not data.get("title") or not data.get("language") or len(data.get("chapters",[])) > 500:
        raise BookError(f"{expected_slug}: 缺标题/语种或章节过多")
    chapters={}; documents={}; resources={}; activities={}; sources={}
    for chapter in data["chapters"]:
        key=valid_id(chapter["id"], "章节 ID")
        if key in chapters:raise BookError(f"{expected_slug}: 重复章节 {key}")
        source=safe_relative(root,chapter["source"])
        if source.suffix.lower()!='.md':raise BookError(f"{key}: 此入口只接受 Markdown 源章")
        documents[key]=parse_markdown(source,root)
        chapters[key]=chapter
    for res in data.get("resources",[]):
        key=valid_id(res["id"],"资源 ID")
        if key in resources:raise BookError(f"{expected_slug}: 重复资源 {key}")
        source=safe_relative(root,res["path"])
        if source.stat().st_size > MAX_RESOURCE_BYTES:raise BookError(f"{key}: 资源大于当前浏览器单件读取预算；请定义独立发布分块")
        if res.get("kind") not in ("code","data","image","audio","video","pdf","model","geometry","text") or res.get("minimumUnit")!="file":
            raise BookError(f"{key}: 资源类型或最低获取单元不受支持")
        if res["kind"]=="video":
            if res.get("mediaType")!="video/webm" or source.read_bytes()[:4]!=b'\x1a\x45\xdf\xa3':
                raise BookError(f"{key}: 当前视频需要实际 WebM 容器与明确身份")
        if res["kind"]=="audio":
            if res.get("mediaType")!="audio/wav" or not source.read_bytes().startswith(b'RIFF'):
                raise BookError(f"{key}: 当前音频需要实际 WAV 容器与明确身份")
        if res["kind"]=="image":
            header=source.read_bytes()[:24]
            if res.get("mediaType")!="image/png" or not header.startswith(b'\x89PNG\r\n\x1a\n'):
                raise BookError(f"{key}: 当前图片需要实际 PNG 文件；SVG/HTML 不直接进入宿主")
            width=int.from_bytes(header[16:20],'big');height=int.from_bytes(header[20:24],'big')
            if not 1<=width<=8192 or not 1<=height<=8192 or width*height>32_000_000:
                raise BookError(f"{key}: 图片尺寸超过当前解码预算")
        chunk=res.get("downloadChunkBytes")
        if chunk is not None and (not isinstance(chunk,int) or not 1024<=chunk<=4*1024*1024):
            raise BookError(f"{key}: 分块尺寸须为 1 KiB—4 MiB")
        resources[key]=res
    for key,res in resources.items():
        if not isinstance(res.get("dependsOn",[]),list) or any(item not in resources or item==key for item in res.get("dependsOn",[])):
            raise BookError(f"{key}: 必需依赖未知或自指")
    def check_dependencies(key,trail):
        if len(trail)>32:raise BookError(f"{key}: 资源依赖深度超过 32")
        if key in trail:raise BookError(f"资源依赖形成循环：{' → '.join((*trail,key))}")
        for dependency in resources[key].get("dependsOn",[]):check_dependencies(dependency,(*trail,key))
    for key in resources:check_dependencies(key,())
    for chapter_id,document in documents.items():
        essential=chapters[chapter_id].setdefault('essential',[])
        if not isinstance(essential,list) or len(set(essential))!=len(essential) or any(key not in resources for key in essential):
            raise BookError(f"{chapter_id}: 章节必需资源未登记或重复")
        for node in document['nodes']:
            if node['type']=='image':
                key=node['id']
                if key not in resources or resources[key]['kind']!='image':
                    raise BookError(f"{chapter_id}: 插图 {key} 需登记为 image 资源")
                if key not in essential:essential.append(key)
    for src in data.get("sources",[]):
        key=valid_id(src["id"],"来源 ID")
        if key in sources:raise BookError(f"{expected_slug}: 重复来源 {key}")
        url=src.get("url")
        if url is not None and (not isinstance(url,str) or not url.startswith("https://") or any(x in url for x in ("@", "\\", "#"))):
            raise BookError(f"{key}: 外部来源需要无凭据、无片段的 HTTPS URL；页段用 locator")
        sources[key]=src
    for activity in data.get("activities",[]):
        key=valid_id(activity["id"],"活动 ID")
        if any(name in activity for name in ("shell","command","cwd","pythonPath","python_path","env","trusted","script","permissions")):
            raise BookError(f"{key}: 书籍活动不得声明命令或本机执行权限")
        if key in activities or activity["chapter"] not in chapters:
            raise BookError(f"{expected_slug}: 活动 {key} 重复或所属章不存在")
        if not CAPABILITY.fullmatch(activity.get("capability","")) or activity["capability"] not in SUPPORTED_CAPABILITIES:
            raise BookError(f"{key}: 能力版本未在此阅读器登记；需另加受审查的扩展")
        if activity.get("anchor") not in documents[activity["chapter"]]["anchors"]:
            raise BookError(f"{key}: 章内锚点不存在")
        needed=activity.get("required",[]);optional=activity.get("optional",[])
        if not needed or any(x not in resources for x in needed+optional) or len(set(needed+optional))!=len(needed+optional):
            raise BookError(f"{key}: 必需/可选资源遗漏或重复")
        if activity["capability"]=="least-squares@1" and (len(needed)!=1 or resources[needed[0]]["kind"]!="data"):
            raise BookError(f"{key}: least-squares@1 需要且只需一份声明的数据输入")
        for pname,spec in activity.get("parameters",{}).items():
            if not ID.fullmatch(pname) or spec.get("type") not in ("number","integer") or not isinstance(spec.get("default"),(float,int)) or not spec["min"]<=spec["default"]<=spec["max"]:
                raise BookError(f"{key}: 参数 {pname} 的类型、默认值或范围无效")
        if set(activity.get("parameters",{}))!={"rate","steps"}:
            raise BookError(f"{key}: least-squares@1 需要 rate/steps 的明确范围")
        activities[key]=activity
    for chapter_id,doc in documents.items():
        for ref in doc["references"]:
            kind,target=ref["type"],ref["id"]
            if kind=="activity" and (target not in activities or activities[target]["chapter"]!=chapter_id):raise BookError(f"{chapter_id}: 活动入口 {target} 未登记在本章")
            if kind=="resource" and target not in resources:raise BookError(f"{chapter_id}: 资源入口 {target} 未登记")
            if kind=="image" and (target not in resources or resources[target]['kind']!='image'):raise BookError(f"{chapter_id}: 插图 {target} 未登记")
            if kind=="source" and target not in sources:raise BookError(f"{chapter_id}: 来源入口 {target} 未登记")
            if kind=="chapter" and (target not in chapters or (ref.get("anchor") and ref["anchor"] not in documents[target]["anchors"])):
                raise BookError(f"{chapter_id}: 目标章或段落不存在：{target}")
    return data,documents
