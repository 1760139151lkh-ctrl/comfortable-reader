"""Create a private, review-pending book from Markdown or a simple EPUB.

The EPUB path deliberately rejects structures that this format cannot preserve.
It never edits the source, public catalog, or a personal reading library.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import posixpath
import re
import sys
import uuid
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from book_format import BookError, valid_id, validate_book

MAX_EPUB_MEMBER = 2 * 1024 * 1024
MAX_EPUB_ENTRIES = 1000
MAX_EPUB_TOTAL = 256 * 1024 * 1024
OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
XHTML_NS = "http://www.w3.org/1999/xhtml"


def epub_member(archive: zipfile.ZipFile, name: str) -> bytes:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or name.startswith("/") or any(x in ("", ".", "..") for x in name.split("/")):
        raise BookError(f"EPUB 内部地址不安全：{name!r}")
    try:
        item = archive.getinfo(name)
    except KeyError as error:
        raise BookError(f"EPUB 缺少内部文件：{name}") from error
    if item.file_size > MAX_EPUB_MEMBER:
        raise BookError(f"EPUB 的这一章节或清单过大，不能作为简单导入：{name}")
    data = archive.read(item)
    if len(data) != item.file_size:
        raise BookError(f"EPUB 文件长度不符：{name}")
    return data


def nested_text(node: ET.Element) -> str:
    def walk(item: ET.Element) -> str:
        local = item.tag.rsplit("}", 1)[-1]
        if local not in ("p", "h1", "h2", "h3", "li", "strong", "span", "b"):
            raise BookError(f"简单 EPUB 导入暂不保真处理 <{local}>；请改用原 EPUB 阅读，或手工迁移并核对")
        text = item.text or ""
        for child in item:
            part = walk(child)
            child_name = child.tag.rsplit("}", 1)[-1]
            if child_name in ("strong", "b"):
                if "**" in part:
                    raise BookError("嵌套强调需手工核对")
                part = f"**{part}**"
            text += part + (child.tail or "")
        return text
    result = " ".join(walk(node).split())
    if not result:
        raise BookError("简单 EPUB 中有空结构；需手工检查后再转换")
    if any(x in result for x in ("@activity ", "@source ", "@resource ", "@chapter ", "```", "~~~", "<", "![")):
        raise BookError("原 EPUB 文本包含本格式的控制记号；需手工转义与核对")
    return result


def epub_chapter(data: bytes, fallback_title: str) -> tuple[str, str]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise BookError("EPUB 的 XHTML 无法作为 XML 解析") from error
    body = root.find(f".//{{{XHTML_NS}}}body")
    if body is None:
        raise BookError("EPUB 章节缺少 XHTML 正文")
    lines: list[str] = []
    title = None
    for node in body:
        local = node.tag.rsplit("}", 1)[-1]
        if local in ("h1", "h2", "h3"):
            value = nested_text(node)
            if title is None:
                title = value
            lines.append(f"{'#' * int(local[1])} {value}")
        elif local == "p":
            lines.append(nested_text(node))
        elif local in ("ul", "ol"):
            if local == "ol":
                raise BookError("原 EPUB 有有序列表；此导入器不把序号关系无声改成无序列表")
            for child in node:
                if child.tag.rsplit("}", 1)[-1] != "li":
                    raise BookError("EPUB 列表含不支持的节点")
                lines.append("- " + nested_text(child))
        else:
            raise BookError(f"简单 EPUB 导入暂不保真处理正文 <{local}>；原件仍可直接读")
        lines.append("")
    if not lines:
        raise BookError("EPUB 的章节没有可转换的正文")
    if lines[0] and not lines[0].startswith("# "):
        lines.insert(0, f"# {fallback_title}")
        lines.insert(1, "")
    elif not lines[0]:
        lines.insert(0, f"# {fallback_title}")
    return "\n".join(lines).rstrip() + "\n", title or fallback_title


def extract_simple_epub(source: Path) -> tuple[list[tuple[str, str]], dict]:
    if source.stat().st_size > MAX_EPUB_TOTAL:
        raise BookError("EPUB 超过简单导入预算；请先审查原件并按章迁移")
    with zipfile.ZipFile(source) as archive:
        items = archive.infolist()
        if len(items) > MAX_EPUB_ENTRIES or sum(x.file_size for x in items) > MAX_EPUB_TOTAL:
            raise BookError("EPUB 内部文件数或展开尺寸过大")
        if archive.read("mimetype") != b"application/epub+zip":
            raise BookError("EPUB 缺少正确的 mimetype")
        container = ET.fromstring(epub_member(archive, "META-INF/container.xml"))
        roots = container.findall(f".//{{{CONTAINER_NS}}}rootfile")
        if len(roots) != 1:
            raise BookError("EPUB 应有且只有一个 OPF 主清单")
        opf_path = roots[0].get("full-path")
        opf = ET.fromstring(epub_member(archive, opf_path))
        title = opf.findtext(f".//{{{DC_NS}}}title") or source.stem
        creator = opf.findtext(f".//{{{DC_NS}}}creator")
        declared = {row.get("id"): row for row in opf.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item")}
        spine = [x.get("idref") for x in opf.findall(f".//{{{OPF_NS}}}spine/{{{OPF_NS}}}itemref")]
        if not spine or len(spine) > 500:
            raise BookError("EPUB 阅读顺序为空或过长")
        base = posixpath.dirname(opf_path)
        chapters = []
        for number, entry in enumerate(spine, 1):
            item = declared.get(entry)
            if item is None or item.get("media-type") != "application/xhtml+xml":
                raise BookError("EPUB spine 引用了缺失或非 XHTML 内容")
            href = item.get("href")
            if not href or "?" in href or "#" in href or href.startswith("/"):
                raise BookError("EPUB spine 地址需要人工核对")
            name = posixpath.normpath(posixpath.join(base, href))
            if name.startswith("../") or name == "..":
                raise BookError("EPUB spine 越过清单目录")
            draft, heading = epub_chapter(epub_member(archive, name), f"第 {number} 章")
            chapters.append((heading, draft))
        provenance = {"sourceKind": "simple_epub", "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(), "humanReview": "required", "originalCreator": creator}
        return chapters, {"title": title, "provenance": provenance}


def markdown_chapters(source: Path) -> list[tuple[str, str]]:
    files = sorted(source.glob("*.md")) if source.is_dir() else [source]
    if not files or len(files) > 500:
        raise BookError("请指定一个 Markdown 文件，或含 1—500 个 .md 的目录")
    result = []
    for file in files:
        if file.is_symlink() or not file.is_file() or file.stat().st_size > MAX_EPUB_MEMBER:
            raise BookError(f"章节文件不是普通文件或过大：{file.name}")
        text = file.read_text(encoding="utf-8")
        first = text.lstrip().splitlines()[0] if text.strip() else ""
        match = re.fullmatch(r"#\s+(.+?)(?:\s+\{#[a-z][a-z0-9-]*\})?\s*", first)
        if not match:
            raise BookError(f"{file.name}: 每章需要以 # 标题开始；不会擅自为原文补标题")
        result.append((match.group(1), text))
    return result


def create(source: Path, kind: str, output: Path, slug: str, title: str | None, author: str | None) -> dict:
    valid_id(slug, "书籍目录名")
    if output.exists():
        raise BookError("目标目录已存在；为保护原稿，请指定一个新目录")
    if output.name != slug:
        raise BookError("输出目录名需与 --slug 一致")
    if not source.exists() or source.is_symlink():
        raise BookError("原稿不存在或是符号链接")
    if kind == "markdown":
        chapters = markdown_chapters(source)
        imported = {"sourceKind": "markdown", "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None, "humanReview": "required"}
        metadata = {"title": title or chapters[0][0], "provenance": imported}
    else:
        chapters, metadata = extract_simple_epub(source)
    output.parent.mkdir(parents=True,exist_ok=True)
    parent = output.parent.resolve(strict=True)
    if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
        raise BookError("输出父目录不能是目录联接")
    staging = parent / f".{slug}-{uuid.uuid4().hex}.building"
    staging.mkdir()
    try:
        (staging / "chapters").mkdir()
        manifest = {"schemaVersion": 1, "id": f"urn:uuid:{uuid.uuid4()}", "slug": slug,
                    "revision": "0.1.0", "revisionModified":datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), "language": "zh-CN", "title": title or metadata["title"],
                    "description": "请写明这本书带读者解决什么问题。", "authorship": author or metadata["provenance"].get("originalCreator") or "作者待填写",
                    "rights": {"status": "review_required", "scope": "private_draft"},
                    "importProvenance": metadata["provenance"], "chapters": [], "resources": [], "sources": [], "activities": []}
        for index, (heading, body) in enumerate(chapters, 1):
            name = f"{index:02d}.md"; (staging / "chapters" / name).write_text(body, encoding="utf-8")
            manifest["chapters"].append({"id": f"c{index:02d}", "title": heading, "source": f"chapters/{name}"})
        (staging / "book.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        validate_book(staging, slug)
        staging.rename(output)
        return {"status": "private_draft_created", "bookId": manifest["id"], "chapters": len(chapters), "path": str(output), "publicCatalogChanged": False}
    except Exception:
        # This directory is created by this invocation, under the verified parent.
        for file in (staging / "chapters").glob("*.md") if (staging / "chapters").exists() else []:
            file.unlink()
        if (staging / "chapters").exists(): (staging / "chapters").rmdir()
        if (staging / "book.json").exists(): (staging / "book.json").unlink()
        staging.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a private portable book draft; never changes the public catalog")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--markdown", type=Path)
    group.add_argument("--epub", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new directory whose name equals --slug")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title")
    parser.add_argument("--author")
    args = parser.parse_args()
    source = args.markdown or args.epub
    kind = "markdown" if args.markdown else "epub"
    print(json.dumps(create(source, kind, args.out, args.slug, args.title, args.author), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (BookError, OSError, UnicodeError, ValueError, zipfile.BadZipFile, ET.ParseError) as error:
        print(f"建书停止：{error}", file=sys.stderr)
        raise SystemExit(2)
