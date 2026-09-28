#!/usr/bin/env python3
"""Build a reflowable EPUB, optionally add it to calibre, and optionally make an HTML fallback."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import webbrowser
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET


SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = SKILL_DIR / "assets" / "reader-template.html"
VENDOR_PATH = SKILL_DIR / "vendor"
# Optional per-machine binding for legacy private imports. Shareable book
# source projects use the repository's book builder instead of this path.
PORTABLE_CALIBRE_ROOT = Path(os.environ["COMFORTABLE_READER_CALIBRE_ROOT"]).expanduser() if os.environ.get("COMFORTABLE_READER_CALIBRE_ROOT") else None
SUPPORTED_DIRECT = {".md", ".markdown", ".txt", ".html", ".htm", ".docx", ".epub", ".pdf"}
MARKDOWN_PLUGINS = ["table", "strikethrough", "task_lists", "url", "footnotes"]
MATH_BOX_CLASS = "comfortable-math-box"

if VENDOR_PATH.is_dir() and str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_rendering import MathRenderingError, render_mathml


class BuildError(RuntimeError):
    """Raised for user-facing conversion failures."""


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "big5", "cp1252"):
        try:
            decoded = raw.decode(encoding)
            if "\ufffd" in decoded:
                raise BuildError(f"Source text contains the Unicode replacement character U+FFFD: {path}")
            if "\x00" in decoded:
                raise BuildError(f"Source text contains NUL bytes and is not safe to treat as text: {path}")
            return decoded
        except UnicodeDecodeError:
            continue
    raise BuildError(f"Could not decode the source text without replacing characters: {path}")


def strip_tags(value: str) -> str:
    value = re.sub(r"<script\b[^>]*>.*?</script>", "", value, flags=re.I | re.S)
    value = re.sub(r"<style\b[^>]*>.*?</style>", "", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def safe_inline_markdown(value: str) -> str:
    escaped = html.escape(value, quote=False)
    code_tokens: list[str] = []

    def save_code(match: re.Match[str]) -> str:
        code_tokens.append(f"<code>{html.escape(match.group(1), quote=False)}</code>")
        return f"\x00CODE{len(code_tokens) - 1}\x00"

    escaped = re.sub(r"`([^`]+)`", save_code, escaped)
    escaped = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)", r'<img alt="\1" src="\2">', escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)", r'<a href="\2">\1</a>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)", lambda m: f"<em>{m.group(1) or m.group(2)}</em>", escaped)
    escaped = re.sub(r"~~([^~]+)~~", r"<del>\1</del>", escaped)
    for index, token in enumerate(code_tokens):
        escaped = escaped.replace(f"\x00CODE{index}\x00", token)
    return escaped


def fallback_markdown(source: str) -> str:
    """A dependency-free Markdown subset used only when no Markdown library exists."""
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        fence = re.match(r"^\s*(```+|~~~+)\s*([\w.+-]*)\s*$", line)
        if fence:
            marker = fence.group(1)
            language = fence.group(2)
            code: list[str] = []
            index += 1
            while index < len(lines) and not re.match(rf"^\s*{re.escape(marker[0])}{{{len(marker)},}}\s*$", lines[index]):
                code.append(lines[index])
                index += 1
            index += 1
            class_name = f' class="language-{html.escape(language)}"' if language else ""
            output.append(f"<pre><code{class_name}>{html.escape(chr(10).join(code))}</code></pre>")
            continue

        heading = re.match(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level = len(heading.group(1))
            output.append(f"<h{level}>{safe_inline_markdown(heading.group(2))}</h{level}>")
            index += 1
            continue

        if re.match(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", line):
            output.append("<hr>")
            index += 1
            continue

        if re.match(r"^\s*>\s?", line):
            quote: list[str] = []
            while index < len(lines) and re.match(r"^\s*>\s?", lines[index]):
                quote.append(re.sub(r"^\s*>\s?", "", lines[index]))
                index += 1
            output.append(f"<blockquote><p>{safe_inline_markdown(' '.join(quote))}</p></blockquote>")
            continue

        list_match = re.match(r"^\s*([-+*]|\d+[.)])\s+(.+)$", line)
        if list_match:
            ordered = list_match.group(1)[0].isdigit()
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while index < len(lines):
                current = re.match(r"^\s*([-+*]|\d+[.)])\s+(.+)$", lines[index])
                if not current or current.group(1)[0].isdigit() != ordered:
                    break
                item = current.group(2)
                task = re.match(r"^\[([ xX])\]\s*(.*)$", item)
                if task:
                    checked = " checked" if task.group(1).lower() == "x" else ""
                    items.append(f'<li class="task-list-item"><input type="checkbox" disabled{checked}> {safe_inline_markdown(task.group(2))}</li>')
                else:
                    items.append(f"<li>{safe_inline_markdown(item)}</li>")
                index += 1
            output.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines) and lines[index].strip():
            candidate = lines[index]
            if re.match(r"^\s*(#{1,6})\s+", candidate) or re.match(r"^\s*(```+|~~~+|>|[-+*]\s+|\d+[.)]\s+)", candidate):
                break
            paragraph.append(candidate.strip())
            index += 1
        output.append(f"<p>{safe_inline_markdown(' '.join(paragraph))}</p>")
    return "\n".join(output)


def prepare_math_markdown(source: str, warnings: list[str]) -> tuple[str, list[tuple[str, str, bool]]]:
    """Replace ChatGPT-style LaTeX delimiters outside code with restorable MathML tokens."""
    token_prefix = f"COMFORTABLEREADERMATHTOKEN{uuid.uuid4().hex.upper()}"
    code_prefix = f"COMFORTABLEREADERCODETOKEN{uuid.uuid4().hex.upper()}"
    replacements: list[tuple[str, str, bool]] = []
    failed = 0

    def convert_formula(latex: str, display: bool) -> str:
        nonlocal failed
        try:
            mathml = render_mathml(latex, display)
            wrapper = "div" if display else "span"
            css_class = "math-block" if display else "math-inline"
            return f'<{wrapper} class="{css_class}">{mathml}</{wrapper}>'
        except MathRenderingError as exc:
            failed += 1
            warnings.append(f"Formula {len(replacements) + 1}: {exc}")
        delimiters = ("\\[", "\\]") if display else ("\\(", "\\)")
        return f'<code class="math-source">{html.escape(delimiters[0] + latex.strip() + delimiters[1])}</code>'

    def transform_segment(segment: str) -> str:
        code_spans: list[str] = []

        def protect_code(match: re.Match[str]) -> str:
            token = f"{code_prefix}{len(code_spans):06d}END"
            code_spans.append(match.group(0))
            return token

        segment = re.sub(r"(`+)([^\n]*?)\1", protect_code, segment)

        def replace_math(match: re.Match[str], display: bool) -> str:
            token = f"{token_prefix}{len(replacements):06d}END"
            replacements.append((token, convert_formula(match.group(1), display), display))
            return f"\n{token}\n" if display else token

        segment = re.sub(r"\\\[(.+?)\\\]", lambda match: replace_math(match, True), segment, flags=re.S)
        segment = re.sub(r"\$\$(.+?)\$\$", lambda match: replace_math(match, True), segment, flags=re.S)
        segment = re.sub(r"\\\((.+?)\\\)", lambda match: replace_math(match, False), segment, flags=re.S)
        for index, code_span in enumerate(code_spans):
            segment = segment.replace(f"{code_prefix}{index:06d}END", code_span)
        return segment

    lines = source.splitlines(keepends=True)
    rendered_parts: list[str] = []
    plain_buffer: list[str] = []
    fence_character: str | None = None
    fence_length = 0

    def flush_plain() -> None:
        if plain_buffer:
            rendered_parts.append(transform_segment("".join(plain_buffer)))
            plain_buffer.clear()

    for line in lines:
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence_character is None:
            if fence:
                flush_plain()
                marker = fence.group(1)
                fence_character = marker[0]
                fence_length = len(marker)
                rendered_parts.append(line)
            else:
                plain_buffer.append(line)
        else:
            rendered_parts.append(line)
            if fence and fence.group(1)[0] == fence_character and len(fence.group(1)) >= fence_length:
                fence_character = None
                fence_length = 0
    flush_plain()
    if failed:
        warnings.append(f"{failed} formula(s) could not be converted to MathML and were preserved as LaTeX source.")
    return "".join(rendered_parts), replacements


def restore_math_html(rendered: str, replacements: list[tuple[str, str, bool]]) -> str:
    for token, replacement, display in replacements:
        if display:
            rendered = re.sub(rf"<p>\s*{re.escape(token)}\s*</p>", lambda _: replacement, rendered)
        rendered = rendered.replace(token, replacement)
    return rendered


def render_markdown(source: str, warnings: list[str]) -> str:
    prepared, math_replacements = prepare_math_markdown(source, warnings)
    try:
        import mistune  # type: ignore

        try:
            renderer = mistune.create_markdown(escape=True, hard_wrap=False, plugins=MARKDOWN_PLUGINS)
        except Exception:
            renderer = mistune.create_markdown(escape=True, hard_wrap=False, plugins=["table", "strikethrough"])
        return restore_math_html(str(renderer(prepared)), math_replacements)
    except ImportError:
        pass

    try:
        from markdown_it import MarkdownIt  # type: ignore

        renderer = MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": False})
        try:
            renderer.enable("table")
        except Exception:
            pass
        return restore_math_html(renderer.render(prepared), math_replacements)
    except ImportError:
        warnings.append("Neither mistune nor markdown-it-py was available; used the built-in basic Markdown renderer.")
        return restore_math_html(fallback_markdown(prepared), math_replacements)


def looks_like_markdown(source: str) -> bool:
    patterns = (
        r"(?m)^\s{0,3}#{1,6}\s+",
        r"(?m)^\s*[-+*]\s+",
        r"(?m)^\s*\d+[.)]\s+",
        r"```|~~~",
        r"\[[^\]]+\]\([^)]+\)",
        r"\*\*[^*]+\*\*|__[^_]+__",
        r"(?m)^\s*>\s+",
        r"(?m)^\s*\|.+\|\s*$",
    )
    return any(re.search(pattern, source) for pattern in patterns)


def is_strong_plain_heading(value: str) -> bool:
    value = value.strip()
    if not value or len(value) > 60:
        return False
    return bool(
        re.match(r"^(?:第[零〇一二三四五六七八九十百千万两\d]+[章节卷篇部]|序章|前言|序言|后记|附录(?:\s*[A-Z一二三四五六七八九十\d]+)?)(?:\s|$|[:：])", value, re.I)
        or re.match(r"^(?:chapter|part|book|appendix)\s+[\divxlcdm]+(?:\s|$|[:.-])", value, re.I)
    )


def plain_text_to_html(source: str) -> str:
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    output: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            output.append(f"<p>{html.escape(' '.join(part.strip() for part in paragraph if part.strip()))}</p>")
            paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
        elif is_strong_plain_heading(stripped):
            flush()
            output.append(f"<h2>{html.escape(stripped)}</h2>")
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(output)


def clean_html_fragment(source: str, warnings: list[str]) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(source, "html.parser")
        for tag in soup.find_all(["script", "style", "iframe", "object", "embed", "form", "meta", "link"]):
            tag.decompose()
        for tag in soup.find_all(True):
            for attribute in list(tag.attrs):
                if attribute.lower().startswith("on") or attribute.lower() in {"srcdoc"}:
                    del tag.attrs[attribute]
        container = soup.body if soup.body else soup
        return "".join(str(child) for child in container.contents)
    except ImportError:
        warnings.append("BeautifulSoup was unavailable; HTML sanitization used a limited fallback.")
        cleaned = re.sub(r"<(script|style|iframe|object|embed|form)\b[^>]*>.*?</\1>", "", source, flags=re.I | re.S)
        body = re.search(r"<body\b[^>]*>(.*?)</body>", cleaned, flags=re.I | re.S)
        cleaned = body.group(1) if body else cleaned
        return re.sub(r"\s+on\w+\s*=\s*([\"']).*?\1", "", cleaned, flags=re.I | re.S)


def local_image_to_data_uri(reference: str, base_dir: Path) -> str | None:
    parsed = urlparse(html.unescape(reference))
    if parsed.scheme in {"http", "https", "data", "mailto", "javascript"} or reference.startswith("#"):
        return None
    if parsed.scheme == "file":
        candidate = Path(unquote(parsed.path.lstrip("/")))
    else:
        raw = unquote(parsed.path)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
    try:
        candidate = candidate.resolve()
        if not candidate.is_file():
            return None
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except OSError:
        return None


def embed_local_images(fragment: str, base_dir: Path, warnings: list[str]) -> str:
    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        quote = match.group(1)
        reference = match.group(2)
        data_uri = local_image_to_data_uri(reference, base_dir)
        if data_uri:
            return f"src={quote}{data_uri}{quote}"
        parsed = urlparse(html.unescape(reference))
        if not parsed.scheme and not reference.startswith(("#", "data:")):
            missing.add(reference)
        return match.group(0)

    result = re.sub(r"\bsrc\s*=\s*([\"'])(.*?)\1", replace, fragment, flags=re.I | re.S)
    if missing:
        preview = ", ".join(sorted(missing)[:4])
        warnings.append(f"Some referenced local images were not found and were left unchanged: {preview}")
    return result


def extract_docx(path: Path, warnings: list[str]) -> str:
    word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ns = {"w": word_ns}

    def attr(element: ET.Element | None, name: str) -> str | None:
        if element is None:
            return None
        return element.attrib.get(f"{{{word_ns}}}{name}")

    try:
        with zipfile.ZipFile(path) as archive:
            document = ET.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise BuildError(f"DOCX could not be read: {exc}") from exc

    output: list[str] = []
    body = document.find("w:body", ns)
    if body is None:
        raise BuildError("DOCX has no document body.")

    for child in body:
        local = child.tag.rsplit("}", 1)[-1]
        if local == "p":
            style_node = child.find("w:pPr/w:pStyle", ns)
            style = attr(style_node, "val") or ""
            segments: list[str] = []
            for run in child.findall(".//w:r", ns):
                text_parts: list[str] = []
                for node in run:
                    node_local = node.tag.rsplit("}", 1)[-1]
                    if node_local == "t":
                        text_parts.append(node.text or "")
                    elif node_local == "tab":
                        text_parts.append("\t")
                    elif node_local in {"br", "cr"}:
                        text_parts.append("\n")
                value = html.escape("".join(text_parts))
                if not value:
                    continue
                properties = run.find("w:rPr", ns)
                if properties is not None and properties.find("w:b", ns) is not None:
                    value = f"<strong>{value}</strong>"
                if properties is not None and properties.find("w:i", ns) is not None:
                    value = f"<em>{value}</em>"
                segments.append(value)
            content = "".join(segments).replace("\n", "<br>")
            if not content:
                continue
            heading = re.search(r"(?:Heading|标题|標題)\s*([1-6])", style, re.I)
            if heading:
                level = int(heading.group(1))
                output.append(f"<h{level}>{content}</h{level}>")
            else:
                output.append(f"<p>{content}</p>")
        elif local == "tbl":
            rows: list[str] = []
            for row in child.findall(".//w:tr", ns):
                cells: list[str] = []
                for cell in row.findall("w:tc", ns):
                    text_value = " ".join((node.text or "") for node in cell.findall(".//w:t", ns)).strip()
                    cells.append(f"<td>{html.escape(text_value)}</td>")
                rows.append(f"<tr>{''.join(cells)}</tr>")
            if rows:
                output.append(f"<table><tbody>{''.join(rows)}</tbody></table>")

    warnings.append("DOCX text, headings, emphasis, and simple tables were extracted; floating objects and complex page layout may not be preserved.")
    return "\n".join(output)


def extract_pdf(path: Path, warnings: list[str]) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise BuildError("PDF extraction requires pypdf. Use the PDF skill/OCR first, or install pypdf, then retry.") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise BuildError(f"PDF could not be opened: {exc}") from exc

    sections: list[str] = []
    extracted_chars = 0
    blank_pages = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text_value = page.extract_text() or ""
        except Exception:
            raise BuildError(f"Text extraction failed on PDF page {page_number}; no page was silently omitted.")
        extracted_chars += len(text_value.strip())
        if not text_value.strip():
            blank_pages += 1
            continue
        sections.append(f'<section class="source-page" data-source-page="{page_number}">{plain_text_to_html(text_value)}</section>')

    if not sections:
        raise BuildError("No usable text was extracted from the PDF. It is probably scanned or uses unsupported fonts; OCR is required.")
    if blank_pages:
        warnings.append(f"PDF extraction returned no text for {blank_pages} of {len(reader.pages)} pages.")
    warnings.append("PDF was reflowed from extracted text; original columns, formulas, footnotes, and page geometry are not guaranteed.")
    if extracted_chars < max(200, len(reader.pages) * 40):
        warnings.append("The PDF yielded unusually little text; inspect the result for OCR or font-extraction errors.")
    return "\n".join(sections)


def xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_zip_text(archive: zipfile.ZipFile, name: str) -> str:
    raw = archive.read(name)
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def epub_resource_data_uri(archive: zipfile.ZipFile, chapter_path: str, reference: str) -> str | None:
    parsed = urlparse(html.unescape(reference))
    if parsed.scheme or reference.startswith(("#", "data:")):
        return None
    resource = posixpath.normpath(posixpath.join(posixpath.dirname(chapter_path), unquote(parsed.path)))
    try:
        raw = archive.read(resource)
    except KeyError:
        return None
    mime = mimetypes.guess_type(resource)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def extract_epub(path: Path, warnings: list[str]) -> str:
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise BuildError("EPUB is not a valid ZIP container.") from exc

    with archive:
        try:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next(element for element in container.iter() if xml_local_name(element.tag) == "rootfile")
            opf_path = rootfile.attrib["full-path"]
            opf = ET.fromstring(archive.read(opf_path))
        except (KeyError, StopIteration, ET.ParseError) as exc:
            raise BuildError(f"EPUB package metadata could not be read: {exc}") from exc

        opf_dir = posixpath.dirname(opf_path)
        manifest: dict[str, tuple[str, str]] = {}
        for element in opf.iter():
            if xml_local_name(element.tag) == "item" and element.attrib.get("id") and element.attrib.get("href"):
                item_path = posixpath.normpath(posixpath.join(opf_dir, element.attrib["href"]))
                manifest[element.attrib["id"]] = (item_path, element.attrib.get("media-type", ""))

        spine_ids = [element.attrib.get("idref", "") for element in opf.iter() if xml_local_name(element.tag) == "itemref"]
        chapter_paths = [manifest[item_id][0] for item_id in spine_ids if item_id in manifest]
        if not chapter_paths:
            chapter_paths = [item_path for item_path, media in manifest.values() if media in {"application/xhtml+xml", "text/html"}]
        if not chapter_paths:
            raise BuildError("EPUB contains no readable XHTML spine items.")

        chapters: list[str] = []
        missing_images = 0
        for chapter_index, chapter_path in enumerate(chapter_paths, start=1):
            try:
                source = read_zip_text(archive, chapter_path)
            except KeyError:
                continue
            fragment = clean_html_fragment(source, warnings=[])

            def replace_image(match: re.Match[str]) -> str:
                nonlocal missing_images
                quote = match.group(1)
                reference = match.group(2)
                data_uri = epub_resource_data_uri(archive, chapter_path, reference)
                if data_uri:
                    return f"src={quote}{data_uri}{quote}"
                if not urlparse(reference).scheme and not reference.startswith(("#", "data:")):
                    missing_images += 1
                return match.group(0)

            fragment = re.sub(r"\bsrc\s*=\s*([\"'])(.*?)\1", replace_image, fragment, flags=re.I | re.S)
            chapters.append(f'<section class="epub-chapter" data-chapter="{chapter_index}">{fragment}</section>')

        if not chapters:
            raise BuildError("EPUB spine items were present but none could be extracted.")
        if missing_images:
            warnings.append(f"{missing_images} EPUB image reference(s) could not be embedded.")
        warnings.append("EPUB text and inline images were reflowed; publisher-specific CSS and scripts were intentionally omitted.")
        return "\n".join(chapters)


def find_calibre_executable(name: str) -> Path | None:
    executable = name if name.lower().endswith(".exe") else f"{name}.exe"
    candidates: list[Path] = []
    if os.name == "nt" and PORTABLE_CALIBRE_ROOT is not None:
        portable_launchers = {
            "calibre": PORTABLE_CALIBRE_ROOT / "calibre-portable.exe",
            "ebook-viewer": PORTABLE_CALIBRE_ROOT / "ebook-viewer-portable.exe",
            "ebook-edit": PORTABLE_CALIBRE_ROOT / "ebook-edit-portable.exe",
        }
        portable_name = Path(executable).stem.lower()
        if portable_name in portable_launchers:
            candidates.append(portable_launchers[portable_name])
        candidates.append(PORTABLE_CALIBRE_ROOT / "Calibre" / executable)
    command = shutil.which(name)
    if command:
        candidates.append(Path(command))
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            candidates.extend([Path(base) / "Calibre2" / executable, Path(base) / "calibre" / executable])
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def find_calibre_library(explicit: Path | None) -> Path:
    configured = os.environ.get("COMFORTABLE_READER_CALIBRE_LIBRARY")
    candidate = explicit or (Path(configured) if configured else None)
    if candidate is None:
        raise BuildError("Choose the existing personal Calibre library explicitly with --library PATH; no machine path is assumed.")
    resolved = candidate.expanduser().resolve()
    if not (resolved / "metadata.db").is_file():
        raise BuildError("The selected Calibre library has no metadata.db; no new library was created or modified.")
    return resolved


def run_calibredb(arguments: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    executable = find_calibre_executable("calibredb")
    if not executable:
        raise BuildError("calibredb.exe was not found; the EPUB was not added to a persistent library.")
    return subprocess.run(
        [str(executable), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def search_library_identifier(library_path: Path, identifier_value: str) -> list[int]:
    result = run_calibredb(
        ["search", "--with-library", str(library_path), f"identifiers:comfortable_reader:{identifier_value}"]
    )
    if result.returncode not in {0, 1}:
        detail = (result.stderr or result.stdout or "unknown calibre database error").strip().splitlines()[-1]
        raise BuildError(f"Could not search the calibre library: {detail}")
    return [int(value) for value in re.findall(r"\d+", result.stdout)]


def add_to_calibre_library(
    epub_path: Path,
    library_path: Path,
    title: str,
    author: str,
    language: str,
    library_key: str,
) -> dict[str, object]:
    if not (library_path / "metadata.db").is_file():
        raise BuildError(f"Refusing to create a new or mistyped library. Verify the reader's existing Calibre library: {library_path}")
    existing_ids = search_library_identifier(library_path, library_key)
    if existing_ids:
        return {
            "path": str(library_path),
            "status": "already_present",
            "book_ids": existing_ids,
            "identifier": f"comfortable_reader:{library_key}",
        }

    result = run_calibredb(
        [
            "add",
            "--with-library",
            str(library_path),
            "--title",
            title,
            "--authors",
            author,
            "--languages",
            language,
            "--tags",
            "舒适阅读",
            "--identifier",
            f"comfortable_reader:{library_key}",
            str(epub_path),
        ],
        timeout=300,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown calibre database error").strip().splitlines()[-1]
        raise BuildError(f"Could not add the EPUB to the calibre library: {detail}")
    added_ids = search_library_identifier(library_path, library_key)
    if not added_ids:
        raise BuildError("calibre reported success, but the imported book could not be found by its identifier.")
    return {
        "path": str(library_path),
        "status": "added",
        "book_ids": added_ids,
        "identifier": f"comfortable_reader:{library_key}",
    }


def open_calibre_library(library_path: Path) -> str:
    launcher = find_calibre_executable("calibre")
    if not launcher:
        raise BuildError("calibre was requested, but its desktop launcher was not found.")
    subprocess.Popen([str(launcher), "--with-library", str(library_path)])
    return str(launcher)


def convert_unknown_with_calibre(path: Path, warnings: list[str]) -> str:
    converter = find_calibre_executable("ebook-convert")
    if not converter:
        raise BuildError(f"Unsupported input format {path.suffix or '(none)'} and calibre ebook-convert was not found.")
    with tempfile.TemporaryDirectory(prefix="comfortable-reader-") as temp_dir:
        epub_path = Path(temp_dir) / "converted.epub"
        result = subprocess.run(
            [str(converter), str(path), str(epub_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        if result.returncode != 0 or not epub_path.is_file():
            detail = (result.stderr or result.stdout or "unknown calibre error").strip().splitlines()[-1]
            raise BuildError(f"calibre could not convert {path.name}: {detail}")
        warnings.append(f"Converted {path.suffix} to temporary EPUB with calibre before reflowing it.")
        return extract_epub(epub_path, warnings)


def load_source(path: Path, requested_format: str, warnings: list[str]) -> tuple[str, str, str]:
    suffix = path.suffix.lower()
    if requested_format == "markdown":
        source = read_text(path)
        return render_markdown(source, warnings), "markdown", source
    if requested_format == "text":
        source = read_text(path)
        return plain_text_to_html(source), "text", source
    if requested_format == "html":
        source = read_text(path)
        return clean_html_fragment(source, warnings), "html", source

    if suffix in {".md", ".markdown"}:
        source = read_text(path)
        return render_markdown(source, warnings), "markdown", source
    if suffix == ".txt":
        source = read_text(path)
        if looks_like_markdown(source):
            return render_markdown(source, warnings), "markdown", source
        return plain_text_to_html(source), "text", source
    if suffix in {".html", ".htm"}:
        source = read_text(path)
        return clean_html_fragment(source, warnings), "html", source
    if suffix == ".docx":
        return extract_docx(path, warnings), "docx", path.name
    if suffix == ".epub":
        return extract_epub(path, warnings), "epub", path.name
    if suffix == ".pdf":
        return extract_pdf(path, warnings), "pdf", path.name
    return convert_unknown_with_calibre(path, warnings), "calibre-converted", path.name


def infer_title(fragment: str, path: Path, override: str | None) -> str:
    if override and override.strip():
        return override.strip()
    heading = re.search(r"<h1\b[^>]*>(.*?)</h1>", fragment, flags=re.I | re.S)
    if heading:
        value = strip_tags(heading.group(1))
        if value:
            return value[:160]
    return path.stem or "舒适阅读"


def slugify(value: str, fallback: str) -> str:
    value = re.sub(r"[^\w\s-]", "", strip_tags(value).lower(), flags=re.UNICODE)
    value = re.sub(r"[\s_]+", "-", value).strip("-")
    return value or fallback


def add_heading_ids(fragment: str) -> tuple[str, list[tuple[int, str, str]]]:
    toc: list[tuple[int, str, str]] = []
    used: set[str] = set()
    counter = 0

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(fragment, "html.parser")
        for heading in soup.find_all(re.compile(r"^h[1-6]$", re.I)):
            counter += 1
            title = heading.get_text(" ", strip=True) or f"Section {counter}"
            identifier = heading.get("id") or slugify(title, f"section-{counter}")
            base = identifier
            suffix = 2
            while identifier in used:
                identifier = f"{base}-{suffix}"
                suffix += 1
            used.add(identifier)
            heading["id"] = identifier
            toc.append((int(heading.name[1]), identifier, title))
        rendered = "".join(str(child) for child in soup.contents)
        return rendered, toc
    except ImportError:
        pass

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        level = int(match.group(1))
        attrs = match.group(2) or ""
        body = match.group(3)
        title = strip_tags(body) or f"Section {counter}"
        id_match = re.search(r"\bid\s*=\s*([\"'])(.*?)\1", attrs, flags=re.I | re.S)
        identifier = id_match.group(2) if id_match else slugify(title, f"section-{counter}")
        base = identifier
        suffix = 2
        while identifier in used:
            identifier = f"{base}-{suffix}"
            suffix += 1
        used.add(identifier)
        if not id_match:
            attrs += f' id="{html.escape(identifier, quote=True)}"'
        toc.append((level, identifier, title))
        return f"<h{level}{attrs}>{body}</h{level}>"

    return re.sub(r"<h([1-6])(\s[^>]*)?>(.*?)</h\1>", replace, fragment, flags=re.I | re.S), toc


def xhtml_fragment(fragment: str) -> str:
    fragment = re.sub(
        r"<input\b[^>]*\bchecked(?:=[^\s>]*)?[^>]*>",
        '<span class="task-box">☑</span>',
        fragment,
        flags=re.I,
    )
    fragment = re.sub(r"<input\b[^>]*>", '<span class="task-box">☐</span>', fragment, flags=re.I)
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(fragment, "html.parser")
        return "".join(child.decode(formatter="minimal") if hasattr(child, "decode") else str(child) for child in soup.contents)
    except ImportError:
        fragment = re.sub(r"<(br|hr|img)(\b[^>]*?)(?<!/)>\s*", r"<\1\2 />", fragment, flags=re.I)
        return fragment.replace("&nbsp;", "&#160;")


def make_epub(
    output_path: Path,
    title: str,
    author: str,
    fragment: str,
    document_key: str,
    original_source: str | None = None,
    source_archive_name: str | None = None,
    source_reference: str | None = None,
    *,
    original_source_bytes: bytes | None = None,
    max_section_chars: int = 0,
    navigation: list[dict] | None = None,
    authors: list[str] | None = None,
) -> dict[str, object]:
    fragment_with_ids, toc = add_heading_ids(fragment)
    fragment_xml = xhtml_fragment(fragment_with_ids)
    if navigation is not None:
        available = {node.get('id') for node in ET.fromstring('<root>'+fragment_xml+'</root>').iter() if node.get('id')}
        toc=[]
        for item in navigation:
            level, target, label = int(item['level']), item['target'], item['label']
            if not 1 <= level <= 6 or target not in available or not str(label).strip():
                raise BuildError('Navigation needs a valid hierarchy, readable target, and source label.')
            toc.append((level,target,str(label)))
    language = "zh-CN" if re.search(r"[\u3400-\u9fff]", strip_tags(fragment)) else "en"
    identifier = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, document_key)}"
    nav_tree, nav_stack = [], []
    for level, identifier_value, label in toc:
        node={'level':level,'id':identifier_value,'label':label,'children':[]}
        while nav_stack and nav_stack[-1]['level'] >= level:
            nav_stack.pop()
        (nav_stack[-1]['children'] if nav_stack else nav_tree).append(node)
        nav_stack.append(node)
    def render_nav(nodes):
        return ''.join('<li><a href="content.xhtml#'+html.escape(node['id'],quote=True)+'">'+html.escape(node['label'])+'</a>'+
                       ('<ol>'+render_nav(node['children'])+'</ol>' if node['children'] else '')+'</li>' for node in nodes)
    nav_items = render_nav(nav_tree) or '<li><a href="content.xhtml">开始阅读</a></li>'



    stylesheet = (SKILL_DIR / 'assets' / 'epub-reading.css').read_text(encoding='utf-8')

    content_xhtml = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{language}" lang="{language}">
<head><meta charset="utf-8"/><title>{html.escape(title)}</title><link rel="stylesheet" type="text/css" href="style.css"/></head>
<body>{fragment_xml}</body>
</html>'''

    nav_xhtml = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{language}" lang="{language}">
<head><meta charset="utf-8"/><title>目录</title></head>
<body><nav epub:type="toc" id="toc"><h1>目录</h1><ol>{nav_items}</ol></nav></body>
</html>'''

    container_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>'''

    source_payload = original_source_bytes if original_source_bytes is not None else (original_source.encode("utf-8") if original_source is not None else None)
    source_name = source_archive_name if source_payload is not None and source_archive_name else None
    if source_name and (Path(source_name).name != source_name or "/" in source_name or "\\" in source_name):
        raise BuildError("Embedded source name must be a filename, not a path.")
    source_media_types = {".md": "text/markdown", ".markdown": "text/markdown", ".txt": "text/plain", ".html": "text/html", ".htm": "text/html", ".pdf": "application/pdf", ".epub": "application/epub+zip", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    source_manifest = ""
    if source_name:
        source_media_type = source_media_types.get(Path(source_name).suffix.lower(), "text/plain")
        source_manifest = f'    <item id="original-source" href="original/{html.escape(source_name, quote=True)}" media-type="{source_media_type}"/>\n'
    source_metadata = f'    <dc:source>{html.escape(source_reference)}</dc:source>\n' if source_reference else ""
    content_properties = ' properties="mathml"' if "<math " in fragment_xml else ""

    creators = authors or [author]
    creator_metadata = ''.join('    <dc:creator>'+html.escape(name)+'</dc:creator>\n' for name in creators)
    content_opf = f'''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id" xml:lang="{language}">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">{identifier}</dc:identifier>
    <dc:title>{html.escape(title)}</dc:title>
{creator_metadata.rstrip()}
    <dc:language>{language}</dc:language>
{source_metadata}    <meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="content" href="content.xhtml" media-type="application/xhtml+xml"{content_properties}/>
    <item id="style" href="style.css" media-type="text/css"/>
{source_manifest}  </manifest>
  <spine><itemref idref="content"/></spine>
</package>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container_xml, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OEBPS/content.opf", content_opf, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OEBPS/content.xhtml", content_xhtml, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OEBPS/nav.xhtml", nav_xhtml, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OEBPS/style.css", stylesheet, compress_type=zipfile.ZIP_DEFLATED)
        if source_name and source_payload is not None:
            archive.writestr(f"OEBPS/original/{source_name}", source_payload, compress_type=zipfile.ZIP_DEFLATED)

    math_box_count = fragment_xml.count(MATH_BOX_CLASS)
    validate_epub(output_path, original_source, source_name, math_box_count, original_source_bytes=source_payload)
    segmentation: dict[str, object] = {"spine_sections": 1, "oversized_sections": []}
    if max_section_chars:
        from epub_tools import EpubAuditError, segment_generated_epub
        try:
            segmentation = segment_generated_epub(output_path, max_section_chars)
        except EpubAuditError as exc:
            raise BuildError(str(exc)) from exc
        # Segmentation rewrites the OPF/spine and XHTML links. The pre-split
        # validation above is insufficient; audit the bytes actually returned.
        from epub_tools import audit_epub
        final_audit = audit_epub(output_path)
        if final_audit["errors"] or final_audit["raw_math_fallbacks"] or final_audit["external_render_resources"]:
            raise BuildError("Segmented EPUB failed its final structural audit: " + "; ".join(final_audit["errors"]))
        if original_source_bytes is not None and source_archive_name:
            with zipfile.ZipFile(output_path) as archive:
                if archive.read(f"OEBPS/original/{source_archive_name}") != original_source_bytes:
                    raise BuildError("Segmentation changed the embedded original source bytes.")
    return {
        "path": str(output_path.resolve()),
        "toc_entries": len(toc),
        "bytes": output_path.stat().st_size,
        "mathml_formulas": fragment_xml.count("<math "),
        "boxed_math_regions": math_box_count,
        "source_embedded": bool(source_name),
        "source_sha256": hashlib.sha256(source_payload).hexdigest() if source_payload is not None else None,
        "source_entry": f"OEBPS/original/{source_name}" if source_name else None,
        **segmentation,
    }


def validate_epub(
    path: Path,
    original_source: str | None = None,
    source_archive_name: str | None = None,
    expected_math_boxes: int | None = None,
    *,
    original_source_bytes: bytes | None = None,
) -> None:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or entries[0].filename != "mimetype" or entries[0].compress_type != zipfile.ZIP_STORED:
            raise BuildError("Generated EPUB has an invalid mimetype entry.")
        if archive.read("mimetype") != b"application/epub+zip":
            raise BuildError("Generated EPUB has the wrong mimetype.")
        if archive.testzip() is not None:
            raise BuildError("Generated EPUB failed its ZIP integrity check.")
        for name in ("META-INF/container.xml", "OEBPS/content.opf", "OEBPS/content.xhtml", "OEBPS/nav.xhtml"):
            try:
                ET.fromstring(archive.read(name))
            except (KeyError, ET.ParseError) as exc:
                raise BuildError(f"Generated EPUB XML is invalid in {name}: {exc}") from exc
        if expected_math_boxes is not None:
            content = archive.read("OEBPS/content.xhtml").decode("utf-8")
            stylesheet = archive.read("OEBPS/style.css").decode("utf-8")
            if content.count(MATH_BOX_CLASS) != expected_math_boxes:
                raise BuildError("The EPUB did not preserve every rendered boxed-math region.")
            if expected_math_boxes and "menclose[notation~=" not in stylesheet:
                raise BuildError("The EPUB is missing the WebView2-compatible boxed-math CSS fallback.")
        source_payload = original_source_bytes if original_source_bytes is not None else (original_source.encode("utf-8") if original_source is not None else None)
        if source_payload is not None and source_archive_name:
            embedded = archive.read(f"OEBPS/original/{source_archive_name}")
            if embedded != source_payload:
                raise BuildError("The EPUB's embedded source is not byte-for-byte identical to the input file.")


def safe_json_for_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def make_html(output_path: Path, title: str, fragment: str, columns: str, theme: str, document_key: str) -> dict[str, object]:
    if not TEMPLATE_PATH.is_file():
        raise BuildError(f"Reader template is missing: {TEMPLATE_PATH}")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    payload = base64.b64encode(fragment.encode("utf-8")).decode("ascii")
    config = safe_json_for_script({"title": title, "columns": columns, "theme": theme, "documentKey": document_key})
    rendered = template.replace("@@TITLE_HTML@@", html.escape(title)).replace("@@CONTENT_B64@@", payload).replace("@@CONFIG_JSON@@", config)
    if "@@" in rendered:
        unresolved = sorted(set(re.findall(r"@@[A-Z0-9_]+@@", rendered)))
        if unresolved:
            raise BuildError(f"Reader template has unresolved placeholders: {', '.join(unresolved)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8", newline="\n")
    validate_html(output_path, columns, theme)
    return {"path": str(output_path.resolve()), "bytes": output_path.stat().st_size}


def validate_html(path: Path, columns: str, theme: str) -> None:
    source = path.read_text(encoding="utf-8")
    required = ('id="flow"', 'id="viewport"', "const CONTENT_B64", 'id="columnSelect"', 'id="tocPanel"')
    missing = [token for token in required if token not in source]
    if missing:
        raise BuildError(f"Generated HTML failed structural validation: missing {', '.join(missing)}")
    if f'"columns":"{columns}"' not in source or f'"theme":"{theme}"' not in source:
        raise BuildError("Generated HTML does not contain the requested initial settings.")
    if path.stat().st_size < 10_000:
        raise BuildError("Generated HTML is unexpectedly small.")


def open_artifact(reader: str, html_path: Path | None, epub_path: Path | None) -> tuple[str, str]:
    viewer = find_calibre_executable("ebook-viewer")
    selected = reader
    if reader == "auto":
        selected = "calibre" if viewer and epub_path else "html"
    if selected == "calibre":
        if not viewer:
            raise BuildError("calibre E-book Viewer was requested but ebook-viewer.exe was not found.")
        if not epub_path or not epub_path.is_file():
            raise BuildError("calibre E-book Viewer was requested but no EPUB was generated.")
        subprocess.Popen([str(viewer), str(epub_path)])
        return "calibre", str(viewer)
    if not html_path or not html_path.is_file():
        raise BuildError("The HTML reader was requested, but no HTML fallback was generated.")
    if os.name == "nt":
        os.startfile(str(html_path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(html_path.resolve().as_uri())
    return "html", "default browser"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source document path")
    parser.add_argument("--output", type=Path, help="Standalone HTML output path")
    parser.add_argument("--no-html", action="store_true", help="Do not create the browser-based HTML fallback")
    parser.add_argument("--epub", type=Path, help="EPUB output path (default: beside HTML)")
    parser.add_argument("--no-epub", action="store_true", help="Do not create an EPUB companion")
    parser.add_argument("--format", choices=("auto", "markdown", "text", "html"), default="auto")
    parser.add_argument("--columns", choices=("auto", "1", "2", "3", "4"), default="auto")
    parser.add_argument("--theme", choices=("auto", "paper", "sepia", "dark"), default="auto")
    parser.add_argument("--title", help="Override the inferred title")
    parser.add_argument("--author", default="未知", help="Author stored in the EPUB and calibre library")
    parser.add_argument("--source-reference", help="Original conversation, document, or page reference stored in EPUB metadata")
    parser.add_argument("--max-section-chars", type=int, default=0, help="Optional visible-character budget per spine section; never splits an authored block")
    parser.add_argument("--allow-math-fallback", action="store_true", help="Explicitly allow visible unrendered math in a draft; never permitted with library import")
    parser.add_argument("--add-to-library", action="store_true", help="Add the EPUB to a persistent calibre library")
    parser.add_argument("--library", type=Path, help="calibre library path; auto-detected when omitted")
    parser.add_argument("--open-library", action="store_true", help="Open the calibre desktop library after import")
    parser.add_argument("--open", action="store_true", dest="open_after", help="Open after successful generation")
    parser.add_argument("--reader", choices=("auto", "html", "calibre"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args(argv or sys.argv[1:])
    input_path = args.input.expanduser().resolve()
    warnings: list[str] = []
    try:
        if not input_path.is_file():
            raise BuildError(f"Input file does not exist: {input_path}")
        if args.max_section_chars < 0:
            raise BuildError("--max-section-chars must be non-negative.")
        if args.allow_math_fallback and args.add_to_library:
            raise BuildError("Visible math fallback is a draft only. Resolve it before importing.")
        if input_path.suffix.lower() in {".pdf", ".epub"} and args.add_to_library:
            raise BuildError("PDF extraction and EPUB flattening are diagnostic, not fidelity approval. Build without --add-to-library, follow the source-specific reference, then import the exact reviewed EPUB with scripts/persist_epub.py.")

        if args.no_html and args.output:
            raise BuildError("Use either --output or --no-html, not both.")
        if args.no_html and args.no_epub:
            raise BuildError("At least one artifact is required; do not combine --no-html and --no-epub.")
        output_path = None if args.no_html else (args.output or input_path.with_name(f"{input_path.stem}.reader.html")).expanduser().resolve()
        if output_path == input_path:
            raise BuildError("Output path must differ from the input path.")
        if args.no_epub and args.epub:
            raise BuildError("Use either --epub or --no-epub, not both.")
        if args.no_epub:
            epub_path = None
        elif args.epub:
            epub_path = args.epub.expanduser().resolve()
        elif output_path:
            epub_path = output_path.with_suffix(".epub")
        else:
            epub_path = input_path.with_name(f"{input_path.stem}.reader.epub").resolve()
        if epub_path and epub_path in {input_path, output_path}:
            raise BuildError("EPUB output path must differ from input and HTML output paths.")
        if args.add_to_library and not epub_path:
            raise BuildError("--add-to-library requires an EPUB; remove --no-epub.")
        if args.source_reference and not epub_path:
            raise BuildError("--source-reference requires an EPUB; remove --no-epub.")
        if args.library and not (args.add_to_library or args.open_library):
            raise BuildError("--library is only meaningful with --add-to-library or --open-library.")

        fragment, source_kind, raw_identity = load_source(input_path, args.format, warnings)
        fragment = embed_local_images(fragment, input_path.parent, warnings)
        if re.search(r'class=[\"\'][^\"\']*\bmath-source\b', fragment) and not args.allow_math_fallback:
            raise BuildError("Unrendered mathematics remains. Repair the formula/source; --allow-math-fallback is available only for an explicitly accepted diagnostic draft.")
        if not strip_tags(fragment):
            raise BuildError("The source produced no readable text.")
        title = infer_title(fragment, input_path, args.title)
        document_key = hashlib.sha256((str(input_path) + "\n" + raw_identity + "\n" + fragment).encode("utf-8", errors="replace")).hexdigest()[:20]
        library_key = hashlib.sha256((title + "\n" + fragment).encode("utf-8", errors="replace")).hexdigest()[:24]
        language = "zh" if re.search(r"[\u3400-\u9fff]", strip_tags(fragment)) else "en"
        original_source = raw_identity if source_kind in {"markdown", "text", "html"} else None
        source_archive_names = {"markdown": "source.md", "text": "source.txt", "html": "source.html"}
        source_archive_name = source_archive_names.get(source_kind)
        if source_archive_name is None:
            source_archive_name = "source" + input_path.suffix.lower()
        input_bytes = input_path.read_bytes()
        fidelity_info = None
        if original_source is not None:
            fidelity_info = {
                "source_sha256": hashlib.sha256(input_bytes).hexdigest(),
                "source_characters": len(original_source),
                "replacement_characters": original_source.count("\ufffd"),
                "markdown_headings": len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+", original_source)),
                "code_fences": len(re.findall(r"(?m)^\s*(?:```+|~~~+)", original_source)),
                "block_math_regions": fragment.count('class="math-block"'),
                "inline_math_regions": fragment.count('class="math-inline"'),
                "rendered_mathml_formulas": fragment.count("<math "),
                "boxed_math_regions": fragment.count(MATH_BOX_CLASS),
            }

        html_info = make_html(output_path, title, fragment, args.columns, args.theme, document_key) if output_path else None
        epub_info = (
            make_epub(
                epub_path,
                title,
                args.author,
                fragment,
                document_key,
                original_source,
                source_archive_name,
                args.source_reference,
                original_source_bytes=input_bytes,
                max_section_chars=args.max_section_chars,
            )
            if epub_path
            else None
        )
        library_info = None
        library_path = None
        if args.add_to_library or args.open_library:
            library_path = find_calibre_library(args.library)
        if args.add_to_library and epub_path and library_path:
            from epub_tools import EpubAuditError, audit_epub
            if warnings:
                raise BuildError("Conversion warnings need review before a library write. Review the draft and use scripts/persist_epub.py for the final EPUB.")
            try:
                audit = audit_epub(epub_path)
            except (EpubAuditError, KeyError, ET.ParseError) as exc:
                raise BuildError(str(exc)) from exc
            if audit["errors"] or audit["raw_math_fallbacks"] or audit["external_render_resources"]:
                raise BuildError("EPUB structural/resource audit failed; no library write was attempted. " + "; ".join(audit["errors"]))
            library_info = add_to_calibre_library(
                epub_path, library_path, title, args.author, language, library_key
            )
        opened = None
        if args.open_after:
            opened_reader, opened_with = open_artifact(args.reader, output_path, epub_path)
            opened = {"reader": opened_reader, "with": opened_with}
        library_opened_with = open_calibre_library(library_path) if args.open_library and library_path else None

        result = {
            "status": "ok",
            "title": title,
            "source": str(input_path),
            "source_kind": source_kind,
            "library_key": library_key,
            "source_reference": args.source_reference,
            "fidelity": fidelity_info,
            "initial_columns": args.columns,
            "initial_theme": args.theme,
            "html": html_info,
            "epub": epub_info,
            "library": library_info,
            "opened": opened,
            "library_opened_with": library_opened_with,
            "warnings": warnings,
            "delivery_status": "awaiting_desktop_verification" if library_info else "draft_not_imported",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (BuildError, OSError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "warnings": warnings}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
