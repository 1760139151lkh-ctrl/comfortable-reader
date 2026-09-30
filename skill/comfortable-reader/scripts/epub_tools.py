"""Offline structural audit and block-preserving segmentation for EPUBs."""
from __future__ import annotations

import copy
import hashlib
import posixpath
import re
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

from math_rendering import MATH_NS, MathRenderingError, validate_mathml

XHTML = "http://www.w3.org/1999/xhtml"
OPF = "http://www.idpf.org/2007/opf"
DC = "http://purl.org/dc/elements/1.1/"
CONTAINER = "urn:oasis:names:tc:opendocument:xmlns:container"


class EpubAuditError(ValueError):
    pass


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def resolve_resource(document: str, href: str) -> tuple[str, str]:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return href, ""
    path = posixpath.normpath(posixpath.join(posixpath.dirname(document), unquote(parsed.path))) if parsed.path else document
    if path.startswith(("../", "/")) or path == "..":
        raise EpubAuditError(f"Reference escapes the EPUB: {href}")
    return path, unquote(parsed.fragment)


def package_path(entries: dict[str, bytes]) -> str:
    root = ET.fromstring(entries["META-INF/container.xml"])
    item = root.find(".//{" + CONTAINER + "}rootfile")
    if item is None or not item.get("full-path"):
        raise EpubAuditError("EPUB container has no package document.")
    return item.attrib["full-path"]


def visible_text(root: ET.Element) -> str:
    if local(root.tag) in {"head", "script", "style", "annotation", "annotation-xml"}:
        return ""
    return (root.text or "") + "".join(visible_text(child) + (child.tail or "") for child in root)


def audit_epub(path: Path) -> dict[str, object]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as archive:
        info = archive.infolist()
        if not info or info[0].filename != "mimetype" or info[0].compress_type != zipfile.ZIP_STORED:
            errors.append("mimetype must be the first, uncompressed ZIP entry.")
        if archive.read("mimetype") != b"application/epub+zip":
            errors.append("Incorrect EPUB mimetype.")
        if len({item.filename for item in info}) != len(info):
            errors.append("Duplicate ZIP member names are ambiguous.")
        bad = archive.testzip()
        if bad:
            errors.append("ZIP checksum failed: " + bad)
        entries = {item.filename: archive.read(item.filename) for item in info}
    opf_path = package_path(entries)
    package = ET.fromstring(entries[opf_path])
    manifest = package.find("{" + OPF + "}manifest")
    spine = package.find("{" + OPF + "}spine")
    if manifest is None or spine is None:
        raise EpubAuditError("Package is missing its manifest or spine.")
    items: dict[str, tuple[str, str]] = {}
    xml_documents: dict[str, ET.Element] = {}
    remote_resources: list[str] = []
    nav_count = 0
    for item in manifest:
        ident, href, media = item.get("id", ""), item.get("href", ""), item.get("media-type", "")
        if not ident or ident in items:
            errors.append("Missing or duplicate manifest id: " + ident)
        resource, _ = resolve_resource(opf_path, href)
        items[ident] = (resource, media)
        nav_count += "nav" in item.get("properties", "").split()
        if urlsplit(resource).scheme:
            remote_resources.append(resource)
            continue
        if resource not in entries:
            errors.append("Missing manifest resource: " + resource)
            continue
        if media in {"application/xhtml+xml", "image/svg+xml", "application/x-dtbncx+xml"}:
            try:
                xml_documents[resource] = ET.fromstring(entries[resource])
            except ET.ParseError as exc:
                errors.append(f"Invalid XML in {resource}: {exc}")
    if package.get("version", "").startswith("3") and nav_count != 1:
        errors.append("EPUB 3 requires one navigation document.")
    linear = []
    for ref in spine:
        ident = ref.get("idref", "")
        if ident not in items:
            errors.append("Spine points to an unknown item: " + ident)
        elif ref.get("linear", "yes") != "no":
            linear.append(items[ident][0])
    if not linear:
        errors.append("No readable spine sections.")
    ids = {}
    for name, root in xml_documents.items():
        values = [el.get("id") for el in root.iter() if el.get("id")]
        if len(values) != len(set(values)):
            errors.append("Duplicate element ids: " + name)
        ids[name] = set(values)
    for name, root in xml_documents.items():
        for node in root.iter():
            for attr in ("href", "src", "data", "{" + "http://www.w3.org/1999/xlink" + "}href"):
                href = node.get(attr)
                if not href:
                    continue
                scheme = urlsplit(href).scheme
                if scheme or href.startswith("//"):
                    if scheme != "data" and local(node.tag) != "a":
                        remote_resources.append(href)
                    continue
                target, fragment = resolve_resource(name, href)
                if target not in entries:
                    errors.append(f"Broken resource in {name}: {href}")
                elif fragment and target in ids and fragment not in ids[target]:
                    errors.append(f"Broken fragment in {name}: {href}")
    for name, raw in entries.items():
        if not name.endswith(".css"):
            continue
        css = raw.decode("utf-8-sig")
        refs = re.findall(r"url\(\s*['\"]?([^)'\"]+)['\"]?\s*\)", css)
        refs += re.findall(r"@import\s+['\"]([^'\"]+)['\"]", css)
        for href in refs:
            if href.startswith("#") or urlsplit(href).scheme == "data":
                continue
            target, _ = resolve_resource(name, href)
            if urlsplit(target).scheme or href.startswith("//"):
                remote_resources.append(href)
            elif target not in entries:
                errors.append(f"Broken stylesheet resource in {name}: {href}")
    math_count = box_count = fallback_count = replacement_count = 0
    section_sizes = []
    for name in linear:
        root = xml_documents.get(name)
        if root is None:
            errors.append("Spine section is not parseable XHTML: " + name)
            continue
        text = visible_text(root)
        replacement_count += text.count("\ufffd")
        section_sizes.append(len(re.sub(r"\s+", " ", text)))
        for node in root.iter():
            classes = node.get("class", "").split()
            fallback_count += "math-source" in classes
            box_count += "comfortable-math-box" in classes
            if node.tag == "{" + MATH_NS + "}math":
                math_count += 1
                try:
                    validate_mathml(ET.tostring(node, encoding="unicode"))
                except MathRenderingError as exc:
                    errors.append(f"Invalid rendered math in {name}: {exc}")
    if replacement_count:
        errors.append("Visible Unicode replacement characters remain.")
    metadata = package.find("{" + OPF + "}metadata")
    def dc_text(field: str) -> list[str]:
        return ["".join(node.itertext()) for node in metadata.findall("{" + DC + "}" + field)] if metadata is not None else []
    return {"status": "error" if errors else "ok", "path": str(path.resolve()),
            "epub_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
            "content_sha256": hashlib.sha256(b"".join(name.encode("utf-8") + b"\0" + hashlib.sha256(data).digest() for name, data in sorted(entries.items()))).hexdigest(),
            "title": next(iter(dc_text("title")), ""), "authors": dc_text("creator"),
            "language": next(iter(dc_text("language")), ""), "identifiers": dc_text("identifier"),
            "spine_sections": len(linear), "section_characters": section_sizes,
            "mathml_formulas": math_count, "boxed_math_regions": box_count,
            "raw_math_fallbacks": fallback_count, "replacement_characters": replacement_count,
            "external_render_resources": sorted(set(remote_resources)), "errors": errors}


def serialize_xhtml(root: ET.Element) -> bytes:
    # Keep unprefixed <math xmlns=...>, including when a browser parses srcdoc
    # as HTML. A generic XML serializer may otherwise emit <ns1:math>.
    root = copy.deepcopy(root)
    math = []
    for parent in list(root.iter()):
        for index, child in enumerate(list(parent)):
            if child.tag != "{" + MATH_NS + "}math":
                continue
            marker = ET.Comment("COMFORTABLE-MATH-" + str(len(math)))
            marker.tail, child.tail = child.tail, None
            ET.register_namespace("", MATH_NS)
            math.append(ET.tostring(child, encoding="unicode"))
            parent.remove(child)
            parent.insert(index, marker)
    ET.register_namespace("", XHTML)
    text = ET.tostring(root, encoding="unicode")
    for index, markup in enumerate(math):
        text = text.replace("<!--COMFORTABLE-MATH-" + str(index) + "-->", markup)
    return ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n' + text).encode("utf-8")


def segment_generated_epub(path: Path, max_characters: int) -> dict[str, object]:
    """Split this builder's single XHTML spine; preserve indivisible top-level blocks."""
    if max_characters < 1:
        return {"spine_sections": 1, "oversized_sections": []}
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    opf_path = package_path(entries)
    package = ET.fromstring(entries[opf_path])
    manifest, spine = package.find("{" + OPF + "}manifest"), package.find("{" + OPF + "}spine")
    if manifest is None or spine is None or len(spine) != 1:
        raise EpubAuditError("Segmentation expects one generated spine section; preserve existing multi-section EPUBs.")
    item = next(el for el in manifest if el.get("id") == spine[0].get("idref"))
    source, _ = resolve_resource(opf_path, item.attrib["href"])
    document = ET.fromstring(entries[source])
    body = document.find("{" + XHTML + "}body")
    if body is None:
        raise EpubAuditError("No XHTML body to segment.")
    if (body.text or "").strip() or any((child.tail or "").strip() for child in body):
        raise EpubAuditError("Wrap loose top-level text in semantic paragraphs before segmentation.")
    chunks, current, size = [], [], 0
    for child in body:
        amount = len(visible_text(child))
        previous_heading = bool(current and re.fullmatch(r"h[1-6]", local(current[-1].tag)))
        chapter_break = local(child.tag) in {"h1", "h2"} and size > 500
        if current and not previous_heading and (chapter_break or size + amount > max_characters):
            chunks.append(current)
            current, size = [], 0
        current.append(child)
        size += amount
    if current:
        chunks.append(current)
    if len(chunks) < 2:
        return {"spine_sections": 1, "oversized_sections": [1] if size > max_characters else []}
    stem, suffix = posixpath.splitext(source)
    names = [source] + [f"{stem}-{i + 1:03d}{suffix}" for i in range(1, len(chunks))]
    targets = {el.get("id"): name for name, chunk in zip(names, chunks) for child in chunk for el in child.iter() if el.get("id")}
    def rewrite(root: ET.Element, original_name: str, current_name: str):
        for node in root.iter():
            for attr in ("href", "src", "{http://www.w3.org/1999/xlink}href"):
                href = node.get(attr)
                if not href or urlsplit(href).scheme:
                    continue
                target, fragment = resolve_resource(original_name, href)
                if target == source and fragment in targets:
                    target = targets[fragment]
                    node.set(attr, ("" if target == current_name else posixpath.relpath(target, posixpath.dirname(current_name))) + "#" + fragment)
    sizes = []
    for name, chunk in zip(names, chunks):
        part = copy.deepcopy(document)
        content = part.find("{" + XHTML + "}body")
        content.clear()
        content.attrib.update(body.attrib)
        for child in chunk:
            content.append(copy.deepcopy(child))
        rewrite(part, source, name)
        sizes.append(len(visible_text(content)))
        entries[name] = serialize_xhtml(part)
    for nav in manifest:
        if "nav" in nav.get("properties", "").split():
            name, _ = resolve_resource(opf_path, nav.attrib["href"])
            root = ET.fromstring(entries[name])
            rewrite(root, name, name)
            entries[name] = serialize_xhtml(root)
    position = list(manifest).index(item)
    manifest.remove(item)
    spine.clear()
    for index, name in enumerate(names):
        ident = "reader-section-" + str(index + 1)
        attrs = {"id": ident, "href": posixpath.relpath(name, posixpath.dirname(opf_path)), "media-type": "application/xhtml+xml"}
        if b"<math " in entries[name]:
            attrs["properties"] = "mathml"
        manifest.insert(position + index, ET.Element("{" + OPF + "}item", attrs))
        ET.SubElement(spine, "{" + OPF + "}itemref", {"idref": ident})
    ET.register_namespace("", OPF)
    ET.register_namespace("dc", DC)
    entries[opf_path] = ET.tostring(package, encoding="utf-8", xml_declaration=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".epub", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.writestr("mimetype", b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
            for name, data in entries.items():
                if name != "mimetype":
                    archive.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
        report = audit_epub(temporary)
        if report["errors"]:
            raise EpubAuditError("Segmented EPUB failed validation: " + "; ".join(report["errors"]))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"spine_sections": len(chunks), "section_characters": sizes,
            "oversized_sections": [i + 1 for i, n in enumerate(sizes) if n > max_characters]}
