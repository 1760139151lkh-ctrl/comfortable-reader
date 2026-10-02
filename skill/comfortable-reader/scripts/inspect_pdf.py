#!/usr/bin/env python3
"""Export an inspectable PDF source packet, without creating or importing a book."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

VERSION = 1


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=list), encoding="utf-8")


def page_selection(value: str, count: int) -> list[int]:
    selected = set()
    for part in value.split(","):
        if not part.strip():
            continue
        ends = [int(v) for v in part.split("-")]
        if len(ends) == 1:
            selected.add(ends[0])
        elif len(ends) == 2 and ends[0] <= ends[1]:
            selected.update(range(ends[0], ends[1] + 1))
        else:
            raise ValueError("Use page numbers/ranges such as 1,3,8-10.")
    if any(n < 1 or n > count for n in selected):
        raise ValueError(f"Requested preview page is outside 1–{count}.")
    return sorted(selected)


def inspect(source: Path, output: Path, render_pages: str = "1", refresh: bool = False) -> dict:
    try:
        import pymupdf as pdf
    except ImportError as exc:
        raise RuntimeError("This interpreter lacks PyMuPDF. Use an existing Python with pymupdf, or install pymupdf in the chosen environment; do not switch to lossy text-only import.") from exc
    raw = source.read_bytes()
    source_hash = digest(raw)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pages").mkdir(exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    cached = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    with pdf.open(source) as document:
        selected = page_selection(render_pages, len(document))
        if cached and not refresh and cached.get("source", {}).get("sha256") == source_hash and cached.get("tool_version") == VERSION:
            manifest = cached
        else:
            records, assets, fonts, errors = [], {}, {}, []
            for pi, page in enumerate(document, 1):
                units = []
                try:
                    raw_page = page.get_text("rawdict", flags=pdf.TEXT_PRESERVE_LIGATURES | pdf.TEXT_PRESERVE_WHITESPACE)
                    counter = 0
                    for bi, block in enumerate(raw_page["blocks"]):
                        for li, line in enumerate(block.get("lines", [])):
                            for span in line["spans"]:
                                counter += 1
                                text = "".join(char["c"] for char in span["chars"])
                                units.append({"id": f"p{pi:04d}.text{counter:05d}", "kind": "text", "pdf_block": bi, "pdf_line": li,
                                              **span, "text": text})
                    for font in page.get_fonts():
                        xref = font[0]
                        if str(xref) not in fonts:
                            obj = document.xref_object(xref) if xref else ""
                            match = re.search(r"/FontDescriptor (\d+) \d+ R", obj)
                            descriptor = document.xref_object(int(match.group(1))) if match else ""
                            fonts[str(xref)] = {"pdf_font": list(font), "descriptor": descriptor}
                    for ii, info in enumerate(page.get_image_info(xrefs=True), 1):
                        xref = info["xref"]
                        asset_id = f"image-{xref}" if xref else f"inline-{pi}-{ii}"
                        if asset_id not in assets:
                            if xref:
                                extracted = document.extract_image(xref)
                            else:
                                image_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 1]
                                matches = [b for b in image_blocks if b.get("number") == info.get("number")]
                                if len(matches) != 1:
                                    matches = [b for b in image_blocks if max(abs(a-bb) for a,bb in zip(b["bbox"], info["bbox"])) < .1]
                                if len(matches) != 1:
                                    raise ValueError(f"Inline image {ii} could not be uniquely extracted.")
                                extracted = matches[0]
                            if not extracted:
                                raise ValueError(f"No image bytes for {asset_id}.")
                            data, extension = extracted["image"], extracted["ext"]
                            method = "extracted_image"
                            if extracted.get("smask"):
                                base = pdf.Pixmap(document, xref)
                                mask = pdf.Pixmap(document, extracted["smask"])
                                data, extension = pdf.Pixmap(base, mask).tobytes("png"), "png"
                                method = "source_image_with_its_alpha_mask"
                            filename = f"assets/{asset_id}.{extension}"
                            (output / filename).write_bytes(data)
                            assets[asset_id] = {"path": filename, "sha256": digest(data), "method": method,
                                                "width": info["width"], "height": info["height"]}
                        units.append({"id": f"p{pi:04d}.image{ii:04d}", "kind": "image", "asset_id": asset_id,
                                      "bbox": info["bbox"], "transform": info["transform"]})
                    for di, drawing in enumerate(page.get_drawings(), 1):
                        units.append({"id": f"p{pi:04d}.vector{di:04d}", "kind": "vector", "drawing": drawing})
                    result = {"page": pi, "size": list(page.rect), "units": units, "links": page.get_links()}
                    filename = f"pages/page-{pi:04d}.json"
                    write_json(output / filename, result)
                    records.append({"page": pi, "path": filename, "unit_ids": [u["id"] for u in units],
                                    "sha256": digest((output / filename).read_bytes())})
                except Exception as exc:
                    errors.append({"page": pi, "error": str(exc)})
            manifest = {"schema_version": 1, "tool_version": VERSION,
                        "source": {"path": str(source.resolve()), "sha256": source_hash, "bytes": len(raw), "pages": len(document)},
                        "metadata": document.metadata, "outline": document.get_toc(simple=False),
                        "pages": records, "assets": assets, "fonts": fonts, "errors": errors,
                        "classification": "unreviewed", "rendered_pages": []}
            manifest["inventory_sha256"] = digest(json.dumps({"source": source_hash, "pages": records, "assets": assets}, sort_keys=True).encode())
        for number in selected:
            document[number - 1].get_pixmap(matrix=pdf.Matrix(1.5, 1.5)).save(output / "previews" / f"page-{number:04d}.png")
        manifest["rendered_pages"] = sorted(set(manifest.get("rendered_pages", [])) | set(selected))
        manifest["source"]["path"] = str(source.resolve())
        write_json(manifest_path, manifest)
    return {"status": "needs_attention" if manifest["errors"] else "ok", "packet": str(manifest_path.resolve()),
            "source_sha256": source_hash, "pages": manifest["source"]["pages"],
            "source_units": sum(len(p["unit_ids"]) for p in manifest["pages"]), "assets": len(manifest["assets"]),
            "preview_pages": manifest["rendered_pages"], "errors": manifest["errors"],
            "next": "Inspect previews and source units; create a semantic reconstruction.json. This packet is not a finished book."}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--render-pages", default="1")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = inspect(args.pdf, args.out, args.render_pages, args.refresh)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "ok" else 1
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
