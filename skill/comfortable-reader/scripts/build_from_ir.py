#!/usr/bin/env python3
"""Build a PDF reconstruction JSON into EPUB, with source-unit coverage checks."""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import json
import mimetypes
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from build_reader import make_epub
from epub_tools import audit_epub, local
from math_rendering import render_mathml


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_packet(path: Path) -> tuple[dict, dict]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    units = {}
    for page in packet["pages"]:
        raw = (path.parent / page["path"]).read_bytes()
        if sha(raw) != page["sha256"]:
            raise ValueError("Source packet page changed: " + page["path"])
        content = json.loads(raw)
        if [u["id"] for u in content["units"]] != page["unit_ids"]:
            raise ValueError("Source packet unit index differs from its page data.")
        for unit in content['units']:
            if unit['id'] in units:
                raise ValueError('Duplicate source-packet unit ID: '+unit['id'])
            units[unit['id']]={**unit,'page':page['page']}
    return packet, units


class SemanticRenderer:
    def __init__(self, packet_path: Path, packet: dict, units: dict):
        self.packet_path, self.packet, self.units = packet_path, packet, units
        self.targets = []
        self.pending = []
        self.ids = set()
        self.counter = 0

    def runs(self, runs) -> str:
        if isinstance(runs, str):
            return html.escape(runs)
        if not isinstance(runs, list):
            raise ValueError("runs must be a string or a list of text/math objects, not an arbitrary object.")
        out = []
        for run in runs:
            if isinstance(run, str):
                out.append(html.escape(run)); continue
            if not isinstance(run, dict):
                raise ValueError("Each run must be text or a text/math object.")
            if "math" in run:
                text = '<span class="math-inline">' + render_mathml(run["math"], False) + '</span>'
            elif "text" in run:
                text = html.escape(run["text"])
                for field, tag in (("em", "em"), ("strong", "strong"), ("underline", "u"), ("sub", "sub"), ("sup", "sup")):
                    if run.get(field):
                        text = '<'+tag+'>'+text+'</'+tag+'>'
            else:
                raise ValueError("A run needs text or math; no raw-HTML fallback is supported.")
            if run.get("href"):
                text = '<a href="'+html.escape(run["href"], quote=True)+'">'+text+'</a>'
            out.append(text)
        return ''.join(out)

    def block(self, block: dict, inherited_sources: bool = False) -> str:
        if not isinstance(block, dict):
            raise ValueError("Each semantic block must be an object with type, sources and content.")
        self.counter += 1
        ident = block.get("id", f"block-{self.counter:05d}")
        if ident in self.ids or not ident or re.search(r"[\s#]", ident):
            raise ValueError("Use unique, non-empty block ids without whitespace or #: " + ident)
        self.ids.add(ident)
        refs = block.get("sources", [])
        if not isinstance(refs, list):
            raise ValueError("sources must be a list of source-unit IDs.")
        if not refs and not inherited_sources:
            raise ValueError("Visible block has no source binding: " + ident)
        for ref in refs:
            if ref not in self.units:
                raise ValueError("Unknown source unit: " + ref)
            self.targets.append({"source_id": ref, "block_id": ident})
        evidence = block.get("source_evidence", [])
        if block.get("source_reviewed") is not True or not isinstance(evidence, list) or not evidence or any(not isinstance(item, dict) or not item.get("page") or not item.get("observation") for item in evidence):
            self.pending.append(ident)
        kind = block["type"]
        attr = ' id="'+html.escape(ident, quote=True)+'"'
        role=block.get('role')
        role_classes={'caption':'reader-caption','bibliography':'reader-bibliography','index-entry':'reader-index-entry',
                      'toc':'reader-contents-entry','local-heading':'reader-local-heading','colophon':'reader-colophon',
                      'algorithm-analysis':'algorithm-cost'}
        if role:
            if role not in role_classes:raise ValueError('Unknown semantic layout role: '+str(role))
            attr+=' class="'+role_classes[role]+'"'
        if kind in {"paragraph", "heading"}:
            level = int(block.get("level", 1))
            if kind == "heading" and not 1 <= level <= 6:
                raise ValueError("Heading level must be 1–6.")
            tag = "p" if kind == "paragraph" else "h" + str(level)
            runs = block.get("runs", [])
            result = self.runs(runs)
            if refs and all(self.units[ref]["kind"] == "text" for ref in refs) and not any(isinstance(run,dict) and 'math' in run for run in (runs if isinstance(runs,list) else [])):
                from epub_tools import visible_text
                actual = visible_text(ET.fromstring('<root>'+result+'</root>'))
                expected = ''.join(self.units[ref]['text'] for ref in refs)
                normalize = lambda s: ''.join(unicodedata.normalize('NFKC',s).split())
                if normalize(actual) != normalize(expected):
                    raise ValueError("Plain paragraph/heading text differs from its bound source units: "+ident+". Check order and characters; do not claim coverage from IDs alone.")
            return '<'+tag+attr+'>'+result+'</'+tag+'>'
        if kind == "math":
            return '<div class="math-block"'+attr+'>'+render_mathml(block["latex"], True)+'</div>'
        if kind == "code":
            return '<pre'+attr+'><code>'+html.escape(block["text"])+'</code></pre>'
        if kind == 'rule':
            return '<hr'+attr+'/>'
        if kind in {"section", "quote"}:
            tag = "section" if kind == "section" else "blockquote"
            return '<'+tag+attr+'>'+''.join(self.block(b, inherited_sources or bool(refs)) for b in block["blocks"])+'</'+tag+'>'
        if kind == "list":
            tag = "ol" if block.get("ordered") else "ul"
            start = ' start="'+str(int(block.get("start", 1)))+'"' if tag == "ol" else ''
            items=[]
            for item in block['items']:
                content=''.join(self.block(b,inherited_sources or bool(refs)) for b in item['blocks']) if isinstance(item,dict) and 'blocks' in item else self.runs(item)
                items.append('<li>'+content+'</li>')
            return '<'+tag+attr+start+'>'+''.join(items)+'</'+tag+'>'
        if kind == "algorithm":
            rows = []
            groups=block.get('case_groups',[])
            group_starts={};covered=set()
            for group in groups:
                start,end=int(group['start_row']),int(group['end_row'])
                if start<0 or end<start or end>=len(block['lines']) or any(i in covered for i in range(start,end+1)):
                    raise ValueError('Invalid or overlapping algorithm case group: '+ident)
                group_starts[start]=group;covered.update(range(start,end+1))
            for row_index,line in enumerate(block["lines"]):
                indent = int(line.get("indent", 0))
                if indent < 0:
                    raise ValueError("Algorithm indentation cannot be negative.")
                runs=line.get('runs',[])
                split=int(line.get('comment_start',len(runs)))
                if not 0<=split<=len(runs):raise ValueError('Invalid algorithm comment boundary: '+ident)
                content=self.runs(runs[:split])
                if split<len(runs):content+='<span class="algorithm-comment">'+self.runs(runs[split:])+'</span>'
                if line.get('role')=='title':
                    rows.append('<tr class="algorithm-title"><td colspan="'+str(3 if groups else 2)+'">'+content+'</td></tr>')
                else:
                    case=''
                    if row_index in group_starts:
                        group=group_starts[row_index]
                        case='<td class="algorithm-case" rowspan="'+str(int(group['end_row'])-row_index+1)+'">'+render_mathml(group['latex'],False)+'</td>'
                    elif groups and row_index not in covered:case='<td class="algorithm-case-empty"/>'
                    rows.append('<tr><td>'+html.escape(str(line.get("number", "")))+'</td><td class="algorithm-code" style="padding-left:'+str(indent)+
                                'em">'+content+'</td>'+case+'</tr>')
            parts=[];cursor=0
            while cursor<len(rows):
                if cursor in group_starts:
                    end=int(group_starts[cursor]['end_row'])+1
                    parts.append('<tbody class="algorithm-case-group">'+''.join(rows[cursor:end])+'</tbody>');cursor=end
                else:
                    end=min([start for start in group_starts if start>cursor] or [len(rows)])
                    parts.append('<tbody>'+''.join(rows[cursor:end])+'</tbody>');cursor=end
            return '<table class="reader-algorithm"'+attr+'>'+''.join(parts)+'</table>'
        if kind == "table":
            rows = block["rows"]
            if not rows or not rows[0]:
                raise ValueError('A table needs explicit rows and cells: '+ident)
            rendered = []
            occupied=set()
            for index, row in enumerate(rows):
                tag = "th" if index < int(block.get("header_rows", 0)) else "td"
                cells=[];column=0
                for cell in row:
                    while (index,column) in occupied:column+=1
                    if isinstance(cell,dict):
                        colspan,rowspan=int(cell.get('colspan',1)),int(cell.get('rowspan',1))
                        content=cell.get('runs',[])
                    else:colspan=rowspan=1;content=cell
                    if colspan<1 or rowspan<1 or index+rowspan>len(rows):
                        raise ValueError('Invalid merged-cell extent: '+ident)
                    for y in range(index,index+rowspan):
                        for x in range(column,column+colspan):
                            if (y,x) in occupied:raise ValueError('Overlapping merged cells: '+ident)
                            occupied.add((y,x))
                    attrs=(' colspan="'+str(colspan)+'"' if colspan!=1 else '')+(' rowspan="'+str(rowspan)+'"' if rowspan!=1 else '')
                    cells.append('<'+tag+attrs+'>'+self.runs(content)+'</'+tag+'>');column+=colspan
                rendered.append('<tr>'+''.join(cells)+'</tr>')
            width=max(x for _,x in occupied)+1
            if any((y,x) not in occupied for y in range(len(rows)) for x in range(width)):
                raise ValueError('Table rows leave undeclared cells; include explicit blanks: '+ident)
            header_rows=int(block.get('header_rows',0))
            if not 0<=header_rows<=len(rendered):raise ValueError('Invalid header row count: '+ident)
            head='<thead>'+''.join(rendered[:header_rows])+'</thead>' if header_rows else ''
            return '<table'+attr+'>'+head+'<tbody>'+''.join(rendered[header_rows:])+'</tbody></table>'
        if kind == "figure":
            asset_id = block["asset_id"]
            if block.get("figure_kind") not in {"diagram", "photo", "illustration", "composite_visualization"} or not block.get("visual_elements_checked"):
                raise ValueError("A figure needs its visual category and checked visual elements, not merely a figure number: " + ident)
            if not any(self.units[ref].get("asset_id") == asset_id for ref in refs):
                raise ValueError("Figure asset must be bound to its image occurrence source unit: " + ident)
            asset = self.packet["assets"][asset_id]
            data = (self.packet_path.parent / asset["path"]).read_bytes()
            if sha(data) != asset["sha256"]:
                raise ValueError("Figure asset changed after source inspection: " + asset_id)
            mime = mimetypes.guess_type(asset["path"])[0] or "image/png"
            caption_id=ident+'-caption'
            described=' aria-describedby="'+caption_id+'"' if block.get('caption') else ''
            image = '<img alt="'+html.escape(block.get("alt", ""), quote=True)+'"'+described+' src="data:'+mime+';base64,'+base64.b64encode(data).decode()+'"/>'
            captions = ''.join(self.block(b, True) for b in block.get("caption", []))
            return '<figure'+attr+'>'+image+'<figcaption id="'+caption_id+'">'+captions+'</figcaption></figure>'
        raise ValueError("Unknown semantic block type: " + kind)


def build(ir_path: Path, output: Path, draft: bool = False, max_section_chars: int = 16000) -> dict:
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    packet_path = (ir_path.parent / ir["packet"]).resolve()
    packet, units = load_packet(packet_path)
    source_path = Path(packet["source"]["path"])
    source = source_path.read_bytes()
    if sha(source) != packet["source"]["sha256"]:
        raise ValueError("Original PDF changed after inspection.")
    scope = ir.get("scope", {})
    if scope.get("kind") not in {"sample", "full"}:
        raise ValueError("Declare scope.kind as sample or full.")
    expected = set(units) if scope["kind"] == "full" else {ident for ident,u in units.items() if u["page"] in scope.get("pages", [])}
    if not expected:
        raise ValueError("The declared scope has no source units.")
    renderer = SemanticRenderer(packet_path, packet, units)
    fragment = '\n'.join(renderer.block(b) for b in ir["blocks"])
    counts = Counter(row["source_id"] for row in renderer.targets)
    exclusions = ir.get("excluded_units", [])
    for exclusion in exclusions:
        ident = exclusion["source_id"]
        if ident not in units or exclusion.get("reason") not in {"page_background", "running_header", "page_number", "verified_duplicate", "explicit_user_exclusion"} or not exclusion.get("evidence"):
            raise ValueError("An exclusion needs a known source id, a specific permitted reason, and evidence.")
        if exclusion["reason"] == "verified_duplicate" and exclusion.get("same_as") not in counts:
            raise ValueError("A verified duplicate must name a source unit retained in the reconstruction.")
        counts[ident] += 1
    duplicate = sorted(ident for ident,n in counts.items() if n != 1)
    missing, extra = sorted(expected - set(counts)), sorted(set(counts) - expected)
    ready = scope["kind"] == "full" and not (duplicate or missing or extra or renderer.pending or ir.get("issues") or packet["errors"])
    if not draft and not ready:
        raise ValueError("Source reconstruction is not ready. Use --draft for a sample; resolve missing/duplicate units, pending source reviews and issues before a full build.")
    title = ir.get("title")
    if not title:
        raise ValueError("Set metadata title; do not insert an invented body title.")
    if not isinstance(ir.get('authors',[]),list):
        raise ValueError('authors must be a list, for example ["Author Name"].')
    author = " & ".join(ir.get("authors", [])) or "未知"
    if not fragment.strip():
        raise ValueError('No reconstructed visible blocks were supplied.')
    make_epub(output, title, author, fragment, sha(source)+sha(ir_path.read_bytes()),
              source_archive_name="source.pdf", source_reference=str(source_path),
              original_source_bytes=source, max_section_chars=max_section_chars,
              navigation=ir.get('navigation'), authors=ir.get('authors'))
    audit = audit_epub(output)
    if audit["errors"]:
        raise ValueError("Generated EPUB failed structural audit: " + '; '.join(audit["errors"]))
    targets = {}
    with zipfile.ZipFile(output) as z:
        for name in z.namelist():
            if name.endswith('.xhtml'):
                for element in ET.fromstring(z.read(name)).iter():
                    if element.get('id'):
                        targets[element.get('id')] = name+'#'+element.get('id')
    return {"schema_version": 1, "epub_sha256": audit["epub_sha256"], "source_sha256": sha(source),
            "inventory_sha256": packet["inventory_sha256"], "scope": scope["kind"],
            "source_packet": str(packet_path), "expected_source_ids": sorted(expected),
            "mapped_units": [{"source_id": row["source_id"], "target": targets[row["block_id"]]} for row in renderer.targets],
            "excluded_units": exclusions, "missing_source_ids": missing, "duplicate_source_ids": duplicate,
            "out_of_scope_source_ids": extra, "unreviewed_block_ids": renderer.pending,
            "issues": ir.get("issues", []) + packet["errors"], "source_review_passed": ready,
            "spine_sections": audit["spine_sections"], "mathml_formulas": audit["mathml_formulas"],
            "delivery_status": "draft_not_imported"}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path)
    parser.add_argument("--epub", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--max-section-chars", type=int, default=16000)
    args = parser.parse_args(argv)
    try:
        report = build(args.ir.resolve(), args.epub.resolve(), args.draft, args.max_section_chars)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "ok", "coverage_report": str(args.report.resolve()),
                          "source_review_passed": report["source_review_passed"],
                          "missing_units": len(report["missing_source_ids"]), "pending_blocks": len(report["unreviewed_block_ids"]),
                          "epub_sha256": report["epub_sha256"], "delivery_status": "draft_not_imported"}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
