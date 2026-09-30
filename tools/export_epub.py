"""A static, reflowable EPUB fallback from the validated book format.

This is a whole-book export selected separately from chapter-level delivery.
It includes readable activities and bibliography, never executable book code.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import zipfile
from pathlib import Path

from book_format import BookError, safe_relative, validate_book

H = lambda value: html.escape(str(value), quote=True)
XHTML = 'http://www.w3.org/1999/xhtml'


def inline(parts: list[dict], spec: dict) -> str:
    chapters={x['id']:x for x in spec['chapters']};resources={x['id']:x for x in spec.get('resources',[])}
    sources={x['id']:x for x in spec.get('sources',[])};activities={x['id']:x for x in spec.get('activities',[])}
    out=[]
    for part in parts:
        kind=part['type']
        if kind=='text':out.append(H(part['text']))
        elif kind=='strong':out.append(f"<strong>{H(part['text'])}</strong>")
        elif kind=='source':out.append(f'<a href="sources.xhtml#source-{H(part["id"])}">依据：{H(sources[part["id"]]["title"])}</a>')
        elif kind=='resource':out.append(f'<a href="resources.xhtml#resource-{H(part["id"])}">资料：{H(resources[part["id"]]["title"])}</a>')
        elif kind=='chapter':out.append(f'<a href="chapter-{H(part["id"])}.xhtml{("#"+H(part["anchor"])) if part.get("anchor") else ""}">接着读：{H(chapters[part["id"]]["title"])}</a>')
        elif kind=='activity':out.append(f'<a href="chapter-{H(activities[part["id"]]["chapter"])}.xhtml#activity-{H(part["id"])}">活动：{H(activities[part["id"]]["title"])}</a>')
    return ''.join(out)


def frame(title: str, body: str) -> bytes:
    document=f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="{XHTML}" xml:lang="zh-CN"><head><meta charset="utf-8"/><title>{H(title)}</title><link rel="stylesheet" type="text/css" href="style.css"/></head><body>{body}</body></html>'''
    return document.encode('utf-8')


def chapter_xhtml(doc: dict, chapter: dict, spec: dict) -> bytes:
    activities={a['id']:a for a in spec.get('activities',[])}
    rendered=[]
    for node in doc['nodes']:
        kind=node['type']
        if kind=='heading':rendered.append(f'<h{node["level"]} id="{H(node["id"])}">{inline(node["content"],spec)}</h{node["level"]}>')
        elif kind=='paragraph':rendered.append(f'<p>{inline(node["content"],spec)}</p>')
        elif kind=='code':rendered.append(f'<pre><code>{H(node["text"])}</code></pre>')
        elif kind=='list':rendered.append('<ul>'+''.join(f'<li>{inline(item,spec)}</li>' for item in node['items'])+'</ul>')
        elif kind=='image':
            resource=next(r for r in spec['resources'] if r['id']==node['id'])
            rendered.append(f'<figure><img src="images/{H(node["id"])}.png" alt="{H(resource.get("description") or resource["title"])}"/><figcaption>{H(resource["title"])}</figcaption></figure>')
        elif kind=='table':
            head=''.join(f'<th scope="col">{inline(cell,spec)}</th>' for cell in node['head'])
            rows=''.join('<tr>'+''.join(f'<td>{inline(cell,spec)}</td>' for cell in row)+'</tr>' for row in node['rows'])
            rendered.append(f'<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>')
        elif kind=='activity':
            activity=activities[node['id']]
            rendered.append(f'<aside id="activity-{H(node["id"])}"><h3>{H(activity["title"])}</h3><p>{H(activity["question"])}</p><p><a href="resources.xhtml#activity-{H(node["id"])}">打开活动：{H(activity["title"])}</a></p><p>查看输入与参数，按需要动手。在其他电子书阅读器中，此链接会打开活动说明。</p></aside>')
    return frame(chapter['title'],'\n'.join(rendered))


def source_xhtml(spec: dict) -> bytes:
    out=['<h1>原始出处与身份</h1><p>这些信息无需先取得大文件；原始网页是否可访问以来源方当前状态为准。</p>']
    for row in spec.get('sources',[]):
        out.append(f'<section id="source-{H(row["id"])}"><h2>{H(row["title"])}</h2>')
        out.append(f'<p>作者或机构：{H(row.get("creator") or "待核对")}。版本：{H(row.get("year") or "待核对")}。页段或范围：{H(row.get("locator") or "待核对")}。</p>')
        out.append(f'<p>这里用它支持：{H(row.get("supports") or "待核对")}</p>')
        if row.get('url'):out.append(f'<p><a href="{H(row["url"])}">访问来源方原始入口</a></p>')
        out.append('</section>')
    return frame('原始出处与身份',''.join(out))


def resource_xhtml(spec: dict) -> bytes:
    out=['<h1>配套材料与活动说明</h1><p>代码、数据和媒体为独立材料，可从同版书籍源项目取得。本文件含正文、插图、活动说明与来源。</p>']
    resources={r['id']:r for r in spec.get('resources',[])}
    for activity in spec.get('activities',[]):
        out.append(f'<section id="activity-{H(activity["id"])}"><h2>{H(activity["title"])}</h2><p>{H(activity["question"])}</p><p>{H(activity.get("observe", ""))}</p><h3>需要的材料</h3><ul>'+''.join(f'<li><a href="#resource-{H(key)}">{H(resources[key]["title"])}</a></li>' for key in activity['required'])+'</ul><h3>起始参数</h3><ul>'+''.join(f'<li>{H(value.get("label",name))}：{H(value["default"])}</li>' for name,value in activity.get('parameters',{}).items())+'</ul><p>在舒适阅读书库中可打开对应活动。其他阅读器可以按以上材料、参数和观察目标继续学习。</p></section>')
    for row in spec.get('resources',[]):
        out.append(f'<section id="resource-{H(row["id"])}"><h2>{H(row["title"])}</h2><p>{H(row.get("description") or "用途待说明")}</p><p>类型：{H(row["kind"])}。最小取得单位：{H(row["minimumUnit"])}。发行许可：{H(row.get("rights") or "待核对")}。</p></section>')
    out.append(f'<section><h2>版本与创作</h2><p>书 ID：{H(spec["id"])}；内容修订：{H(spec["revision"])}。</p><p>{H(spec.get("authorship") or "创作信息待核对")}</p></section>')
    return frame('配套材料',''.join(out))


def export_book(root: Path, output: Path):
    if output.exists():raise BookError('EPUB 目标已存在；请选择新文件，避免覆盖审阅过的版本')
    spec,documents=validate_book(root,root.name)
    if spec.get('sourceFormat') == 'epub-source@1':
        from rich_book import export_rich_book
        return export_rich_book(root, spec, output)
    parts=[('OEBPS/style.css',b'''body{font-family:serif;line-height:1.72;margin:5%;color:#24231f}h1,h2,h3{line-height:1.3}p,li{max-width:38em}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f0e8;padding:1em}table{border-collapse:collapse;max-width:100%}td,th{border:1px solid #bbb;padding:.3em;vertical-align:top}aside{border-left:3px solid #8b6845;padding:.7em 1em;background:#f6f3ea}a{color:#41576a}figure{margin:1.5em 0;padding:.7em;background:#f6f3ea}figure img{max-width:100%;height:auto}figcaption{font-size:.9em;color:#5f6864}''')]
    for chapter in spec['chapters']:
        parts.append((f'OEBPS/chapter-{chapter["id"]}.xhtml',chapter_xhtml(documents[chapter['id']],chapter,spec)))
    parts.extend([('OEBPS/sources.xhtml',source_xhtml(spec)),('OEBPS/resources.xhtml',resource_xhtml(spec))])
    image_ids={node['id'] for document in documents.values() for node in document['nodes'] if node['type']=='image'}
    resources={row['id']:row for row in spec.get('resources',[])}
    for key in sorted(image_ids):
        source=safe_relative(root,resources[key]['path'])
        parts.append((f'OEBPS/images/{key}.png',source.read_bytes()))
    toc=''.join(f'<li><a href="chapter-{H(c["id"])}.xhtml">{H(c["title"])}</a></li>' for c in spec['chapters'])
    parts.append(('OEBPS/nav.xhtml',frame('目录',f'<nav epub:type="toc" xmlns:epub="http://www.idpf.org/2007/ops"><h1>目录</h1><ol>{toc}<li><a href="sources.xhtml">原始出处与身份</a></li><li><a href="resources.xhtml">配套材料</a></li></ol></nav>')))
    manifest=''.join(f'<item id="item-{i}" href="{H(name.removeprefix("OEBPS/"))}" media-type="{("text/css" if name.endswith(".css") else "image/png" if name.endswith(".png") else "application/xhtml+xml")}"{(" properties="+chr(34)+"nav"+chr(34)) if name.endswith("nav.xhtml") else ""}/>' for i,(name,_) in enumerate(parts))
    spine=''.join(f'<itemref idref="item-{i}"/>' for i,(name,_) in enumerate(parts) if name.endswith('.xhtml') and not name.endswith('nav.xhtml'))
    modified=spec['revisionModified']
    opf=f'''<?xml version="1.0" encoding="UTF-8"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="book-id">{H(spec['id'])}</dc:identifier><dc:title>{H(spec['title'])}</dc:title><dc:language>{H(spec['language'])}</dc:language><meta property="dcterms:modified">{modified}</meta></metadata><manifest>{manifest}</manifest><spine>{spine}</spine></package>'''
    container=b'<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>'
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w') as archive:
        def put(name,data,compression):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.create_system=3
            info.compress_type=compression;info.external_attr=0o644<<16
            archive.writestr(info,data)
        put('mimetype',b'application/epub+zip',zipfile.ZIP_STORED)
        put('META-INF/container.xml',container,zipfile.ZIP_DEFLATED)
        put('OEBPS/content.opf',opf.encode('utf-8'),zipfile.ZIP_DEFLATED)
        for name,data in parts:put(name,data,zipfile.ZIP_DEFLATED)
    return {"bytes":output.stat().st_size,"sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"chapters":len(spec['chapters']),"fallback":"static_epub_no_executable_content"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('book',type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(export_book(args.book,args.out),ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except (BookError,OSError,ValueError,KeyError) as error:
        print(f'EPUB 导出停止：{error}',file=sys.stderr);raise SystemExit(2)
