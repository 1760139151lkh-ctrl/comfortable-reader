"""Publish only positively declared files into a versioned static-site tree.

--preview is local and includes review-pending original demo books. --release
requires a separate signed-off rights decision and never guesses it.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import subprocess
import html
from book_format import BookError, file_identity, safe_relative, valid_id, validate_book
from export_epub import export_book
from streamed_epub import prepare_reader

ROOT=Path(__file__).resolve().parents[1]
BOOKS=ROOT/'books'
WEB=ROOT/'site'
READER=ROOT/'desktop'
PRIVATE=re.compile(rb'(?i)(?:[A-Za-z]:[\\/]Users[\\/]|(?<![A-Za-z0-9~])/home/[^/\s]+/|(?<![A-Za-z0-9~])/Users/[^/\s]+/|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')


def write_json(path: Path, value: object):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')


def ensure_release_rights() -> dict:
    decision=ROOT/'release-rights.json'
    if not decision.is_file():
        raise BookError('尚无经权利人核对的 release-rights.json；只可构建本地 --preview')
    value=json.loads(decision.read_text(encoding='utf-8'))
    if value.get('approved') is not True or not value.get('repositoryOwner') or not value.get('codeLicense') or not value.get('bookLicense'):
        raise BookError('公开归属、软件许可与书籍许可尚未全部确认')
    for relative in ('LICENSE','BOOK-LICENSE'):
        if not (ROOT/relative).is_file():raise BookError(f'发行许可正文尚不存在：{relative}')
    return value


def validate_declaration(spec: dict, approval: dict|None):
    if approval is None:return
    if spec.get('rights',{}).get('status')!='approved':
        raise BookError(f"书籍 {spec.get('slug')} 的公开授权仍待核对")
    for resource in spec.get('resources',[]):
        if resource.get('rights')!='approved':
            raise BookError(f"资源 {spec.get('slug')}/{resource.get('id')} 尚未取得再分发确认")
    entry=approval.get('books',{}).get(spec['id'])
    if not entry or entry.get('revision')!=spec['revision'] or entry.get('license')!=approval['bookLicense']:
        raise BookError(f"发行许可决定未绑定这本书的当前 ID/修订：{spec['id']}")


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preview',action='store_true',help='local-only preview with pending original examples')
    mode.add_argument('--release',action='store_true',help='rights-approved publishing artifact')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--books-dir',type=Path,default=BOOKS,help='preview only: another self-contained book collection root')
    parser.add_argument('--reader-dist',type=Path,help='reuse a reviewed build of the shared reader shell')
    args=parser.parse_args()
    books_dir=args.books_dir.resolve(strict=True)
    if args.release and books_dir!=BOOKS.resolve(strict=True):
        raise BookError('公开发行只能使用仓库里经纳入清单确认的书目')
    output=args.out.resolve()
    if output==ROOT or ROOT in output.parents and (output.parent!=ROOT or not re.fullmatch(r'site-dist(?:-r\d{2,})?',output.name)):
        raise BookError('仓库内输出只可用顶层独立的 site-dist 或 site-dist-rNN')
    if output.exists() and any(output.iterdir()):
        raise BookError('输出目录必须为空；先审阅并另选一个新目录，避免覆盖用户文件')
    approval=ensure_release_rights() if args.release else None
    listing=json.loads((books_dir/'publication-list.json').read_text(encoding='utf-8'))
    if listing.get('schemaVersion')!=1 or not isinstance(listing.get('books'),list) or len(listing['books'])>500:
        raise BookError('书目结构错误或书数超过预算')
    if args.release and listing.get('releaseApproved') is not True:
        raise BookError('书目未被标为已审查可发行')
    if len(set(listing['books']))!=len(listing['books']):raise BookError('书目有重复书籍')
    built=[]
    for slug in listing['books']:
        valid_id(slug,'书目中的目录名')
        folder=books_dir/slug
        spec,documents=validate_book(folder,slug)
        validate_declaration(spec,approval)
        built.append((slug,folder,spec,documents))
    output.mkdir(parents=True,exist_ok=True)
    catalog={'schemaVersion':1,'status':'release' if args.release else 'local_preview_rights_pending','books':[]}
    if approval:
        owner=approval['repositoryOwner'];name=approval.get('repositoryName','comfortable-reader')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',owner) or not re.fullmatch(r'[A-Za-z0-9_.-]+',name):raise BookError('公开仓库名称无效')
        base=f'https://github.com/{owner}/{name}'
        tag=approval.get('releaseTag','v0.5.0')
        if not re.fullmatch(r'v\d+\.\d+\.\d+',tag):raise BookError('发行标签格式无效')
        catalog['project']={'repository':base,'download':base+'/releases/tag/'+tag}
    written=[]
    for slug,folder,spec,documents in built:
        dest=output/'books'/slug
        epub=output/'ebooks'/f'{slug}-{spec["revision"]}.epub'
        epub_result=export_book(folder,epub)
        epub_final=epub.with_name(f'{slug}-{spec["revision"]}-{epub_result["sha256"][:16]}.epub')
        epub.rename(epub_final);epub=epub_final;written.append(epub)
        book_manifest={k:v for k,v in spec.items() if k not in ('chapters','resources','epubSource','studySource')}
        book_manifest['chapters']=[];book_manifest['resources']=[]
        reader,reading_documents,reader_files=prepare_reader(epub,dest,spec)
        book_manifest['reader']=reader;written.extend(reader_files)
        if spec.get('studySource'):
            from rich_book import compiled_study
            study_file=dest/'learning.json'
            write_json(study_file,compiled_study(folder,spec,epub_result['sha256']))
            book_manifest['studyDescriptor']={'path':'learning.json',**file_identity(study_file)}
            written.append(study_file)
        for chapter in spec['chapters']:
            cid=chapter['id'];doc=documents[cid]
            if spec.get('sourceFormat') == 'epub-source@1':
                book_manifest['chapters'].append({k:v for k,v in chapter.items() if k!='source'}|{'body':reading_documents[cid],'readingDocument':reading_documents[cid],'readingDependencies':reader['chapterDependencies'][cid]})
                continue
            references=doc.pop('references',[])
            source_ids=sorted({ref['id'] for ref in references if ref['type']=='source'})
            resource_ids=sorted({ref['id'] for ref in references if ref['type'] in ('resource','image')})
            bodybytes=(json.dumps(doc,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')
            bodyname=f"{cid}-{hashlib.sha256(bodybytes).hexdigest()[:16]}.json"
            bodypath=dest/'chapters'/bodyname
            write_json(bodypath,doc);written.append(bodypath)
            book_manifest['chapters'].append({k:v for k,v in chapter.items() if k!='source'}|{'body':{'path':f'chapters/{bodyname}',**file_identity(bodypath)},'readingDocument':reading_documents[cid],'readingDependencies':reader['chapterDependencies'][cid],'sourceIds':source_ids,'resourceIds':resource_ids})
        for res in spec.get('resources',[]):
            source=safe_relative(folder,res['path'])
            item={k:v for k,v in res.items() if k not in ('downloadChunkBytes','path')}|{'logicalPath':res['path']}|file_identity(source)
            if res['kind']=='image':
                header=source.read_bytes()[:24]
                item['width']=int.from_bytes(header[16:20],'big');item['height']=int.from_bytes(header[20:24],'big')
            chunk_size=res.get('downloadChunkBytes') or 4*1024*1024
            if chunk_size and source.stat().st_size>chunk_size:
                item['chunks']=[]
                with source.open('rb') as incoming:
                    index=0
                    while piece:=incoming.read(chunk_size):
                        relative=f"chunks/{res['id']}/{index:05d}-{hashlib.sha256(piece).hexdigest()[:16]}.part"
                        part=dest/relative;part.parent.mkdir(parents=True,exist_ok=True);part.write_bytes(piece);written.append(part)
                        item['chunks'].append({'path':relative,**file_identity(part)})
                        index+=1
            else:
                relative=f"assets/{res['id']}/{source.stem}-{item['sha256'][:16]}{source.suffix}"
                item['path']=relative
                destfile=dest/relative;destfile.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,destfile);written.append(destfile)
            book_manifest['resources'].append(item)
        manifest=dest/'manifest.json';write_json(manifest,book_manifest);written.append(manifest)
        catalog['books'].append({'id':spec['id'],'slug':slug,'revision':spec['revision'],'example':bool(spec.get('example',False)),'author':spec.get('author',''),'language':spec['language'],'title':spec['title'],'description':spec.get('description',''),'chapters':[{'id':c['id'],'title':c['title']} for c in spec['chapters']],'manifest':{'path':f'books/{slug}/manifest.json',**file_identity(manifest)},'epub':{'path':f'ebooks/{epub.name}',**file_identity(epub)}})
    write_json(output/'catalog.json',catalog);written.append(output/'catalog.json')
    reader_dist=args.reader_dist.resolve() if args.reader_dist else READER/'dist'
    if not args.reader_dist:
        npm=shutil.which('npm.cmd' if os.name=='nt' else 'npm')
        if not npm or not (READER/'node_modules').is_dir():raise BookError('共享阅读器依赖尚未准备；先在 desktop 执行 npm ci，或用 --reader-dist 指向已构建版本')
        subprocess.run([npm,'run','build'],cwd=READER,check=True)
    if not (reader_dist/'index.html').is_file():raise BookError('共享阅读器构建入口不存在')
    shell_files=[]
    for source in reader_dist.rglob('*'):
        if not source.is_file():continue
        relative=source.relative_to(reader_dist).as_posix()
        if source.is_symlink():raise BookError('共享阅读器构建中不能含符号链接')
        destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination);written.append(destination)
        if relative=='index.html' or relative.startswith('assets/'):shell_files.append('./'+relative)
    shell_hash=hashlib.sha256(b''.join((output/p.removeprefix('./')).read_bytes() for p in sorted(shell_files))).hexdigest()[:16]
    worker=(WEB/'sw.js').read_text(encoding='utf-8').replace('__SHELL_VERSION__',shell_hash).replace('__SHELL_FILES__',json.dumps(['./']+shell_files+['./catalog.json']))
    (output/'sw.js').write_text(worker,encoding='utf-8');written.append(output/'sw.js')
    # Retain component notices with the JavaScript distribution, without
    # fetching them during ordinary reading or copying personal build state.
    notices=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>组件来源与许可 · 舒适阅读书库</title><style>body{max-width:70ch;margin:3rem auto;padding:0 1.5rem;background:#faf8f1;color:#292c29;font:16px/1.8 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace}summary{cursor:pointer}a{color:#875b35}</style><h1>组件来源与许可</h1><p>阅读器使用的第三方组件保留各自的许可。这里列出网页依赖，书籍内容另按各自来源使用。</p>']
    dependencies=json.loads((READER/'licenses/dependencies.json').read_text(encoding='utf-8'))
    for dependency in dependencies:
        if dependency['ecosystem']!='npm':continue
        notices.append('<details><summary>'+html.escape(dependency['name']+' '+dependency['version']+' · '+str(dependency['license']))+'</summary>')
        for name in dependency['notice_files']:notices.append('<pre>'+html.escape((READER/name).read_text(encoding='utf-8',errors='replace'))+'</pre>')
        notices.append('</details>')
    notices.append('<p>PDF.js、OpenJPEG 与标准字体的许可随静态文件保留在 vendor/pdfjs 中。</p></html>')
    legal=output/'third-party.html';legal.write_text(''.join(notices),encoding='utf-8');written.append(legal)
    if approval:
        shutil.copyfile(ROOT/'LICENSE',output/'LICENSE');shutil.copyfile(ROOT/'BOOK-LICENSE',output/'BOOK-LICENSE');written.extend([output/'LICENSE',output/'BOOK-LICENSE'])
    # Last positive-output scan. Do not print matched values or include logs.
    for file in written:
        if PRIVATE.search(file.read_bytes()):
            raise BookError(f'待发行内容包含本机身份/路径模式：{file.relative_to(output)}；停止发布，不打印原值')
    receipt={'mode':'release' if approval else 'preview_only','books':len(built),'files':len(written),'total_bytes':sum(p.stat().st_size for p in written),'catalog_sha256':file_identity(output/'catalog.json')['sha256'],'private_pattern_matches':0,'source_history_included':False}
    write_json(output/'build-receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except (BookError,FileNotFoundError,UnicodeDecodeError,ValueError,json.JSONDecodeError) as error:
        print(f'构建停止：{error}',file=sys.stderr)
        raise SystemExit(2)
