"""Build/audit CRLearn v1 data. This tool never executes recipes or grants trust."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, re, stat, sys, zipfile, wave
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

SCHEMA_VERSION=1
MAX_ASSET=128*1024*1024
RESERVED={'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}

def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def relative_file(root:Path,relative:str)->Path:
    if not isinstance(relative,str) or not relative or any(c in relative for c in ('\\',':','%','\0')):
        raise ValueError('ambiguous or unsafe relative asset path')
    parts=relative.split('/')
    if any(p in ('','.','..') or p.endswith(('.', ' ')) or p.split('.')[0].upper() in RESERVED for p in parts):
        raise ValueError('asset path traversal, device name or ambiguous component')
    base=root.resolve(strict=True);path=base
    for part in parts:
        path=path/part;info=path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
            raise ValueError('symlink or Windows reparse point in asset path')
    resolved=path.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():raise ValueError('asset is not a file in its bound root')
    return resolved

def epub_identity(path:Path)->dict:
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('duplicate ZIP entries')
        for info in archive.infolist():
            name=info.filename
            if name.startswith('/') or '\\' in name or ':' in name or '..' in PurePosixPath(name).parts:
                raise ValueError('unsafe EPUB ZIP entry')
            if stat.S_ISLNK((info.external_attr>>16)&0xffff):raise ValueError('ZIP link entry')
        container=ET.fromstring(archive.read('META-INF/container.xml'))
        package=next(e.attrib['full-path'] for e in container.iter() if e.tag.endswith('rootfile'))
        opf=ET.fromstring(archive.read(package));ns={'o':'http://www.idpf.org/2007/opf'}
        identifiers=[e.text for e in opf.iter() if e.tag.endswith('identifier')]
        manifest={e.attrib['id']:e.attrib for e in opf.findall('o:manifest/o:item',ns)}
        spine=[manifest[e.attrib['idref']]['href'] for e in opf.findall('o:spine/o:itemref',ns)]
    return {'sha256':sha(path),'identifiers':identifiers,'spine':spine}

def audit(pack:dict,epub:Path,root:Path,derived_root:Path|None=None,profile_root:Path|None=None)->dict:
    identity=epub_identity(epub)
    if pack.get('schema_version')!=SCHEMA_VERSION:raise ValueError('unsupported CRLearn version')
    if pack.get('book_revision_sha256')!=identity['sha256']:raise ValueError('stale EPUB binding')
    if pack.get('book_uuid') not in identity['identifiers']:raise ValueError('wrong publication identity')
    assets=pack.get('assets',[]);ids=set()
    for asset in assets:
        identifier=asset['id']
        if identifier in ids or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',identifier):raise ValueError('invalid or duplicate asset id')
        ids.add(identifier)
        kind=asset.get('kind')
        if kind not in {'code','text','image','audio','video','pdf','geometry','data','model'}:raise ValueError('unknown asset kind')
        which=asset.get('root','publication')
        if which not in ('publication','derived','profile'):raise ValueError('unknown asset root')
        selected=profile_root if which=='profile' else derived_root if which=='derived' else root
        if selected is None:raise ValueError('derived root is required')
        path=relative_file(selected,asset['relative_path'])
        if path.stat().st_size!=asset['bytes'] or sha(path)!=asset['sha256']:raise ValueError(f'asset bytes changed: {identifier}')
        # Formats with clear signatures must match their declared kind.
        with path.open('rb') as f:head=f.read(16)
        if kind=='audio' and path.suffix.lower()=='.wav' and not(head[:4]==b'RIFF' and head[8:12]==b'WAVE'):raise ValueError('invalid WAV signature')
        if kind=='video' and path.suffix.lower()=='.mp4' and head[4:8]!=b'ftyp':raise ValueError('invalid MP4 signature')
        if kind=='pdf' and not head.startswith(b'%PDF-'):raise ValueError('invalid PDF signature')
    chapters=pack.get('chapters',[]);chapter_ids={c['id'] for c in chapters}
    if len(chapter_ids)!=len(chapters):raise ValueError('duplicate chapter ids')
    for chapter in chapters:
        if chapter['href'] not in identity['spine']:raise ValueError('chapter absent from actual EPUB spine')
        if any(a not in ids for a in chapter.get('assets',[])):raise ValueError('chapter has missing resource identity')
    for value in pack.get('href_assets',{}).values():
        if value not in ids:raise ValueError('broken EPUB resource mapping')
    for activity in pack.get('activities',[]):
        if any(key in activity for key in ('command','shell','cwd','python_path','trusted','env')):raise ValueError('execution authority must not be supplied by a book')
        if activity['entry_asset'] not in ids or activity['chapter'] not in chapter_ids:raise ValueError('activity points outside registered content')
        if 'run_definition_json' in activity:
            raw=activity['run_definition_json']
            if not isinstance(raw,str) or len(raw.encode('utf-8'))>65536:raise ValueError('run definition exceeds budget')
            definition=json.loads(raw)
            if definition.get('schemaVersion')!=1 or definition.get('kind')!='python-cli@1':raise ValueError('unsupported run definition')
            if definition.get('entry_asset')!=activity['entry_asset']:raise ValueError('visible source and execution entry differ')
            references=[definition['entry_asset'],*definition.get('input_assets',[]),*([definition['adapter_asset']] if definition.get('adapter_asset') else [])]
            if any(key not in ids for key in references):raise ValueError('run definition has missing resource')
            if any(key in definition for key in ('trusted','command','shell','cwd','python_path','env')):raise ValueError('book definition cannot grant native authority')
    for source in pack.get('source_claims',[]):
        url=urlsplit(source['url'])
        if url.scheme not in ('http','https') or not url.hostname or url.username or url.password:raise ValueError('unsafe source URL')
    for asset in assets:
        for frame in asset.get('media',{}).get('decoded_frames',[]):
            if frame['asset_id'] not in ids:raise ValueError('decoded frame not registered')
    return {'status':'structurally_verified','schema_version':1,'book_uuid':pack['book_uuid'],'epub_sha256':identity['sha256'],
            'chapters':len(chapters),'publication_assets':sum(a.get('root')!='derived' for a in assets),'derived_assets':sum(a.get('root')=='derived' for a in assets),
            'activities':len(pack.get('activities',[])),'execution_authorized':False,'desktop_verification':'awaiting_desktop_verification',
            'scope':'resource identity, package structure and source URLs; not pedagogy, mathematical correctness or OS isolation'}

def build(spec:dict,epub:Path,root:Path,derived_root:Path|None=None,profile_root:Path|None=None)->dict:
    pack=copy.deepcopy(spec);identity=epub_identity(epub)
    pack['schema_version']=1;pack['book_revision_sha256']=identity['sha256']
    if 'book_uuid' not in pack:
        if len(identity['identifiers'])!=1:raise ValueError('select the intended EPUB identifier explicitly')
        pack['book_uuid']=identity['identifiers'][0]
    for asset in pack.get('assets',[]):
        selected=profile_root if asset.get('root')=='profile' else derived_root if asset.get('root')=='derived' else root
        if selected is None:raise ValueError('derived root missing')
        file=relative_file(selected,asset['relative_path']);asset['bytes']=file.stat().st_size;asset['sha256']=sha(file)
        asset.setdefault('filename',file.name);asset.setdefault('role','author_recorded');asset.setdefault('chapters',[])
        if asset['kind']=='audio' and file.suffix.lower()=='.wav':
            with wave.open(str(file),'rb') as audio:
                asset['media']={**asset.get('media',{}),'sample_rate':audio.getframerate(),'channels':audio.getnchannels(),'frames':audio.getnframes(),'duration':audio.getnframes()/audio.getframerate()}
    for index,chapter in enumerate(pack.get('chapters',[]),1):
        chapter.setdefault('number',index);chapter.setdefault('title',f'第 {index} 部分');chapter.setdefault('question','');chapter.setdefault('assets',[])
        for asset in pack['assets']:
            if asset['id'] in chapter['assets'] and chapter['id'] not in asset['chapters']:asset['chapters'].append(chapter['id'])
    for key,default in [('audio_groups',[]),('activities',[]),('source_claims',[]),('dependencies',[]),('history',[]),('href_assets',{})]:pack.setdefault(key,default)
    for activity in pack['activities']:
        if 'run_definition' in activity:
            definition=activity.pop('run_definition')
            activity['run_definition_json']=json.dumps(definition,ensure_ascii=False,sort_keys=True,separators=(',',':'))
            activity['required_assets']=list(dict.fromkeys([definition['entry_asset'],*definition.get('input_assets',[]),*([definition['adapter_asset']] if definition.get('adapter_asset') else [])]))
    audit(pack,epub,root,derived_root,profile_root)
    return pack

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('build','audit'))
    parser.add_argument('input',type=Path,help='Declarative JSON spec or existing .crlearn file')
    parser.add_argument('--epub',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True,help='Explicit local publication resource root; never saved in the pack')
    parser.add_argument('--derived-root',type=Path)
    parser.add_argument('--profile-root',type=Path,help='Book-owned profile files; keep these beside the output .crlearn')
    parser.add_argument('--output',type=Path,help='Required for build; output is not installed automatically')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();pack=json.loads(args.input.read_text(encoding='utf-8'))
    if args.action=='build':
        if not args.output:parser.error('--output required for build')
        pack=build(pack,args.epub,args.root,args.derived_root,args.profile_root or args.input.parent)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(pack,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=audit(pack,args.epub,args.root,args.derived_root,args.profile_root or args.input.parent)
    report['package_sha256']=sha(args.output if args.action=='build' else args.input)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError,ET.ParseError,zipfile.BadZipFile) as error:
        print(json.dumps({'status':'failed','error':str(error)},ensure_ascii=False),file=sys.stderr);raise SystemExit(1)
