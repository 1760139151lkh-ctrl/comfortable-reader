"""Expose the checked EPUB as separately retrievable immutable chapter files."""
from pathlib import Path
import hashlib,json,subprocess,zipfile,xml.etree.ElementTree as ET,posixpath
from urllib.parse import unquote,urlsplit

def member_path(base:str,href:str)->str:
    if not isinstance(href,str):raise ValueError('Invalid EPUB member reference')
    parsed=urlsplit(href)
    if parsed.scheme or parsed.netloc or parsed.query:raise ValueError('External EPUB member reference')
    path=unquote(parsed.path)
    if not path or path.startswith('/') or '\\' in path or ':' in path:
        raise ValueError('Invalid EPUB member reference')
    name=posixpath.normpath(posixpath.join(base,path))
    if name in ('.','..') or name.startswith('../'):
        raise ValueError('EPUB member reference leaves the package')
    return name

def prepare_reader(epub:Path,destination:Path,spec:dict):
    root=destination/'reader';entries=[];opf=None;spine=[]
    with zipfile.ZipFile(epub) as archive:
        container=ET.fromstring(archive.read('META-INF/container.xml'))
        package=container.find('.//{*}rootfile')
        if package is None:raise ValueError('Missing EPUB rootfile')
        opf=member_path('',package.attrib['full-path'])
        for name in archive.namelist():
            if archive.getinfo(name).is_dir() or name=='mimetype':continue
            if '..' in Path(name).parts or name.startswith('/') or ':' in name:raise ValueError('Invalid generated EPUB member')
            body=archive.read(name);file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(body)
            entries.append({'path':'reader/'+name,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
        if not opf:raise ValueError('Missing package document')
        xml=ET.fromstring(archive.read(opf));ns={'o':'http://www.idpf.org/2007/opf'}
        opf_dir=posixpath.dirname(opf)
        manifest=xml.findall('o:manifest/o:item',ns)
        items={e.attrib['id']:member_path(opf_dir,e.attrib['href']) for e in manifest}
        nav_paths={items[e.attrib['id']] for e in manifest if 'nav' in e.get('properties','').split()}
        for ref in xml.findall('o:spine/o:itemref',ns):
            spine.append({'path':items[ref.attrib['idref']],'linear':ref.get('linear')!='no'})
        spine_index=list(xml).index(xml.find('o:spine',ns))
    script=Path(__file__).with_name('reader_index.cjs')
    run=subprocess.run(['node',str(script)],input=json.dumps({'root':str(root),'spine':spine,'spineNodeIndex':spine_index}),capture_output=True,text=True,encoding='utf-8',check=True)
    index=json.loads(run.stdout)
    known={row['path']:row for row in entries}
    descriptor={'format':'epub-stream@1','epubSha256':hashlib.sha256(epub.read_bytes()).hexdigest(),'package':known['reader/'+opf],'entries':entries,'locations':index['locations'],'locationScheme':index['version']}
    identities=sorted([[e['path'].removeprefix('reader/'),e['bytes'],e['sha256']] for e in entries],key=lambda row:row[0].encode('utf-8'))
    descriptor['contentDigest']=hashlib.sha256(json.dumps(identities,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
    spine_paths={row['path'] for row in spine}
    mapping={}
    for chapter in spec['chapters']:
        path=member_path(opf_dir,chapter.get('href',f'chapter-{chapter["id"]}.xhtml'))
        if path not in spine_paths:raise ValueError('Declared chapter absent from EPUB spine: '+chapter['id'])
        member='reader/'+path
        if member not in known:raise ValueError('Declared chapter missing from EPUB: '+chapter['id'])
        mapping[chapter['id']]=known[member]
    chapter_paths={item['path'] for item in mapping.values()}
    descriptor['support']=[row for row in entries if row['path'] not in chapter_paths and (row['path'].removeprefix('reader/') in nav_paths or Path(row['path']).suffix in ('.opf','.ncx','.css','.woff','.woff2','.ttf','.otf') or Path(row['path']).name in ('sources.xhtml','resources.xhtml'))]
    dependencies={}
    for key,item in mapping.items():
        document=ET.parse(destination/item['path']);found=[]
        for element in document.iter():
            if element.tag.rsplit('}',1)[-1]!='img':continue
            href=element.get('src','')
            if href.startswith('data:'):continue
            target='reader/'+member_path(posixpath.dirname(item['path'].removeprefix('reader/')),href)
            if target not in known:raise ValueError('Unregistered EPUB illustration: '+href)
            if known[target] not in found:found.append(known[target])
        dependencies[key]=found
    descriptor['chapterDependencies']=dependencies
    return descriptor,mapping,[root/e['path'].removeprefix('reader/') for e in entries]
