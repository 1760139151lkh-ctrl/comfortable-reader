"""Connect an existing, byte-preserved EPUB and declared learning pack locally.

Explicitly private: this is a compatibility/reading preview, not a release tool.
It copies only declared assets and exposes them on demand. Runtime bindings,
personal records and code-execution grants are never copied.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile,xml.etree.ElementTree as ET,mimetypes,re
from PIL import Image
from streamed_epub import prepare_reader
from book_format import safe_relative,BookError,file_identity
from reader_build_receipt import verify_reader_dist, RECEIPT

def main():
 p=argparse.ArgumentParser();p.add_argument('--epub',type=Path,required=True);p.add_argument('--learning-pack',type=Path,required=True);p.add_argument('--content-root',type=Path,required=True);p.add_argument('--derived-root',type=Path);p.add_argument('--shell',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--slug',default='existing-book');p.add_argument('--revision',required=True);a=p.parse_args()
 if a.out.exists():raise BookError('请使用新的空预览目录，避免覆盖已审阅内容')
 if not re.fullmatch(r'\d+\.\d+\.\d+',a.revision):raise BookError('阅读清单修订使用三段数字，如 0.4.0；原 EPUB 身份另由实际字节哈希固定')
 pack=json.loads(a.learning_pack.read_text(encoding='utf-8'));epub_hash=file_identity(a.epub)['sha256']
 if pack['book_revision_sha256']!=epub_hash:raise BookError('增强包不是这份 EPUB 的版本')
 shell_receipt=verify_reader_dist(a.shell)
 a.out.mkdir(parents=True)
 for name in ('index.html','sw.js','assets','vendor','third-party.html',RECEIPT):
  source=a.shell/name
  if source.is_dir():shutil.copytree(source,a.out/name)
  elif source.is_file():shutil.copyfile(source,a.out/name)
 # An existing build may contain example content. The resulting private catalogue
 # names exactly the requested book; it does not silently promote the examples.
 destination=a.out/'books'/a.slug;destination.mkdir(parents=True,exist_ok=False)
 chapters=[{**c,'id':c['id'].lower()} for c in pack['chapters']]
 reader,mapping,_=prepare_reader(a.epub,destination,{'chapters':chapters})
 resources=[]
 for asset in pack['assets']:
  root=a.learning_pack.parent if asset.get('root')=='profile' else a.derived_root if asset.get('root')=='derived' else a.content_root
  if root is None:raise BookError('增强包包含派生材料，需要明确指定其根目录')
  source=safe_relative(root,asset['relative_path']);identity=file_identity(source)
  if identity!={k:asset[k] for k in ('sha256','bytes')}:raise BookError('增强包资源字节已改变：'+asset['id'])
  target=destination/'assets'/asset['id']/source.name;target.parent.mkdir(parents=True,exist_ok=True)
  resource={'id':asset['id'],'logicalPath':asset['relative_path'],'path':target.relative_to(destination).as_posix(),**identity,'title':asset['title'],'description':asset['title'],'kind':asset['kind'],'role':asset.get('role','author_recorded'),'chapters':[c.lower() for c in asset['chapters']],'minimumUnit':'完整文件','rights':'private_preview_only','mediaType':mimetypes.guess_type(source.name)[0] or 'application/octet-stream'}
  if identity['bytes']>4*1024*1024:
   resource.pop('path');resource['chunks']=[]
   with source.open('rb') as stream:
    index=0
    while data:=stream.read(4*1024*1024):
     part=destination/'chunks'/asset['id']/f'{index:05d}.part';part.parent.mkdir(parents=True,exist_ok=True);part.write_bytes(data)
     resource['chunks'].append({'path':part.relative_to(destination).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
  else:shutil.copyfile(source,target)
  if asset['kind']=='image':
   with Image.open(source) as image:
    if image.format not in ('PNG','JPEG','WEBP'):raise BookError('图片格式未适配：'+source.name)
    resource.update(mediaType={'PNG':'image/png','JPEG':'image/jpeg','WEBP':'image/webp'}[image.format],width=image.width,height=image.height)
  if asset['kind']=='audio':resource['mediaType']='audio/wav'
  resources.append(resource)
 # Preserve legacy teaching metadata as a separately verified, inert adapter file.
 # All path bindings, executable grants and personal outputs remain outside it.
 for c in pack['chapters']:c['id']=c['id'].lower()
 for r in pack['assets']:r['chapters']=[c.lower() for c in r['chapters']]
 for act in pack['activities']:act['chapter']=act['chapter'].lower();act['registered']=False
 for src in pack['source_claims']:
  if isinstance(src.get('chapter'),str):src['chapter']=src['chapter'].lower()
 study_file=destination/'learning.json';study_file.write_text(json.dumps(pack,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
 with zipfile.ZipFile(a.epub) as archive:
  opf=ET.fromstring(archive.read(reader['package']['path'].removeprefix('reader/')))
 metadata=opf.find('{*}metadata')
 if metadata is None:raise BookError('原 EPUB 缺少书籍元数据')
 dc='{http://purl.org/dc/elements/1.1/}'
 title=metadata.findtext(dc+'title')
 if not title:raise BookError('原 EPUB 缺少书名')
 author=metadata.findtext(dc+'creator') or ''
 language=metadata.findtext(dc+'language') or 'und'
 modified=next((item.text for item in metadata.findall('{*}meta') if item.get('property')=='dcterms:modified'),None)
 description=f'私人兼容预览：保留原 EPUB 字节，展示 {len(chapters)} 个已登记章节；材料按需打开，尚未取得公开发行授权。'
 book={'schemaVersion':1,'id':pack['book_uuid'],'slug':a.slug,'title':title,'author':author,'revision':a.revision,'language':language,'rights':{'status':'private_preview_only'},'description':description,'reader':reader,'studyDescriptor':{'path':'learning.json',**file_identity(study_file)},'chapters':[{**{k:v for k,v in c.items() if k not in ('assets','source_sha256')},'body':mapping[c['id']],'readingDocument':mapping[c['id']],'readingDependencies':reader['chapterDependencies'][c['id']],'essential':[]} for c in chapters],'resources':resources,'sources':[],'activities':[{'id':act['id'],'chapter':act['chapter'],'title':act['title'],'question':next((c['question'] for c in chapters if c['id']==act['chapter']),''),'capability':'native-python@1','required':act.get('required_assets',[act['entry_asset']]),'optional':[],'parameters':{}} for act in pack['activities']]}
 if modified and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ',modified):book['revisionModified']=modified
 manifest=destination/'manifest.json';manifest.write_text(json.dumps(book,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
 full=a.out/'ebooks'/f'{a.slug}-{a.revision}.epub';full.parent.mkdir(exist_ok=True);shutil.copyfile(a.epub,full)
 catalog={'schemaVersion':1,'status':'private_preview_only','books':[{'id':book['id'],'slug':a.slug,'revision':a.revision,'title':title,'author':author,'description':book['description'],'language':'zh-CN','chapters':[{'id':c['id'],'title':c['title']} for c in chapters],'manifest':{'path':manifest.relative_to(a.out).as_posix(),**file_identity(manifest)},'epub':{'path':full.relative_to(a.out).as_posix(),**file_identity(full)}}]}
 (a.out/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
 receipt={'status':'private_preview_only','epub_sha256':epub_hash,'epub_byte_equal':a.epub.read_bytes()==full.read_bytes(),'chapters':len(chapters),'declared_assets':len(resources),'asset_bytes':sum(r['bytes'] for r in resources),'personal_state_copied':False,'native_grants_copied':False,'reader_locations':len(reader['locations']),'reader_source_sha256':shell_receipt['sourceTreeSha256'],'reader_build_sha256':shell_receipt['buildTreeSha256'],'reader_verification':'source_and_files_only'}
 (a.out/'existing-book-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))

if __name__=='__main__':main()
