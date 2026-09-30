"""Plan or retrieve only one declared activity's material, without running it."""
from pathlib import Path
import argparse,hashlib,json,os,re,sys,urllib.request,urllib.parse

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args):raise ValueError('资源重定向，需由发行者更新固定地址')
OPENER=urllib.request.build_opener(NoRedirect)
def relative(base,path):
 if not isinstance(path,str) or any(c in path for c in '\\:%?#') or any(part in ('','.','..') or part.endswith(('.', ' ')) for part in path.split('/')):raise ValueError('资源地址不是安全相对路径')
 return urllib.parse.urljoin(base,'/'.join(urllib.parse.quote(part) for part in path.split('/')))
def fetch(url,identity=None,limit=2*1024*1024):
 parts=urllib.parse.urlsplit(url)
 if parts.username or parts.password or not(parts.scheme=='https' or parts.scheme=='http' and parts.hostname in ('127.0.0.1','localhost')):raise ValueError('仅允许无凭据的 HTTPS 来源或本机预览')
 with OPENER.open(urllib.request.Request(url,headers={'User-Agent':'ComfortableReader/0.5'}),timeout=30) as response:
  raw=bytearray()
  while chunk:=response.read(65536):
   raw.extend(chunk)
   if len(raw)>min(limit,identity['bytes'] if identity else limit):raise ValueError('资源超过声明大小')
 result=bytes(raw)
 if identity and (len(result)!=identity['bytes'] or hashlib.sha256(result).hexdigest()!=identity['sha256']):raise ValueError('资源内容与声明不符')
 return result
def safe_output(root,path):
 relative('https://example.invalid/',path)
 current=root
 for part in path.split('/'):
  if re.match(r'^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)',part,re.I):raise ValueError('设备路径无效')
  current=current/part
  if current.is_symlink() or getattr(current,'is_junction',lambda:False)():raise ValueError('输出包含符号链接或目录联接')
 if not current.resolve().is_relative_to(root.resolve()):raise ValueError('输出越过指定目录')
 return current
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',required=True);p.add_argument('--book',required=True);p.add_argument('--activity',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--download',action='store_true');p.add_argument('--cache',type=Path)
 a=p.parse_args();catalog=json.loads(fetch(a.catalog,limit=1024*1024));row=next((b for b in catalog['books'] if b['slug']==a.book or b['id']==a.book),None)
 if row is None:raise ValueError('书目中没有这本书')
 url=relative(a.catalog,row['manifest']['path']);raw=fetch(url,row['manifest']);manifest=json.loads(raw)
 activity=next((item for item in manifest['activities'] if item['id']==a.activity),None)
 if activity is None:raise ValueError('这本书没有所选活动')
 resources={item['id']:item for item in manifest['resources']};ids=[]
 def include(key,trail=()):
  if key in trail:raise ValueError('资源依赖循环')
  if key in ids:return
  resource=resources[key];ids.append(key)
  for dependency in resource.get('dependsOn',[]):include(dependency,(*trail,key))
 for key in activity['required']:include(key)
 output=a.out.resolve();marker=output/'.comfortable-activity-materials.json'
 if output.exists() and any(output.iterdir()):
  if not marker.is_file():raise ValueError('已有目录不是本工具的材料目录；请另选新目录')
  old=json.loads(marker.read_text(encoding='utf-8'))
  if old['manifest_sha256']!=row['manifest']['sha256']:raise ValueError('内容版本不同，请使用新目录，保留旧材料和修改')
 report={'mode':'download' if a.download else 'plan_only','book':manifest['title'],'activity':activity['title'],'resources':[{'id':key,'title':resources[key]['title'],'bytes':resources[key]['bytes']} for key in ids],'total_bytes':sum(resources[key]['bytes'] for key in ids),'code_executed':False,'environment_installed':False}
 if not a.download:print(json.dumps(report,ensure_ascii=False,indent=2));return
 output.mkdir(parents=True,exist_ok=True);cache=a.cache or output/'.chunks';cache.mkdir(parents=True,exist_ok=True)
 descriptor=manifest.get('studyDescriptor');pack_raw=fetch(relative(url,descriptor['path']),descriptor) if descriptor else None
 if pack_raw:
  pack=json.loads(pack_raw);assets={r['id']:r for r in pack['assets']}
 else:assets={key:{'relative_path':resources[key]['logicalPath'],'root':'publication'} for key in ids}
 # Marker is written before any material, so an interrupted request can resume.
 marker.write_text(json.dumps({'book_id':manifest['id'],'manifest_sha256':row['manifest']['sha256'],'activity':a.activity}),encoding='utf-8')
 downloaded=0;reused=0
 for key in ids:
  resource=resources[key];asset=assets[key];kind=asset.get('root','publication');destination=safe_output(output,kind+'/'+asset['relative_path'])
  if destination.is_file():
   if hashlib.sha256(destination.read_bytes()).hexdigest()==resource['sha256']:reused+=resource['bytes'];continue
   raise ValueError('已有材料已被修改，未覆盖：'+destination.name)
  destination.parent.mkdir(parents=True,exist_ok=True);pending=destination.with_suffix(destination.suffix+'.downloading');digest=hashlib.sha256();size=0
  with pending.open('wb') as stream:
   for unit in resource.get('chunks') or [resource]:
    digest_path=safe_output(cache,unit['sha256'])
    data=digest_path.read_bytes() if digest_path.is_file() else None
    if data is None or len(data)!=unit['bytes'] or hashlib.sha256(data).hexdigest()!=unit['sha256']:
     data=fetch(relative(url,unit['path']),unit,128*1024*1024);digest_path.write_bytes(data);downloaded+=len(data)
    else:reused+=len(data)
    digest.update(data);size+=len(data);stream.write(data)
  if size!=resource['bytes'] or digest.hexdigest()!=resource['sha256']:raise ValueError('完整资源校验失败，未登记就绪')
  os.replace(pending,destination)
 (output/'manifest.json').write_bytes(raw)
 if pack_raw:
  (output/'profile').mkdir(exist_ok=True);(output/'profile/book.crlearn').write_bytes(pack_raw)
 report.update(downloaded_bytes=downloaded,reused_bytes=reused,materials_ready=True,reading_content_saved=False)
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError) as error:print('准备停止：'+str(error),file=sys.stderr);raise SystemExit(2)
