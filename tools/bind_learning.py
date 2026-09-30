"""Inspect or bind a declared local learning package. Never install or run book code.

The default prints a concrete plan. --apply links reading materials only unless
--trust-reviewed-code is also supplied for explicitly selected activities.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skill/comfortable-reader/scripts'))
from learning_pack import relative_file,epub_identity,sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--pack',required=True,type=Path);p.add_argument('--root',required=True,type=Path);p.add_argument('--derived-root',type=Path)
 identity=p.add_mutually_exclusive_group(required=True);identity.add_argument('--epub',type=Path);identity.add_argument('--manifest',type=Path)
 p.add_argument('--activity',action='append',default=[]);p.add_argument('--python',default=sys.executable)
 p.add_argument('--app-id',default='com.comfortablereader.desktop');p.add_argument('--state-dir',type=Path);p.add_argument('--learning-dir',type=Path)
 p.add_argument('--apply',action='store_true');p.add_argument('--trust-reviewed-code',action='store_true')
 a=p.parse_args()
 if not a.app_id.replace('.','').replace('-','').isalnum():raise ValueError('应用标识无效')
 state_dir=a.state_dir or Path(os.environ.get('APPDATA',Path.home()/'.local/share'))/a.app_id
 state=json.loads((state_dir/'reader-state.json').read_text(encoding='utf-8'))
 package=a.pack.resolve(strict=True);raw=package.read_bytes()
 if len(raw)>16*1024*1024:raise ValueError('学习包超过读取范围')
 pack=json.loads(raw);book_uuid=pack['book_uuid']
 def same_registered_file(book):
  if not a.epub or not isinstance(book.get('path'),str):return False
  try:return Path(book['path']).samefile(a.epub)
  except OSError:return False
 matches=[book for book in state['books'] if book.get('bookUuid')==book_uuid or (not book.get('bookUuid') and same_registered_file(book))]
 if len(matches)!=1:raise ValueError('先在目标阅读器打开这本书；需要唯一、已核对的书籍身份，不能按标题猜测')
 book=matches[0]
 if a.epub:
  original=epub_identity(a.epub)
  if original['sha256']!=pack['book_revision_sha256'] or book_uuid not in original['identifiers']:raise ValueError('学习包与 EPUB 身份不一致')
  manifest_hash=None
 else:
  manifest_hash=sha(a.manifest);manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
  if manifest_hash!=(book.get('catalogSource') or {}).get('sha256') or manifest['id']!=book_uuid or manifest.get('reader',{}).get('epubSha256')!=pack['book_revision_sha256']:raise ValueError('清单与阅读器当前连接的内容版本不一致')
 activities={item['id']:item for item in pack.get('activities',[])};assets={item['id']:item for item in pack['assets']}
 if len(set(a.activity))!=len(a.activity):raise ValueError('活动选择重复')
 if a.trust_reviewed_code and not a.activity:raise ValueError('执行批准必须明确选择活动，不自动批准全书')
 plans=[];approved={};needed=set();requirements=set()
 for key in a.activity:
  activity=activities.get(key)
  if not activity:raise ValueError('这本书没有所选活动：'+key)
  if 'run_definition_json' not in activity:raise ValueError('此旧学习包尚未声明可迁移的运行定义；请先由作者补足，不按文件名猜执行方式')
  definition_text=activity['run_definition_json']
  if len(definition_text.encode())>65536:raise ValueError('运行定义超过预算')
  definition=json.loads(definition_text)
  if definition.get('schemaVersion')!=1 or definition.get('kind')!='python-cli@1':raise ValueError('不支持的运行定义')
  if definition['entry_asset']!=activity['entry_asset']:raise ValueError('显示源码与运行入口不一致')
  if any(field in definition for field in ('command','shell','cwd','env','python_path','trusted')):raise ValueError('书籍不能声明本机执行权')
  ids=list(dict.fromkeys([definition['entry_asset'],*definition.get('input_assets',[]),*([definition['adapter_asset']] if definition.get('adapter_asset') else [])]))
  needed.update(ids);requirements.update(definition.get('requires',[]))
  for resource_id in ids:
   resource=assets[resource_id];root=package.parent if resource.get('root')=='profile' else a.derived_root if resource.get('root')=='derived' else a.root
   if root is None:raise ValueError('所选活动需要明确的派生资源根')
   file=relative_file(root,resource['relative_path'])
   if file.stat().st_size!=resource['bytes'] or sha(file)!=resource['sha256']:raise ValueError('所选活动资源字节不一致：'+resource_id)
  definition_hash=hashlib.sha256(definition_text.encode('utf-8')).hexdigest()
  approved[key]={'definition_sha256':definition_hash,'assets_sha256':{resource_id:assets[resource_id]['sha256'] for resource_id in ids},'reviewed_pure_local':True}
  plans.append({'activity':key,'title':activity['title'],'definition_sha256':definition_hash,'files':len(ids),'maximum_seconds':definition.get('timeout_seconds'),'requires_separate_budget':bool(definition.get('requires_budget_approval'))})
 python=Path(shutil.which(a.python) or a.python).resolve(strict=True)
 probe="import json,sys,importlib.metadata as m; names=json.loads(sys.argv[1]); found={};\nfor name in names:\n try: found[name]=m.version(name)\n except m.PackageNotFoundError: found[name]=None\nprint(json.dumps({'python':sys.version,'packages':found}))"
 if any(not isinstance(name,str) or len(name)>100 or not all(c.isalnum() or c in '-_.' for c in name) for name in requirements):raise ValueError('依赖名称无效')
 environment=json.loads(subprocess.run([str(python),'-I','-X','utf8','-c',probe,json.dumps(sorted(requirements))],check=True,capture_output=True,text=True,encoding='utf-8',timeout=15).stdout)
 missing=[name for name,version in environment['packages'].items() if version is None]
 report={'mode':'apply' if a.apply else 'inspect_only','book_uuid':book_uuid,'book_id':book['id'],'activities':plans,'selected_resource_bytes':sum(assets[key]['bytes'] for key in needed),'environment':environment,'missing_dependencies':missing,'book_code_executed':False,'dependencies_installed':False,'native_mode':'current-user file and network privileges; not an OS sandbox'}
 if a.apply:
  if missing and a.trust_reviewed_code:raise ValueError('缺少依赖：'+', '.join(missing)+'；未安装、未运行，也未授予本次执行批准')
  learning_dir=a.learning_dir or (Path.home()/'.comfortable-reader' if a.app_id=='com.comfortablereader.desktop' else state_dir)
  config_path=learning_dir/'learning-bindings.json';config=json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {'schema_version':1,'books':{},'environments':{}}
  python_hash=sha(python);environment_id='python-'+python_hash[:16]
  selected={'package':str(package),'package_sha256':hashlib.sha256(raw).hexdigest(),'book_uuid':book_uuid,'epub_sha256':pack['book_revision_sha256'],'content_root':str(a.root.resolve(strict=True)),'recipes':config.get('books',{}).get(book['id'],{}).get('recipes',{})}
  if a.derived_root:selected['derived_root']=str(a.derived_root.resolve(strict=True))
  if manifest_hash:selected['catalog_manifest_sha256']=manifest_hash
  if a.trust_reviewed_code:
   for value in approved.values():value['runtime']=environment_id
   selected['recipes']={**selected['recipes'],**approved}
   config.setdefault('environments',{})[environment_id]={'python':str(python),'sha256':python_hash,'checked':environment}
  config.setdefault('books',{})[book['id']]=selected
  learning_dir.mkdir(parents=True,exist_ok=True)
  if config_path.exists():shutil.copyfile(config_path,learning_dir/f'learning-bindings.before-{time.time_ns()}.json')
  temporary=config_path.with_suffix('.writing');temporary.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temporary,config_path)
  report['linked']=True;report['execution_grants_written']=list(approved) if a.trust_reviewed_code else []
 print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:print('绑定停止：'+str(error),file=sys.stderr);raise SystemExit(2)
