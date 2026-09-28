"""Preview or deploy only the maintained Comfortable Reader skill files."""
from pathlib import Path
import argparse,hashlib,json,os,shutil
ROOT=Path(__file__).resolve().parents[1]/'skill/comfortable-reader'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
def members():
 for name in ('SKILL.md','requirements-authoring.txt'):
  if (ROOT/name).is_file():yield ROOT/name
 for folder in ('references','scripts','assets','agents'):
  for path in sorted((ROOT/folder).rglob('*')):
   if '__pycache__' in path.parts or path.suffix in ('.pyc','.pyo'):continue
   if path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise ValueError('技能主本包含链接，停止同步')
   if path.is_file():yield path
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',type=Path,required=True);p.add_argument('--backup',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
 target=a.target.resolve();backup=a.backup.resolve()
 if target==ROOT.resolve() or backup==target or backup.is_relative_to(target):raise ValueError('目标或备份目录重叠')
 rows=[]
 for source in members():
  relative=source.relative_to(ROOT);dest=target/relative
  for parent in [dest,*list(dest.parents)[:len(relative.parts)]]:
   if parent.is_symlink() or getattr(parent,'is_junction',lambda:False)():raise ValueError('目标包含链接，停止同步')
  rows.append({'path':relative.as_posix(),'source_sha256':digest(source),'before_sha256':digest(dest)})
 changed=[r for r in rows if r['source_sha256']!=r['before_sha256']]
 if a.apply:
  if backup.exists() and any(backup.iterdir()):raise ValueError('备份目录须为空，避免覆盖旧备份')
  backup.mkdir(parents=True,exist_ok=True)
  for row in changed:
   relative=Path(row['path']);source=ROOT/relative;dest=target/relative
   if digest(source)!=row['source_sha256'] or digest(dest)!=row['before_sha256']:raise ValueError('同步期间文件改变，停止以保护现场')
   if dest.exists():saved=backup/relative;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(dest,saved)
   dest.parent.mkdir(parents=True,exist_ok=True);temporary=dest.with_suffix(dest.suffix+'.syncing');shutil.copyfile(source,temporary);os.replace(temporary,dest)
  if any(digest(target/row['path'])!=row['source_sha256'] for row in rows):raise ValueError('部署后的文件校验未通过')
 report={'mode':'applied' if a.apply else 'plan','files':len(rows),'changed':changed,'vendor_and_private_config_preserved':True}
 if a.apply:(backup/'sync-receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except (ValueError,OSError) as error:raise SystemExit(str(error))
