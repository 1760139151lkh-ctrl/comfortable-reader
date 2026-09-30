"""Audit an explicit public file allowlist; optionally export only those bytes."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,io,json,re,shutil,zipfile,tarfile
ROOT=Path(__file__).resolve().parents[1]
SECRET=[re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(rb'(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}'),re.compile(rb'AKIA[0-9A-Z]{16}')]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--allowlist',type=Path);p.add_argument('--private-term',action='append',default=[]);p.add_argument('--artifact',type=Path,action='append',default=[]);p.add_argument('--report',type=Path,required=True);p.add_argument('--export',type=Path);a=p.parse_args()
 root=a.root.resolve(strict=True);listing=json.loads((a.allowlist or root/'release-files.json').read_text(encoding='utf-8'))
 names=listing['files'];findings=[];rows=[];terms=[str(Path.home()),Path.home().as_posix(),*a.private_term]
 def inspect(raw,label):
  for term in terms:
   if term and any(term.encode(enc) in raw for enc in ('utf-8','utf-16-le')):findings.append({'file':label,'category':'local_identity_or_path'})
  if any(pattern.search(raw) for pattern in SECRET):findings.append({'file':label,'category':'credential_pattern'})
  if raw[:4]==b'PK\x03\x04':
   with zipfile.ZipFile(io.BytesIO(raw)) as z:
    if sum(i.file_size for i in z.infolist())>256*1024*1024:raise ValueError('归档超出审查预算：'+label)
    for member in z.infolist():
     if not member.is_dir():inspect(z.read(member),label+'!'+member.filename)
  elif raw[:2]==b'\x1f\x8b':
   with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
    if sum(member.size for member in archive.getmembers())>256*1024*1024:raise ValueError('归档超出审查预算：'+label)
    for member in archive.getmembers():
     if member.isfile():inspect(archive.extractfile(member).read(),label+'!'+member.name)
 if not isinstance(names,list) or len(names)!=len(set(names)) or len(names)>20000:raise ValueError('发布纳入清单无效')
 for name in names:
  rel=PurePosixPath(name)
  if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:raise ValueError('清单路径越界')
  path=root.joinpath(*rel.parts)
  for part in [path,*list(path.parents)[:len(rel.parts)]]:
   if part.is_symlink() or getattr(part,'is_junction',lambda:False)():raise ValueError('纳入文件包含链接：'+name)
  if not path.is_file() or not path.resolve().is_relative_to(root):raise ValueError('纳入文件不存在或越界：'+name)
  raw=path.read_bytes();inspect(raw,name);rows.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 artifacts=[]
 for path in a.artifact:
  raw=path.read_bytes();inspect(raw,'artifact:'+path.name);artifacts.append({'name':path.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 report={'status':'passed_patterns' if not findings else 'hold','allowlist_files':len(rows),'source_files':rows,'artifacts':artifacts,'findings':findings,'history_included':False,'rights_approved':False,'scope':'Positive files, UTF-8/UTF-16 local identity and credential patterns, nested ZIP/EPUB members; not a proof of rights or a complete secret detector.'}
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 if findings:raise ValueError('发现待处理项；报告只列类别和文件，没有打印敏感值')
 if a.export:
  out=a.export.resolve()
  if out==root or out.is_relative_to(root) or root.is_relative_to(out):raise ValueError('导出目录不能与源码重叠')
  if out.exists() and any(out.iterdir()):raise ValueError('使用新的空导出目录')
  for row in rows:
   source=root/row['path'];dest=out/row['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
   if hashlib.sha256(dest.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('导出期间内容改变')
 print(json.dumps({'status':report['status'],'files':len(rows),'artifacts':len(artifacts),'findings':len(findings)}))
if __name__=='__main__':
 try:main()
 except (OSError,ValueError,KeyError) as error:raise SystemExit('审查停止：'+str(error))
