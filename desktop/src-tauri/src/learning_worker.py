"""Trusted process bootstrap, released only after the broker assigns its Job Object.

This is lifecycle management, not a Python sandbox. Edited code requires the
separate native high-trust grant. Published recipes and inputs are frozen first.
"""
import sys
if sys.stdin.readline().strip() != 'START':
    raise SystemExit('Broker did not release this process')
from pathlib import Path
import json, runpy, traceback, platform, hashlib, os, importlib.util, importlib.metadata, ast, math

root=Path(os.environ.get('CR_RUN_ROOT',str(Path.cwd())))
# A packaged parent may redirect AppData into a long MSIX path. Keep the
# extended Windows prefix all the way into authors' Path(__file__) operations.
if sys.platform=='win32' and not str(root).startswith('\\\\?\\'):
    root=Path('\\\\?\\'+str(root))
request=json.loads((root/'request.json').read_text(encoding='utf-8'))
entry=root/request['entry']
sys.path.insert(0,str(entry.parent))
sys.argv=[str(entry),*request['argv']]
for index,value in enumerate(sys.argv[:-1]):
    if value in ('--out-dir','--output-dir'):
        candidate=Path(sys.argv[index+1])
        sys.argv[index+1]=str(candidate if candidate.is_absolute() else root/candidate)
versions={}
for package in ('torch','numpy','tokenizers'):
    try:versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:pass
(root/'environment.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':versions,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'mode':'trusted_native_not_sandbox','recipe':request['activity_id']},ensure_ascii=False,indent=2),encoding='utf-8')

try:
    adapter_hash=request.get('adapter_sha256')
    if adapter_hash:
        adapter=root/'book-adapter.py'
        if hashlib.sha256(adapter.read_bytes()).hexdigest()!=adapter_hash:
            raise RuntimeError('Book adapter bytes changed before execution')
        spec=importlib.util.spec_from_file_location('reviewed_book_adapter',adapter)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run(request,root,entry)
    else:
        runpy.run_path(str(entry),run_name='__main__')
except BaseException:
    traceback.print_exc()
    raise
