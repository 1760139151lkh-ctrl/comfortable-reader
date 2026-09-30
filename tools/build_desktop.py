"""Build the existing desktop identity with local build paths remapped in output."""
from pathlib import Path
import argparse,os,shutil,subprocess,sys
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qa',action='store_true');p.add_argument('--no-bundle',action='store_true');p.add_argument('--target-dir',type=Path);a=p.parse_args()
 if sys.platform!='win32':raise ValueError('此打包入口目前只在 Windows 验证；网页构建不受此限制')
 root=Path(__file__).resolve().parents[1];desktop=root/'desktop';env=os.environ.copy()
 if env.get('RUSTFLAGS') and not env.get('CARGO_ENCODED_RUSTFLAGS'):raise ValueError('已设置自定义 RUSTFLAGS，请先用 CARGO_ENCODED_RUSTFLAGS 明确保留这些选项；不会擅自覆盖')
 flags=[]
 for source,target in [(Path.home(),'/build/user'),(Path.home()/'.cargo','/cargo'),(desktop,'/reader')]:
  flags.extend(['--remap-path-prefix='+str(source)+'='+target,'--remap-path-prefix='+source.as_posix()+'='+target])
 flags.extend(['-C','debuginfo=0','-C','link-arg=/PDBALTPATH:comfortable-reader.pdb'])
 env['CARGO_ENCODED_RUSTFLAGS']='\x1f'.join(([env['CARGO_ENCODED_RUSTFLAGS']] if env.get('CARGO_ENCODED_RUSTFLAGS') else [])+flags)
 if a.target_dir:env['CARGO_TARGET_DIR']=str(a.target_dir.resolve())
 npm=shutil.which('npm.cmd')
 if not npm:raise ValueError('未找到 npm，请先准备 Node.js')
 command=[npm,'run','tauri','--','build']
 if a.qa:command.extend(['--config','src-tauri/tauri.qa.json'])
 if a.no_bundle:command.append('--no-bundle')
 subprocess.run(command,cwd=desktop,env=env,check=True)
if __name__=='__main__':
 try:main()
 except (ValueError,OSError,subprocess.CalledProcessError) as error:print('构建停止：'+str(error),file=sys.stderr);raise SystemExit(2)
