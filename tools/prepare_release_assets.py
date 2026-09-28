"""Retrieve explicitly listed release inputs; never run model or book code."""
from pathlib import Path
import argparse, hashlib, json, os, urllib.request, urllib.parse
from book_format import BookError

ROOT = Path(__file__).resolve().parents[1]

class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or parsed.username or parsed.password:
            raise BookError('发行文件重定向到不安全地址')
        return super().redirect_request(request, fp, code, message, headers, url)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    lock = json.loads((ROOT / 'release-assets.json').read_text(encoding='utf-8'))
    rows=[]
    for item in lock['assets']:
        path = ROOT / item['path']
        if not path.resolve().is_relative_to(ROOT) or any(p.is_symlink() or getattr(p,'is_junction',lambda:False)() for p in [path,*path.parents] if p.is_relative_to(ROOT)):
            raise BookError('发行输入越过项目目录')
        if path.is_file():
            if path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise BookError('已有发行输入不同，保留文件并停止：'+item['path'])
            rows.append({'path':item['path'],'status':'verified','bytes':item['bytes']})
            continue
        rows.append({'path':item['path'],'status':'missing','bytes':item['bytes']})
        if not args.download:
            continue
        url=lock['baseUrl']+'/'+item['name']
        parsed=urllib.parse.urlsplit(url)
        if parsed.scheme!='https' or parsed.username or parsed.password:
            raise BookError('发行输入必须来自无凭据的 HTTPS 地址')
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.downloading')
        digest=hashlib.sha256();size=0
        try:
            with urllib.request.build_opener(HTTPSRedirect).open(url,timeout=60) as response, temporary.open('wb') as stream:
                while piece:=response.read(1024*1024):
                    size+=len(piece)
                    if size>item['bytes']:raise BookError('发行输入超过声明大小')
                    digest.update(piece);stream.write(piece)
            if size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
                raise BookError('发行输入字节身份不符，未替换目标文件')
            os.replace(temporary,path);rows[-1]['status']='downloaded_verified'
        except Exception:
            if temporary.is_file():temporary.unlink()
            raise
    print(json.dumps({'mode':'download' if args.download else 'plan','assets':rows,'code_executed':False},ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError) as error:raise SystemExit('准备停止：'+str(error))
