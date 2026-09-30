"""Serve a built reader on loopback only; never expose source or personal data roots."""
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import argparse
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('site',type=Path);p.add_argument('--port',type=int,default=4173);a=p.parse_args();root=a.site.resolve(strict=True)
 if not (root/'index.html').is_file() or not (root/'catalog.json').is_file():raise ValueError('请选择已构建的站点目录，不要提供源码或个人数据目录')
 class Handler(SimpleHTTPRequestHandler):
  def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(root),**kwargs)
  def do_GET(self):
   if self.headers.get('Host') not in (f'127.0.0.1:{a.port}',f'localhost:{a.port}'):self.send_error(403);return
   super().do_GET()
  def end_headers(self):
   self.send_header('Access-Control-Allow-Origin','http://tauri.localhost');self.send_header('Cache-Control','no-cache');super().end_headers()
 print(f'仅本机预览：http://127.0.0.1:{a.port}/',flush=True)
 ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
