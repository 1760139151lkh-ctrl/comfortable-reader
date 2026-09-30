"""Create an original short video from this sample book's actual Python trace.

Requires ffmpeg only when regenerating the optional sample media. Readers do
not need it to read the chapter or use the browser experiment.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BOOK=ROOT/'books/measurement-lab'
FRAME_DIR=ROOT.parent/'private-audit/generated-frames'
OUT=BOOK/'media/fit-history.webm'
WIDTH,HEIGHT=640,360
FRAME_DIR.mkdir(parents=True,exist_ok=True)
OUT.parent.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('portable_demo_fit',BOOK/'code/fit_line.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=module.load_rows(BOOK/'data/three-points.csv')
trace=module.fit(rows,.12,30)['trace']
def draw(index,w,b):
    canvas=bytearray(bytes((246,241,232))*WIDTH*HEIGHT)
    def dot(x,y,color):
        if 0<=x<WIDTH and 0<=y<HEIGHT:
            at=3*(y*WIDTH+x);canvas[at:at+3]=bytes(color)
    def line(x0,y0,x1,y1,color,thickness=1):
        count=max(abs(x1-x0),abs(y1-y0),1)
        for i in range(count+1):
            x=round(x0+(x1-x0)*i/count);y=round(y0+(y1-y0)*i/count)
            for dy in range(-thickness+1,thickness):
                for dx in range(-thickness+1,thickness):dot(x+dx,y+dy,color)
    xp=lambda x:round(83+(x+.25)/2.5*490)
    yp=lambda y:round(309-(y+.4)/6.4*255)
    for j in range(7):line(70,yp(j),580,yp(j),(219,211,195))
    for j in range(3):line(xp(j),44,xp(j),313,(219,211,195))
    line(70,309,580,309,(103,110,110),2);line(83,41,83,315,(103,110,110),2)
    line(xp(-.25),yp(w*-.25+b),xp(2.25),yp(w*2.25+b),(154,103,69),3)
    for x,y in rows:
        cx,cy=xp(x),yp(y)
        for dx in range(-6,7):
            for dy in range(-6,7):
                if dx*dx+dy*dy<=36:dot(cx+dx,cy+dy,(87,124,120))
    # A quiet progress line, not a fictional performance score.
    line(73,338,73+round(index/30*510),338,(154,103,69),4)
    frame=FRAME_DIR/f'frame-{index:03}.ppm'
    frame.write_bytes(f'P6\n{WIDTH} {HEIGHT}\n255\n'.encode('ascii')+canvas)

draw(0,0,0)
for item in trace:draw(item['step'],item['after']['w'],item['after']['b'])
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','8','-i',str(FRAME_DIR/'frame-%03d.ppm'),'-c:v','libvpx-vp9','-b:v','360k','-pix_fmt','yuv420p','-an',str(OUT)],check=True,capture_output=True,text=True)
data=OUT.read_bytes()
if not data.startswith(b'\x1a\x45\xdf\xa3'):raise ValueError('Unexpected output container')
receipt={'identity':'author_generated_preview_from_actual_python_trace','source_code_sha256':hashlib.sha256((BOOK/'code/fit_line.py').read_bytes()).hexdigest(),'input_sha256':hashlib.sha256((BOOK/'data/three-points.csv').read_bytes()).hexdigest(),'frames':31,'playback_fps':8,'duration_seconds':31/8,'file_sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'not_a_live_reader_run':True}
(ROOT.parent/'private-audit/demo-video-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
