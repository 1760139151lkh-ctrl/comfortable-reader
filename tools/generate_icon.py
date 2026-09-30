"""Regenerate the app's original, code-drawn book icon (requires Pillow)."""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'desktop'/'src-tauri'/'icons'
OUT.mkdir(parents=True,exist_ok=True)
S=1024
image=Image.new('RGBA',(S,S),(0,0,0,0))
d=ImageDraw.Draw(image)
d.rounded_rectangle((40,40,984,984),radius=214,fill='#17333D')
d.rounded_rectangle((56,56,968,968),radius=200,outline='#638187',width=13)
d.polygon([(194,273),(430,304),(501,355),(501,728),(430,684),(194,651)],fill='#EDE3D0')
d.polygon([(830,273),(594,304),(523,355),(523,728),(594,684),(830,651)],fill='#F7EFDE')
d.line([(194,273),(430,304),(501,355),(501,728)],fill='#C08D57',width=19,joint='curve')
d.line([(830,273),(594,304),(523,355),(523,728)],fill='#C08D57',width=19,joint='curve')
d.line([(512,354),(512,732)],fill='#BD8751',width=20)
for y,width in [(411,162),(472,187),(533,176),(594,147)]:
    d.line([(251,y),(251+width,y+18)],fill='#8B9B93',width=15)
    d.line([(773-width,y+18),(773,y)],fill='#8B9B93',width=15)
d.arc((178,690,846,808),start=5,end=175,fill='#C08D57',width=15)
d.ellipse((481,777,543,839),fill='#C08D57')
for size,name in [(32,'32x32.png'),(128,'128x128.png'),(256,'128x128@2x.png'),(512,'icon.png')]:
    image.resize((size,size),Image.Resampling.LANCZOS).save(OUT/name)
image.save(OUT/'icon.ico',format='ICO',sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
image.save(OUT/'icon.icns',format='ICNS')
print('generated',len(list(OUT.iterdir())),'icon artifacts')
