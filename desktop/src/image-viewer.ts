export async function mountImage(parent: HTMLElement, url: string, title: string, saved: any,
  onChange: (view: {zoom: number; x: number; y: number}) => void): Promise<() => void> {
  parent.innerHTML='<div class="study-image-controls"><button class="image-fit">适配窗口</button><button class="image-in">放大</button><button class="image-out">缩小</button><span class="image-scale" role="status"></span></div><div class="study-image-viewport" tabindex="0" aria-label="原图，可用方向键滚动；放大后拖动查看"><img class="study-original-image"/></div><p class="study-code-guide">放大后可拖动或用方向键查看。比例相对于适配窗口的大小；原件没有被缩小保存或重新编码。</p>';
  const viewport=parent.querySelector<HTMLElement>('.study-image-viewport')!,img=parent.querySelector<HTMLImageElement>('img')!,label=parent.querySelector<HTMLElement>('.image-scale')!;
  img.alt=title;img.src=url;img.draggable=false;await img.decode();
  let zoom=Math.min(8,Math.max(1,Number(saved?.zoom)||1)),x=Number(saved?.x)||.5,y=Number(saved?.y)||.5,disposed=false;
  const control=new AbortController(),options={signal:control.signal};
  function remember(){x=(viewport.scrollLeft+viewport.clientWidth/2)/Math.max(1,img.width);y=(viewport.scrollTop+viewport.clientHeight/2)/Math.max(1,img.height);onChange({zoom,x,y});}
  function draw(){if(disposed)return;const width=Math.min(img.naturalWidth,Math.max(100,viewport.clientWidth-24));img.style.width=`${Math.round(width*zoom)}px`;img.style.maxWidth='none';label.textContent=`${Math.round(zoom*100)}% · 原图 ${img.naturalWidth} × ${img.naturalHeight}`;viewport.scrollLeft=x*img.width-viewport.clientWidth/2;viewport.scrollTop=y*img.height-viewport.clientHeight/2;}
  parent.querySelector('.image-fit')!.addEventListener('click',()=>{zoom=1;x=y=.5;draw();remember();},options);
  parent.querySelector('.image-in')!.addEventListener('click',()=>{zoom=Math.min(8,zoom*1.5);draw();remember();},options);
  parent.querySelector('.image-out')!.addEventListener('click',()=>{zoom=Math.max(1,zoom/1.5);draw();remember();},options);
  let drag:{x:number;y:number;left:number;top:number}|null=null;
  viewport.addEventListener('pointerdown',e=>{if(e.button!==0||zoom===1)return;viewport.focus();viewport.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,left:viewport.scrollLeft,top:viewport.scrollTop};e.preventDefault();},options);
  viewport.addEventListener('pointermove',e=>{if(drag){viewport.scrollLeft=drag.left+drag.x-e.clientX;viewport.scrollTop=drag.top+drag.y-e.clientY;}},options);
  viewport.addEventListener('pointerup',()=>{drag=null;remember();},options);viewport.addEventListener('pointercancel',()=>{drag=null;},options);
  viewport.addEventListener('scroll',remember,options);
  const observer=new ResizeObserver(draw);observer.observe(viewport);draw();
  return()=>{disposed=true;control.abort();observer.disconnect();};
}
