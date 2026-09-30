// The worker's signal samples and spectrum pixels remain unchanged. Only the
// drawing surface follows the current CSS size, DPI and theme.
export function mountAudioPlots(scope:HTMLElement, wave:HTMLCanvasElement, spectrum:HTMLCanvasElement, value:any):()=>void {
  const raw=document.createElement('canvas');raw.width=value.width;raw.height=value.height;
  raw.getContext('2d')!.putImageData(new ImageData(value.pixels,value.width,value.height),0,0);
  const configure=(canvas:HTMLCanvasElement,height:number)=>{
    const width=Math.max(220,canvas.parentElement!.clientWidth),ratio=Math.min(devicePixelRatio||1,2);
    canvas.style.width='100%';canvas.style.height=height+'px';canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
    const ctx=canvas.getContext('2d')!;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
    ctx.font='13px "Segoe UI","Microsoft YaHei UI",sans-serif';return {ctx,width,height};
  };
  let disposed=false;
  const paint=()=>{
    if(disposed)return;
    const style=getComputedStyle(scope),ink=style.getPropertyValue('--text').trim(),accent=style.getPropertyValue('--accent').trim();
    const w=configure(wave,130);w.ctx.strokeStyle=accent;w.ctx.lineWidth=2;
    w.ctx.beginPath();
    for(let i=0;i<value.width;i++){const x=(i+.5)/value.width*w.width;w.ctx.moveTo(x,60-value.envelope[i*2+1]*48);w.ctx.lineTo(x,60-value.envelope[i*2]*48);}
    w.ctx.stroke();w.ctx.fillStyle=ink;w.ctx.fillText('0 s',4,124);w.ctx.textAlign='right';w.ctx.fillText(value.duration.toFixed(2)+' s',w.width-4,124);
    const s=configure(spectrum,230),left=64,top=12,plotWidth=s.width-left-10,plotHeight=176;
    s.ctx.drawImage(raw,left,top,plotWidth,plotHeight);s.ctx.fillStyle=ink;
    s.ctx.fillText(value.sampleRate/2+' Hz',2,25);s.ctx.fillText('0 Hz',2,188);s.ctx.fillText('0 s',left,219);s.ctx.textAlign='right';s.ctx.fillText(value.duration.toFixed(2)+' s',s.width-10,219);
    const cursor=spectrum.parentElement?.querySelector<HTMLElement>('.study-play-cursor');
    if(cursor){cursor.dataset.start=String(left/s.width*100);cursor.dataset.span=String(plotWidth/s.width*100);}
  };
  const observer=new ResizeObserver(paint);observer.observe(wave.parentElement!);observer.observe(spectrum.parentElement!);
  window.addEventListener('reader-theme-change',paint);paint();
  return()=>{disposed=true;observer.disconnect();window.removeEventListener('reader-theme-change',paint);};
}
