/** Read-only PDF page rendering. PDF actions, forms and document JavaScript are not run. */
export async function mountPdf(container:HTMLElement,bytes:ArrayBuffer,title:string,onError:(e:unknown)=>void):Promise<()=>void>{
  let disposed=false,document:any,renderTask:any;
  const moduleURL=new URL('./vendor/pdfjs/pdf.mjs',location.href).href;
  const pdfjs:any=await import(/* @vite-ignore */ moduleURL);
  pdfjs.GlobalWorkerOptions.workerSrc=new URL('./vendor/pdfjs/pdf.worker.min.mjs',location.href).href;
  container.innerHTML='<div class="study-pdf-controls"><button class="pdf-prev" type="button">← 上一页</button><label>PDF 索引页 <input class="pdf-page" type="number" min="1" value="1" aria-label="原 PDF 索引页码"/></label><span class="pdf-total"></span><button class="pdf-next" type="button">下一页 →</button><label>放大 <select class="pdf-zoom"><option value="0.75">75%</option><option value="1" selected>100%</option><option value="1.5">150%</option><option value="2">200%</option></select></label></div><p class="study-pdf-status" role="status">正在读取原 PDF…</p><div class="study-pdf-page-area"><canvas class="study-pdf-canvas" role="img"></canvas></div><details class="study-details"><summary>原件文字层（如有，供复制）</summary><p>文字层不代替原页排版，不代表另行完成了 OCR 或语义重建。</p><pre class="study-pdf-text"></pre></details>';
  // Historical scans can contain a 33-megapixel bilevel source image. Limit the
  // displayed canvas separately; a 16-MP source limit silently drops those pages.
  const loadingTask=pdfjs.getDocument({data:new Uint8Array(bytes),isEvalSupported:false,useWasm:false,enableXfa:false,stopAtErrors:true,disableAutoFetch:true,disableStream:true,maxImageSize:64*1024*1024,canvasMaxAreaInBytes:64*1024*1024,cMapUrl:new URL('./vendor/pdfjs/cmaps/',location.href).href,cMapPacked:true,standardFontDataUrl:new URL('./vendor/pdfjs/standard_fonts/',location.href).href,wasmUrl:new URL('./vendor/pdfjs/wasm/',location.href).href});
  const dispose=()=>{disposed=true;renderTask?.cancel();void loadingTask.destroy().catch(()=>{});};
  let pageNumber=1,revision=0,lastPage:any=null;
  const pageInput=container.querySelector<HTMLInputElement>('.pdf-page')!;
  const previous=container.querySelector<HTMLButtonElement>('.pdf-prev')!,next=container.querySelector<HTMLButtonElement>('.pdf-next')!;
  const status=container.querySelector<HTMLElement>('.study-pdf-status')!;
  const render=async(requested:number)=>{
    if(disposed||!document)return;const request=++revision;pageNumber=Math.max(1,Math.min(document.numPages,Math.floor(requested)||1));
    renderTask?.cancel();pageInput.value=String(pageNumber);previous.disabled=pageNumber===1;next.disabled=pageNumber===document.numPages;status.textContent=`正在呈现原 PDF 第 ${pageNumber} 页…`;
    try{
      const page=await document.getPage(pageNumber);if(disposed||request!==revision)return;
      const zoom=Number(container.querySelector<HTMLSelectElement>('.pdf-zoom')!.value);
      const natural=page.getViewport({scale:1});const width=Math.max(250,Math.min(1000,container.clientWidth-24))*zoom;
      const viewport=page.getViewport({scale:width/natural.width});const ratio=Math.min(2,devicePixelRatio,Math.sqrt(16000000/(viewport.width*viewport.height)));
      const canvas=container.querySelector<HTMLCanvasElement>('.study-pdf-canvas')!;
      canvas.width=Math.ceil(viewport.width*ratio);canvas.height=Math.ceil(viewport.height*ratio);canvas.style.width=`${viewport.width}px`;canvas.style.height=`${viewport.height}px`;canvas.setAttribute('aria-label',`${title}，PDF 索引第 ${pageNumber} 页`);
      renderTask=page.render({canvas,canvasContext:canvas.getContext('2d')!,viewport,transform:ratio===1?undefined:[ratio,0,0,ratio,0,0],background:'rgb(255,255,255)'});
      await renderTask.promise;if(disposed||request!==revision)return;
      const text=await page.getTextContent();if(disposed||request!==revision)return;
      const content=text.items.map((item:any)=>typeof item.str==='string'?item.str:'').join(' ');
      container.querySelector<HTMLElement>('.study-pdf-text')!.textContent=content||'此页没有可提取的文字层；以上方原刊影印为准。';
      status.textContent=`原件索引 ${pageNumber} / ${document.numPages} · 与印刷页码分开 · 禁止文档脚本与交互表单`;
      if(lastPage&&lastPage!==page)lastPage.cleanup();lastPage=page;
    }catch(error){if(!disposed&&request===revision&&(error as any)?.name!=='RenderingCancelledException'){status.textContent='这一页未能呈现，原件未改变。';onError(error);}}
  };
  try{document=await loadingTask.promise;if(disposed)return dispose;pageInput.max=String(document.numPages);container.querySelector('.pdf-total')!.textContent=`/ ${document.numPages}`;await render(1);}catch(error){dispose();throw error;}
  previous.addEventListener('click',()=>void render(pageNumber-1));next.addEventListener('click',()=>void render(pageNumber+1));pageInput.addEventListener('change',()=>void render(Number(pageInput.value)));container.querySelector('.pdf-zoom')!.addEventListener('change',()=>void render(pageNumber));
  return dispose;
}
