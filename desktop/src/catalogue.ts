import {invoke,isDesktop} from './platform';
import {fetchCatalog,portableBooks,publicStore,escapePortableText as esc,type CatalogRecord} from './portable-books';
import {planSelection,acquire,resourceUnits,fetchVerified,materialize,sha256} from '../../site/core.mjs';
import './catalogue.css';
import {legacyRecords,mergeLegacyRecords,type LegacyRecords} from './legacy-records';
import {readableError} from './reader-errors';

type Host={flushPersonal:()=>Promise<void>;livePersonal:(bookId:string)=>any;current:()=>{id:string;href:string;title?:string}|null;known:(uuid:string)=>{record:any;sourceHash?:string|null}|null;openExisting:(id:string,chapter?:string)=>Promise<void>;open:(record:CatalogRecord,chapter?:string,replaceExisting?:boolean)=>Promise<void>;toast:(message:string)=>void;importPersonal:(value:any,restorePosition:boolean)=>Promise<void>};
let host:Host;
let catalogPanel:HTMLDialogElement;
let filesPanel:HTMLDialogElement;
let catalogRows:CatalogRecord[]=[];
let downloadAbort:AbortController|null=null;
let catalogAbort:AbortController|null=null;
let catalogGeneration=0;
let filesGeneration=0;
let projectLinks:{repository?:string;download?:string}={};
const bytes=(n:number)=>n>=1048576?`${(n/1048576).toFixed(1)} MB`:`${Math.ceil(n/1024)} KB`;
const button=(text:string,action:string,primary=false)=>`<button type="button" data-collection="${action}" class="${primary?'primary-action':'secondary-action'}">${text}</button>`;
function renderPanel(panel:HTMLDialogElement,html:string):void{
  const template=document.createElement('template');template.innerHTML=html;
  const header=template.content.firstElementChild?.tagName==='HEADER'?template.content.firstElementChild:null;
  const body=document.createElement('div');body.className='collection-body';
  if(header)header.remove();body.append(template.content);panel.replaceChildren(...(header?[header,body]:[body]));
  if(panel===filesPanel)filesGeneration++;
  if(panel.open)header?.querySelector<HTMLButtonElement>('button')?.focus({preventScroll:true});
}
function reportCollectionError(error:unknown):void{const status=filesPanel.querySelector<HTMLElement>('.collection-status');const text=(error instanceof Error?error.message:String(error)).replace(/^Error:\s*/,'');if(status&&filesPanel.open){status.textContent=text;status.classList.add('collection-error');status.setAttribute('role','alert');status.tabIndex=-1;status.scrollIntoView({block:'nearest'});status.focus({preventScroll:true});}else host.toast(text);}
function closeDialog(panel:HTMLDialogElement){panel.close();}
function makeDialog(name:string):HTMLDialogElement{
  const panel=document.createElement('dialog');panel.className='collection-panel';panel.setAttribute('aria-label',name);document.body.append(panel);
  panel.addEventListener('click',event=>{if(event.target===panel){const rect=panel.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)panel.close();}});
  panel.addEventListener('close',()=>{if(panel===catalogPanel){catalogAbort?.abort();catalogAbort=null;catalogGeneration++;}});return panel;
}
export async function initCatalogue(adapter:Host):Promise<void>{
  host=adapter;catalogPanel=makeDialog('发现书籍');filesPanel=makeDialog('离线保存与文件');
  const online=document.createElement('button');online.className='text-action discover-books';online.textContent='发现书籍';online.type='button';
  document.querySelector('.library-actions')?.before(online);online.addEventListener('click',()=>void showCatalogue());
  const files=document.createElement('button');files.className='text-button book-files-toggle';files.type='button';files.textContent='⋯';files.title='离线保存、电子书与学习记录';files.setAttribute('aria-label','打开书籍与保存选项');
  document.querySelector('.fullscreen-toggle')?.before(files);files.addEventListener('click',()=>void showBookFiles());
  if(!isDesktop){
    document.querySelectorAll<HTMLElement>('.add-books,.add-folder,.refresh-library').forEach(el=>el.hidden=true);
    const foot=document.querySelector('.drawer-footer');if(foot)foot.textContent='此浏览器保存 · 不自动跨端同步';
  }
}
export async function showCatalogue():Promise<void>{
  catalogAbort?.abort();const controller=new AbortController();catalogAbort=controller;const generation=++catalogGeneration;
  renderPanel(catalogPanel,`<header><div><span class="collection-eyebrow">舒适阅读书库</span><h1>从一章开始。</h1></div>${button('关闭','close')}</header><p class="collection-intro">直接阅读，在需要时取得材料。阅读方式和外观，随你选择。</p><div class="catalogue-content" role="status">正在读取书目…</div>`);
  catalogPanel.querySelector('[data-collection="close"]')!.addEventListener('click',()=>closeDialog(catalogPanel));
  if(!catalogPanel.open)catalogPanel.showModal();
  const content=catalogPanel.querySelector<HTMLElement>('.catalogue-content')!;
  const saved=localStorage.getItem('comfortable-reader-catalog-url');
  const url=isDesktop?saved:new URL('catalog.json',location.href).href;
  if(!url){renderCatalogAddress(content);return;}
  try{
    const fetched=await fetchCatalog(url,{signal:controller.signal});if(generation!==catalogGeneration||!catalogPanel.open)return;catalogRows=fetched.records;projectLinks=fetched.catalog.project??{};
    content.innerHTML=`<div class="catalogue-grid">${fetched.catalog.books.map((row:any,i:number)=>`<article class="catalogue-book"><div class="catalogue-cover" aria-hidden="true"><span>${String(i+1).padStart(2,'0')}</span><strong>${esc(row.title)}</strong></div><div><span class="collection-eyebrow">${row.example?'创作示例 · ':''}${row.chapters.length} 章</span><h2>${esc(row.title)}</h2><p>${esc(row.description??'')}</p><button class="primary-action" type="button" data-catalog-id="${esc(row.id)}" aria-label="开始阅读：${esc(row.title)}">开始阅读 <span aria-hidden="true">↗</span></button></div></article>`).join('')}</div><footer class="catalogue-footer"><p>笔记、批注与实验结果保存在当前设备。换设备时，可主动导出、导入自己的记录。</p>${isDesktop?button('更换书目地址','address'):''}</footer>`;
    content.querySelectorAll<HTMLButtonElement>('[data-catalog-id]').forEach(el=>el.addEventListener('click',async()=>{
      const row=catalogRows.find(r=>r.bookUuid===el.dataset.catalogId)!;el.disabled=true;
      try{if(await openCatalogRecord(row))catalogPanel.close();}catch(error){host.toast(String(error));}finally{el.disabled=false;}
    }));
    content.querySelector('[data-collection="address"]')?.addEventListener('click',()=>renderCatalogAddress(content));
    const guide=document.createElement('button');guide.type='button';guide.className='secondary-action';guide.textContent='如何使用、安装与贡献';content.querySelector('.catalogue-footer')!.append(guide);guide.addEventListener('click',showReaderGuide);
  }catch(error){if(controller.signal.aborted||generation!==catalogGeneration||!catalogPanel.open)return;content.innerHTML=`<p class="collection-error" role="alert">${esc(readableError(error))}</p>${button('重试','retry')}`;content.querySelector('button')!.addEventListener('click',()=>void showCatalogue());if(isDesktop)renderCatalogAddress(content,true);}
}
function showReaderGuide():void{
  const trustedUrl=(value:unknown)=>{try{const url=new URL(String(value));return url.protocol==='https:'&&!url.username&&!url.password?url.href:null;}catch{return null;}};
  const repository=trustedUrl(projectLinks.repository),downloadUrl=trustedUrl(projectLinks.download);
  renderPanel(filesPanel,`<header><div><span class="collection-eyebrow">舒适阅读书库</span><h2>选一章，顺着问题读下去</h2></div>${button('关闭','close')}</header><p>直接选择书籍即可阅读。右上角切换翻页或连续滚动；Aa 调整主题、字号与行距，两者互不改变。</p><p>读到活动或来源时，就地打开“随书学习”。返回书页保留位置和草稿。只有你点击运行，才会开始计算。</p><p>⋯ 中的离线保存会先列出所选章节或活动需要的文件；它和自动保存位置、导出自己的记录是不同动作。笔记保存在当前设备，目前不自动跨端同步。</p><h3>在桌面实践</h3><p>浏览器可使用内置计算、阅读源码和媒体；Python 与本机任务需要 Windows 桌面应用及已核对的环境。当前不宣称其他系统的原生运行已经通过验收。</p>${downloadUrl?button('获取 Windows 桌面应用','project-download',true):'<p class="collection-help">这个书目尚未登记公开安装包地址。当前预览可以继续阅读；已有桌面应用可以连接同一书目。</p>'}<h3>写自己的书</h3><p>从独立书源开始，复用现有阅读器和活动。你可以长期保持私有；提交公开目录是单独决定。</p>${repository?button('源码与贡献说明','project-source'):'<p class="collection-help">当前为本机发行候选，公共维护地址尚未发布。源项目中的贡献指南提供手工和工具辅助两条路线。</p>'}${!isDesktop?'<p><a href="./third-party.html" target="_blank" rel="noopener noreferrer">查看组件来源与许可</a></p>':'<p class="collection-help">第三方许可随安装包提供，个人书籍另按各自来源与许可使用。</p>'}`);
  if(!filesPanel.open)filesPanel.showModal();filesPanel.querySelector('[data-collection="close"]')!.addEventListener('click',()=>filesPanel.close());
  for(const [action,url] of [['project-download',downloadUrl],['project-source',repository]])if(url)filesPanel.querySelector(`[data-collection="${action}"]`)?.addEventListener('click',()=>{if(isDesktop)void invoke('learning_open_external',{url}).catch(error=>host.toast(String(error)));else window.open(url,'_blank','noopener,noreferrer');});
}
function renderCatalogAddress(content:HTMLElement,append=false):void{
  const html=`<form class="catalogue-address"><label>书目地址<input type="url" name="url" required placeholder="https://…/catalog.json" value="${esc(localStorage.getItem('comfortable-reader-catalog-url')??'')}"/></label><p>使用项目站点或你选择的镜像。连接只取得公开书目，不会上传私人书库。</p><button class="primary-action" type="submit">连接书目</button></form>`;
  if(append)content.insertAdjacentHTML('beforeend',html);else content.innerHTML=html;
  content.querySelector('form')!.addEventListener('submit',event=>{event.preventDefault();const value=new FormData(event.target as HTMLFormElement).get('url') as string;try{const url=new URL(value);if(url.protocol!=='https:'&&!(url.protocol==='http:'&&['127.0.0.1','localhost'].includes(url.hostname)))throw new Error('请选择 HTTPS 书目，或本机预览地址');if(url.username||url.password)throw new Error('地址不能包含账号或密码');localStorage.setItem('comfortable-reader-catalog-url',url.href);void showCatalogue();}catch(e){host.toast(String(e));}});
}
async function openCatalogRecord(record:CatalogRecord,chapter?:string):Promise<boolean>{
  const existing=host.known(record.bookUuid);
  if(existing){
    const changed=existing.record.catalogSource?existing.record.catalogSource.sha256!==record.catalogSource.sha256:existing.sourceHash!==record.catalogSource.epub?.sha256;
    if(changed){
      const decision=await reviewRevision(existing.record,record);if(decision===null)return false;
      if(decision==='old'){await host.openExisting(existing.record.id,chapter);return true;}
      await host.open(record,chapter,true);return true;
    }
    await host.openExisting(existing.record.id,chapter);return true;
  }
  await host.open(record,chapter);return true;
}
export async function openWebLink():Promise<boolean>{
  if(isDesktop)return false;const params=new URLSearchParams(location.search),slug=params.get('book');if(!slug)return false;
  const fetched=await fetchCatalog();projectLinks=fetched.catalog.project??{};
  const record=fetched.records.find(r=>r.catalogSource.slug===slug);if(!record)throw new Error('这个链接中的书已不在当前书目，请从“发现书籍”选择现有内容。');
  return await openCatalogRecord(record,params.get('chapter')??undefined);
}
async function reviewRevision(previous:any,next:CatalogRecord):Promise<'new'|'old'|null>{
  let changes='本机已有这本书的记录。新内容会使用自己的位置与书页批注；原文件和旧记录继续保留。';
  if(previous.catalogSource){
    const oldBytes=await publicStore.getChunk(previous.catalogSource.sha256);
    const newBytes=await fetchVerified(next.catalogSource.url,next.catalogSource,undefined,2*1024*1024);
    await publicStore.putChunk(next.catalogSource.sha256,newBytes);
    const current=JSON.parse(new TextDecoder().decode(newBytes));
    if(oldBytes){const old=JSON.parse(new TextDecoder().decode(oldBytes));const chapterChanges=current.chapters.filter((c:any)=>old.chapters.find((p:any)=>p.id===c.id)?.readingDocument?.sha256!==c.readingDocument?.sha256).length;const resourceChanges=current.resources.filter((c:any)=>old.resources.find((p:any)=>p.id===c.id)?.sha256!==c.sha256).length;changes=`${chapterChanges} 章阅读文件、${resourceChanges} 项材料有变化。未选择的材料不会取得；原位置、批注和运行记录保留在各自的内容版本中。`;}
  }
  renderPanel(filesPanel,`<header><div><span class="collection-eyebrow">书籍内容更新</span><h2>${esc(previous.catalogSource?.revision??'本机内容')} → ${esc(next.catalogSource.revision)}</h2></div>${button('暂不选择','cancel')}</header><p>${esc(changes)}</p><footer>${button('继续原有内容','old')}${button('使用这份新内容','new',true)}</footer>`);
  if(!filesPanel.open)filesPanel.showModal();
  return await new Promise(resolve=>{let done=false;const close=()=>{if(!done)resolve(null);};filesPanel.addEventListener('close',close,{once:true});filesPanel.querySelectorAll<HTMLButtonElement>('[data-collection]').forEach(button=>button.addEventListener('click',()=>{done=true;resolve(button.dataset.collection==='new'?'new':button.dataset.collection==='old'?'old':null);filesPanel.close();}));});
}
export async function showBookFiles():Promise<void>{
  const current=host.current();if(!current){await showCatalogue();return;}
  const portable=portableBooks.get(current.id);
  renderPanel(filesPanel,`<header><div><span class="collection-eyebrow">书籍与保存</span><h2>${esc(portable?.book.title??current.title??'当前书籍')}</h2></div>${button('关闭','close')}</header><div class="book-file-groups"><section class="book-file-group"><h3>阅读材料</h3><div class="book-file-actions">${portable?button('本章离线保存…','chapter',true)+button('全书正文离线保存…','book')+button('下载电子书 · EPUB','epub'):''}${button('管理已保存资源','storage')}</div></section><section class="book-file-group"><h3>这本书的学习记录</h3><div class="book-file-actions" data-learning-record-actions>${button('导出学习记录','export-personal')}${button('导入同版学习记录','import-personal')}</div></section></div><p class="collection-help">阅读位置会自动保存。这里的离线保存只取得你选择的正文与必要材料；电子书文件不包含你的笔记或运行环境。</p><p class="collection-help">${isDesktop?'此桌面应用':'此浏览器'}单独保存个人记录，目前没有跨端自动同步。导入会合并批注、笔记和代码历史，保留此处的阅读位置；运行收据可回看，大型产物留在原设备。</p><div class="collection-status" role="status"></div>`);
  if(!filesPanel.open)filesPanel.showModal();
  const editions=document.createElement('button');editions.type='button';editions.className='secondary-action';editions.textContent='版本记录';filesPanel.querySelector('[data-learning-record-actions]')!.append(editions);editions.addEventListener('click',()=>void showEditionRecords(current.id));
  filesPanel.querySelectorAll<HTMLButtonElement>('[data-collection]').forEach(el=>el.addEventListener('click',async()=>{
    try{
      const action=el.dataset.collection;
      if(action==='close')filesPanel.close();
      if(action==='chapter'){const chapter=portable!.book.chapters.find((c:any)=>current.href.endsWith(c.readingDocument.path.split('/').pop()));if(!chapter)throw new Error('当前位置属于配套资料，请先回到正文，再选择保存章节');await showAcquisition(current.id,'chapter',chapter.id);}
      if(action==='book')await showAcquisition(current.id,'book','all');
      if(action==='storage')await showStorage();
      if(action==='export-personal')await exportPersonal(current.id);
      if(action==='import-personal'){
        const input=document.createElement('input');input.type='file';input.accept='application/json,.json';input.hidden=true;filesPanel.append(input);
        input.addEventListener('change',async()=>{try{const file=input.files?.[0];if(!file)return;if(file.size>10*1024*1024)throw new Error('这份记录超过 10 MB 导入范围');reviewPersonalImport(JSON.parse(await file.text()));}catch(error){reportCollectionError(error);}finally{input.remove();}});input.click();
      }
      if(action==='epub'){
        const item=portable!.record.catalogSource.epub;if(!item)throw new Error('此书尚未提供完整 EPUB 文件');
        const data=await fetchVerified(item.url,item,undefined,128*1024*1024);download(`${portable!.book.slug}-${portable!.book.revision}.epub`,new Blob([data],{type:'application/epub+zip'}));host.toast('电子书已交给系统下载；其中保留正文、来源和活动说明。');
      }
    }catch(error){reportCollectionError(error);}
  }));
  const generation=filesGeneration;
  const snapshot=await invoke<any>('bootstrap'),record=snapshot.books.find((row:any)=>row.id===current.id);
  if(generation!==filesGeneration||!filesPanel.open)return;
  if(record?.bookUuid){const old=await legacyRecords(record.bookUuid);if(old&&generation===filesGeneration&&filesPanel.open){const migrate=document.createElement('button');migrate.type='button';migrate.className='secondary-action';migrate.textContent='接续旧网页记录';filesPanel.querySelector('[data-learning-record-actions]')!.append(migrate);migrate.addEventListener('click',()=>void reviewLegacy(current.id,old));}}
}
async function reviewLegacy(bookId:string,old:LegacyRecords):Promise<void>{
  renderPanel(filesPanel,`<header><div><span class="collection-eyebrow">旧网页记录</span><h2>把已有思考带回这本书</h2></div>${button('关闭','close')}</header><p>${old.notes.length} 份笔记或未提交草稿 · ${old.runs.length} 份运行记录</p><p class="collection-help">这些记录来自当前网站地址下的旧阅读页面。原数据库保持完整；合并后在“本书笔记”和“运行记录”回看。旧网页只保存章节或标题位置，无法据此恢复精确的书页批注，旧锚点会原样保留待核对。</p><footer>${button('合并旧记录','merge-legacy',true)}${button('导出旧记录备份','export-legacy')}${old.reading?.chapter?button('回到记录中的章节','legacy-chapter'):''}</footer><p class="collection-status" role="status"></p>`);
  filesPanel.querySelector('[data-collection="close"]')!.addEventListener('click',()=>filesPanel.close());
  filesPanel.querySelector('[data-collection="export-legacy"]')!.addEventListener('click',()=>download('旧网页记录.json',new Blob([JSON.stringify(old,null,2)],{type:'application/json'})));
  filesPanel.querySelector('[data-collection="legacy-chapter"]')?.addEventListener('click',async()=>{try{await host.openExisting(bookId,old.reading.chapter);filesPanel.close();host.toast('已定位到旧记录对应的章节；旧标题位置仍在导出记录中。');}catch(error){reportCollectionError(error);}});
  filesPanel.querySelector('[data-collection="merge-legacy"]')!.addEventListener('click',async()=>{try{const state=await invoke('learning_load_state',{bookId});await invoke('learning_save_state',{bookId,value:await mergeLegacyRecords(state,old)});filesPanel.close();host.toast('旧记录已接续，原数据库保留；没有重新运行实验或套用旧书页坐标。');}catch(error){reportCollectionError(error);}});
}
function download(name:string,blob:Blob):void{const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function reviewPersonalImport(value:any):void{
  if(!value||value.format!=='comfortable-reader-personal@1')throw new Error('这不是受支持的学习记录文件');
  const count=(list:any)=>Array.isArray(list)?list.length:0;
  renderPanel(filesPanel,`<header><div><span class="collection-eyebrow">导入学习记录</span><h2>${esc(value.book?.title??'待核对的书籍')}</h2></div>${button('取消','cancel-import')}</header><p>${count(value.progress?.annotations)} 条批注 · ${count(value.study?.notes)} 条笔记 · ${count(value.drafts)} 份代码（含历史） · ${count(value.runs)} 份运行收据</p><p class="collection-help">只有书籍与内容身份匹配才会合并。同名冲突保留双方，已有草稿继续保留；导入代码不会获得执行权限。</p><label><input class="import-reading-position" type="checkbox"/> 同时恢复导出时的阅读位置和阅读方式</label><footer>${button('合并到当前书籍','merge-import',true)}</footer><p class="collection-status" role="status"></p>`);
  filesPanel.querySelector('[data-collection="cancel-import"]')!.addEventListener('click',()=>filesPanel.close());
  filesPanel.querySelector<HTMLButtonElement>('[data-collection="merge-import"]')!.addEventListener('click',async event=>{const button=event.currentTarget as HTMLButtonElement;button.disabled=true;try{await host.importPersonal(value,filesPanel.querySelector<HTMLInputElement>('.import-reading-position')!.checked);filesPanel.close();}catch(error){reportCollectionError(error);button.disabled=false;}});
}
async function exportPersonal(bookId:string):Promise<void>{
  let pending=false;try{await host.flushPersonal();}catch{pending=true;}
  const live=host.livePersonal(bookId);
  const snapshot=await invoke<any>('bootstrap');let study=live?.study;
  if(!study){try{study=await invoke<any>('learning_load_state',{bookId});}catch{const raw=await invoke<any>('learning_raw_state',{bookId});download(raw.name,new Blob([new Uint8Array(raw.bytes)],{type:'application/octet-stream'}));host.toast('已另存原始学习记录。它尚未修复，不会覆盖原文件；核对后再恢复。');return;}}
  const runs=await invoke<any[]>('learning_run_history',{bookId});
  const record=snapshot.books.find((b:any)=>b.id===bookId);
  const value={format:'comfortable-reader-personal@1',exportedAt:new Date().toISOString(),containsUnwrittenEdits:pending,book:{id:record.id,bookUuid:record.bookUuid,title:record.title},progress:live?.progress??snapshot.progress[bookId]??null,study,runs:[],drafts:[]};
  for(const item of runs){const run=await invoke<any>('learning_run_status',{bookId,runId:item.run_id});(value.runs as any[]).push({record:run,snapshot:await invoke('learning_run_snapshot',{bookId,runId:item.run_id}).catch(()=>null)});}
  for(const receipt of study.importedRuns??[])if(receipt?.record?.run_id&&!(value.runs as any[]).some(row=>row.record.run_id===receipt.record.run_id))(value.runs as any[]).push(receipt);
  const activityIds=await invoke<string[]>('learning_draft_activities',{bookId});
  for(const activityId of activityIds){
    const draft=await invoke<any>('learning_draft',{bookId,activityId});
    const versions=await invoke<any[]>('learning_draft_versions',{bookId,activityId});
    const history=[];for(const version of versions)history.push(await invoke('learning_draft_revision',{bookId,activityId,revision:version.sha256}));
    if(draft||(history.length>0))(value.drafts as any[]).push({activityId,...(draft??history[0]),history});
  }
  if(live?.draft){const draft=live.draft;const hash=await sha256(new TextEncoder().encode(draft.code).buffer);const row=(value.drafts as any[]).find(r=>r.activityId===draft.activityId);const entry={code:draft.code,sha256:hash,updated_at:Math.floor(Date.now()/1000)};if(row){row.code=entry.code;row.sha256=hash;if(!row.history.some((v:any)=>v.sha256===hash))row.history.push(entry);}else(value.drafts as any[]).push({activityId:draft.activityId,...entry,history:[entry]});}
  download('学习记录-'+bookId+'.json',new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));host.toast('已导出位置、批注、笔记、代码草稿及历史和运行收据；运行大文件留在原设备。');
}
async function showEditionRecords(bookId:string):Promise<void>{
  try{
    const snapshot=await invoke<any>('bootstrap'),book=snapshot.books.find((row:any)=>row.id===bookId);
    const all=await invoke<any[]>('progress_editions',{bookId});
    const rows=all.filter((row,i)=>!all.some((other,j)=>j!==i&&other.progress.sourceSha256===row.progress.sourceSha256&&other.progress.contentDigest&&!row.progress.contentDigest));
    renderPanel(filesPanel,`<header><div><span class="collection-eyebrow">内容版本记录</span><h2>${esc(book.title)}</h2></div>${button('关闭','close')}</header><p class="collection-help">每份内容版本保留自己的阅读位置和书页批注。打开新正文会建立独立记录；再次打开原版文件时，可恢复原版记录。学习笔记与运行历史另行保留。</p><div class="edition-records">${rows.map((row,i)=>`<details><summary>${new Date((row.progress.updatedAt??0)*1000).toLocaleString()} · ${row.progress.annotations?.length??0} 条书页批注${row.key===snapshot.progress[bookId]?.contentDigest?' · 当前内容':''}</summary><p class="collection-help">内容身份 ${esc(row.key.slice(0,12))}</p>${(row.progress.annotations??[]).map((n:any)=>`<p>${esc(n.kind==='text-mark'?n.selectedText+(n.comment?' — '+n.comment:''):n.kind==='free-text'?n.text:'手写笔迹 · '+n.points.length+' 个轨迹点')}</p>`).join('')}<button type="button" data-export-edition="${i}">导出这一版的位置与批注</button></details>`).join('')||'<p>还没有保存过内容版本。</p>'}</div>`);
    filesPanel.querySelector('[data-collection="close"]')!.addEventListener('click',()=>filesPanel.close());
    filesPanel.querySelectorAll<HTMLButtonElement>('[data-export-edition]').forEach(element=>element.addEventListener('click',()=>{const row=rows[Number(element.dataset.exportEdition)];const value={format:'comfortable-reader-personal@1',exportedAt:new Date().toISOString(),book:{id:bookId,bookUuid:row.progress.bookUuid??book.bookUuid,title:book.title},progress:row.progress,study:{notes:[]},runs:[],drafts:[]};download('书页记录-'+row.key.slice(0,12)+'.json',new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));}));
  }catch(error){host.toast(String(error));}
}
export async function showAcquisition(bookId:string,kind:string,id:string,optionals:string[]=[]):Promise<boolean>{
  const session=portableBooks.get(bookId);if(!session)throw new Error('书籍尚未打开');
  if(downloadAbort)throw new Error('已有下载正在进行，请先完成或暂停');
  const plan=await planSelection(publicStore,session.book,kind,id,optionals);
  const activity=session.book.activities.find((a:any)=>a.id===id);
  renderPanel(filesPanel,`<header><div><span class="collection-eyebrow">${kind==='activity'?'准备这个活动':'离线保存'}</span><h2>${esc(activity?.title??(kind==='chapter'?session.book.chapters.find((c:any)=>c.id===id)?.title:kind==='resource'?session.book.resources.find((r:any)=>r.id===id)?.title:'全书正文'))}</h2></div>${button('关闭','close')}</header><p class="download-summary">本次新增 <strong>${bytes(plan.addedBytes)}</strong><span>已有 ${bytes(plan.alreadyPresentBytes)} · 内容按哈希复用</span></p><ul class="acquisition-list">${plan.items.map((item:any)=>`<li><div><strong>${esc(item.title)}</strong><span>${esc(item.reason)}</span></div><small>${item.cached?'已保存':bytes(item.bytes)}</small></li>`).join('')}</ul>${activity?.optional.length?`<details class="acquisition-options"><summary>额外材料 · 默认不取得</summary>${activity.optional.map((key:string)=>{const resource=session.book.resources.find((r:any)=>r.id===key);return `<label><input type="checkbox" value="${esc(key)}" ${optionals.includes(key)?'checked':''}/> ${esc(resource.title)} · ${bytes(resource.bytes)}</label>`;}).join('')}</details>`:''}<p class="collection-help">只取得清单中的文件，不安装环境，也不开始运行。浏览器可能回收临时存储；需要长期保管时，请另行导出。</p><footer>${button(plan.addedBytes?'取得并离线保存':'固定离线保存','acquire',true)}${button('暂停','pause')}</footer><p class="collection-status" role="status"></p>`);
  if(!filesPanel.open)filesPanel.showModal();
  const status=filesPanel.querySelector<HTMLElement>('.collection-status')!;
  const run=filesPanel.querySelector<HTMLButtonElement>('[data-collection="acquire"]')!;
  const pause=filesPanel.querySelector<HTMLButtonElement>('[data-collection="pause"]')!;pause.hidden=true;
  return await new Promise(resolve=>{
    let finished=false;
    const close=()=>{downloadAbort?.abort();downloadAbort=null;if(!finished)resolve(false);};filesPanel.addEventListener('close',close,{once:true});
    filesPanel.querySelector('[data-collection="close"]')!.addEventListener('click',()=>filesPanel.close());
    filesPanel.querySelectorAll<HTMLInputElement>('.acquisition-options input').forEach(input=>input.addEventListener('change',()=>{finished=true;resolve(false);filesPanel.removeEventListener('close',close);void showAcquisition(bookId,kind,id,Array.from(filesPanel.querySelectorAll<HTMLInputElement>('.acquisition-options input:checked')).map(el=>el.value));}));
    pause.addEventListener('click',()=>downloadAbort?.abort());
    run.addEventListener('click',async()=>{
      const controller=new AbortController();downloadAbort=controller;run.disabled=true;pause.hidden=false;
      filesPanel.querySelectorAll<HTMLInputElement>('input').forEach(el=>el.disabled=true);
      try{
        for(let i=0;i<plan.items.length;i++){const item=plan.items[i];await acquire(publicStore,session.url,item,{signal:controller.signal,onProgress:()=>{status.textContent=`正在取得 ${i+1} / ${plan.items.length} · ${item.title}`;}});await materialize(publicStore,item);}
        const hashes=plan.items.flatMap((item:any)=>resourceUnits(item).map((u:any)=>u.sha256));hashes.push(session.record.catalogSource.sha256);
        await publicStore.pin(`${session.book.id}:${kind}:${id}`,hashes);
        status.textContent='所选内容已校验并离线保存。';finished=true;resolve(true);run.textContent='已保存';
      }catch(error){status.textContent=controller.signal.aborted?'已暂停。已核对的分块保留，再次点击即可继续。':String(error);run.disabled=false;run.textContent='继续取得';}
      finally{downloadAbort=null;pause.hidden=true;filesPanel.querySelectorAll<HTMLInputElement>('input').forEach(el=>el.disabled=false);}
    });
  });
}
async function showStorage():Promise<void>{
  const pins:Map<string,string[]>=await publicStore.getPins(),unused=await publicStore.unusedChunks();
  renderPanel(filesPanel,`<header><div><span class="collection-eyebrow">本机内容</span><h2>已保存的阅读材料</h2></div>${button('关闭','close')}</header><p class="collection-help">取消固定后，可清理没有其他用途的公共资源。个人批注、草稿和结果始终单独保留。</p><div class="saved-collections">${[...pins.keys()].map(key=>{const [bookUuid,kind,id]=key.replace('urn:uuid:','').split(':');const session=[...portableBooks.values()].find(b=>b.book.id==='urn:uuid:'+bookUuid);const title=kind==='chapter'?session?.book.chapters.find((c:any)=>c.id===id)?.title:kind==='activity'?session?.book.activities.find((a:any)=>a.id===id)?.title:kind==='book'?'全书正文':session?.book.resources.find((r:any)=>r.id===id)?.title;return `<div><span>${esc(session?.book.title??'已保存书籍')} · ${esc(title??id)}</span><button type="button" data-unpin="${esc(key)}">取消固定</button></div>`;}).join('')||'<p>还没有固定离线内容。</p>'}</div><footer>${button(`清理未使用缓存 · ${bytes(unused.reduce((n:number,r:any)=>n+r.bytes,0))}`,'clear')}</footer>`);
  filesPanel.querySelector('[data-collection="close"]')!.addEventListener('click',()=>filesPanel.close());
  filesPanel.querySelector('[data-collection="clear"]')!.addEventListener('click',async()=>{await publicStore.clearUnused();await showStorage();});
  filesPanel.querySelectorAll<HTMLButtonElement>('[data-unpin]').forEach(el=>el.addEventListener('click',async()=>{await publicStore.unpin(el.dataset.unpin);await showStorage();}));
}
