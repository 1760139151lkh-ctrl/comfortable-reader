import { invoke, isDesktop } from './platform';
import {showAcquisition} from './catalogue';
import {portableBooks,publicStore} from './portable-books';
import {materialize} from '../../site/core.mjs';
import {builtinSource} from './web-backend';
import './learning.css';
import { mountPdf } from './pdf-viewer';
import { mountGeometry } from './geometry-viewer';
import { mountImage } from './image-viewer';
import { connectStudyWorkspace } from '../../site/shared/workspace.mjs';
import { renderReadableText, renderDelimited } from './study-text';
import { readableError } from './reader-errors';
import { mountAudioPlots } from './audio-plots';
import { unfinishedNoteIdentity } from './personal-records';

type Asset = {comparison_reference?:string; root?:string; id:string; relative_path:string; sha256:string; bytes:number; kind:string; title:string; filename:string; role:string; chapters:string[]; media?:any };
type Chapter = {source_sha256?:string; id:string; number:number; title:string; href:string; question:string; assets:string[]};
type Activity = {execution_revision?:string;presentation?:string;result_suffix?:string;reading_guide?:string;practice?:{task:string;starter:string;hints:string[];symbols:Array<[string,string]>};reference_asset?:string;key_symbol?:string;capability?:string; description?:string; editable?:boolean; optional_training?:boolean; parameters?:any[]; registered?:boolean; id:string; chapter:string; entry_asset:string; title:string; runtime:string; timeout_seconds:number; identity:string};
type Pack = {portable?:boolean;href_activities?:Record<string,string>;schema_version:number; book_uuid:string; book_revision_sha256:string; chapters:Chapter[]; assets:Asset[]; href_assets:Record<string,string>; activities:Activity[]; audio_groups:any[]; source_claims:any[]; dependencies:any[]; history:any[]};
type Origin = {contentDigest?:string;sourceSha256?:string;localSource?:boolean;bookId:string; pane:number; cfi:string|null; href:string; quote:string; focus:HTMLElement|null; selection?:Range|null};
type Host = {current:()=>Origin|null; settle:()=>Promise<void>; jump:(target:string,pane:number)=>Promise<void>; toast:(text:string)=>void};
const packs=new Map<string,Pack>();
const failures=new Map<string,string>();
const lastBodyChapter=new Map<string,{revision:string;chapterId:string}>();
const bodyLocations=new WeakMap<object,{size:number;positions:number[]}>();
let host:Host;
let dialog:HTMLElement;
let workspace:{readonly open:boolean;show:()=>void;close:()=>void};
let origin:Origin|null=null;
let pack:Pack|null=null;
let chapter:Chapter|null=null;
let currentAsset:Asset|null=null;
let activity:Activity|null=null;
let study:any={notes:[],media:{}};
let studyLoaded=false;
let studyOpenGeneration=0;
let noteTransition=false;
let referenceCode='';
let latestRun:any=null;
let activeRun:any=null;
let draftMode=false;
let poll:number|null=null;
let saveTimer:number|null=null;
let activeMedia:HTMLMediaElement|null=null;
let audioGraph:AudioContext|null=null;
let audioGain:GainNode|null=null;
let audioAnalysisWorker:Worker|null=null;
const audioAnalyses=new Map<string,any>();
let pageGeneration=0;
let cancelConsent:(()=>void)|null=null;
let pendingMediaAnchor:any=null;
let objectUrls:string[]=[];
let disposeObject:(()=>void)|null=null;
let disposeResponsive:(()=>void)|null=null;
let latestReport:any=null;
let showingAuthorReference=false;
let resultViewRevision=0;
let displayedResultIdentity='';
let trace:any[]=[];
let traceIndex=-1;
let showingTrace=false;
let linearVisualization=true;
let stateWriteQueue:Promise<void>=Promise.resolve();
const enc=(v:unknown)=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
const one=<T extends HTMLElement=HTMLElement>(q:string)=>dialog.querySelector<T>(q)!;
const invokeBook=<T>(command:string,args:Record<string,unknown>={})=>invoke<T>(command,{bookId:origin!.bookId,...args});
const assetById=(id:string)=>pack?.assets.find(a=>a.id===id);
const chapterOf=(href:string):Chapter|null=>{
  if(!pack)return null;const clean=href.split('#')[0];
  const direct=pack.chapters.find(c=>clean.endsWith(c.href));if(direct)return direct;
  const mapped=Object.entries(pack.href_assets).find(([key])=>clean.endsWith(key.split('#')[0]));
  const linked=mapped?pack.assets.find(a=>a.id===mapped[1]):null;
  const owners=pack.chapters.filter(c=>linked?.chapters.includes(c.id));if(owners.length===1)return owners[0];
  const previous=origin?lastBodyChapter.get(origin.bookId+':'+origin.pane):null;
  if(previous?.revision===pack.book_revision_sha256){const known=pack.chapters.find(c=>c.id===previous.chapterId);if(known&&(!owners.length||owners.includes(known)))return known;}
  return {id:'reader-resources',number:0,title:'配套资料',href:clean,question:'从当前资料核对来源与内容；全书目录保留各章的学习入口。',assets:linked?[linked.id]:[]};
};
const label=(kind:string)=>({code:'程序',audio:'声音',video:'视频',image:'图像',pdf:'原典',model:'模型',geometry:'三维',data:'数据',text:'资料'}[kind]??'资料');
const buffer=(value:ArrayBuffer|number[])=>value instanceof ArrayBuffer?value:new Uint8Array(value).buffer;
const formatBytes=(n:number)=>n>1024*1024?`${(n/1024/1024).toFixed(1)} MB`:`${Math.ceil(n/1024)} KB`;
const digest=async(code:string)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(code)))).map(x=>x.toString(16).padStart(2,'0')).join('');

const preparationGeneration=new Map<string,number>();
export async function prepareLearning(bookId:string):Promise<void>{
  const generation=(preparationGeneration.get(bookId)??0)+1;preparationGeneration.set(bookId,generation);
  packs.delete(bookId);failures.delete(bookId);
  try {const p=await invoke<Pack>('learning_pack',{bookId});if(preparationGeneration.get(bookId)!==generation)return;p.audio_groups??=[];p.activities??=[];p.source_claims??=[];p.dependencies??=[];p.history??=[];p.href_assets??={};p.chapters.forEach((c,i)=>{c.number??=i+1;c.title||=(i+1)+' · 本文';c.question??='';c.assets??=[];});packs.set(bookId,p);window.dispatchEvent(new CustomEvent('reader-learning-ready',{detail:{bookId}}));}
  catch(e){if(preparationGeneration.get(bookId)===generation)failures.set(bookId,String(e));}
  refreshLearningButton();
}
export function holdLearning(bookId:string,reason:string):void{preparationGeneration.set(bookId,(preparationGeneration.get(bookId)??0)+1);packs.delete(bookId);failures.set(bookId,reason);refreshLearningButton();}
export function refreshLearningButton():void{
  const button=document.querySelector<HTMLButtonElement>('.study-toggle');if(!button)return;
  const current=host?.current();const reason=current?failures.get(current.bookId):undefined;const actionable=Boolean(reason&&!reason.includes('没有登记学习增强包')&&!reason.includes('资料尚未就绪'));button.hidden=!current||(!packs.has(current.bookId)&&!actionable);if(current&&!packs.has(current.bookId)&&actionable){button.innerHTML='<span aria-hidden="true">!</span><span class="study-toggle-label">学习资料待匹配</span>';button.setAttribute('aria-label','学习资料待匹配');button.title=reason!;}else{button.innerHTML=activeRun?.status==='running'?'<span aria-hidden="true">◌</span><span class="study-toggle-label">实验计算中</span>':'<span aria-hidden="true">✧</span><span class="study-toggle-label">随书学习</span>';button.title='打开当前段落的学习材料';button.setAttribute('aria-label','打开随书学习');}
}
export function learningIsOpen():boolean{return Boolean(workspace?.open);}
export async function flushLearningBeforeClose():Promise<void>{
  if(saveTimer!==null){clearTimeout(saveTimer);saveTimer=null;}
  if(!origin||!studyLoaded)return;
  const editor=dialog?.querySelector<HTMLTextAreaElement>('.study-code');
  if(draftMode&&activity&&editor)await invokeBook('learning_save_draft',{activityId:activity.id,code:editor.value});
  if(activeMedia&&currentAsset){study.media[currentAsset.id]={time:activeMedia.currentTime};activeMedia.pause();}
  captureUnfinishedNote();
  await persistStudy();
}
export function learningMemoryBackup(bookId:string):any {
  if(!origin||origin.bookId!==bookId||!studyLoaded)return null;
  captureUnfinishedNote();
  const editor=dialog?.querySelector<HTMLTextAreaElement>('.study-code');
  return {study:structuredClone(study),draft:draftMode&&activity&&editor?{activityId:activity.id,code:editor.value}:null};
}
function captureUnfinishedNote():boolean{
  const note=dialog?.querySelector<HTMLTextAreaElement>('.study-note-editor textarea');
  if(!studyLoaded||!origin||!note||note.dataset.committed==='true')return false;
  const editor=note.closest<HTMLElement>('.study-note-editor');
  let restored:Record<string,any>={};
  if(editor?.dataset.restoredAnchor){try{const value=JSON.parse(editor.dataset.restoredAnchor);if(value&&typeof value==='object'&&!Array.isArray(value))restored=value;}catch{}}
  const original=(key:string,fallback:unknown)=>Object.prototype.hasOwnProperty.call(restored,key)?restored[key]:fallback;
  const draft:Record<string,any>={...restored,text:note.value,chapter:original('chapter',chapter?.id),quote:original('quote',origin.quote),href:original('href',origin.href),cfi:original('cfi',origin.cfi),content_digest:original('content_digest',origin.contentDigest),editId:editor?.dataset.editId??original('editId',null)};
  if(!Object.prototype.hasOwnProperty.call(draft,'media_anchor')&&pendingMediaAnchor)draft.media_anchor=structuredClone(pendingMediaAnchor);
  delete draft.import_id;
  study.unfinished_note=draft;return true;
}
function persistStudy():Promise<void>{
  if(!origin||!studyLoaded)return Promise.reject(new Error('学习记录未成功读取；未写入空白记录'));
  const bookId=origin.bookId,value=structuredClone(study);
  stateWriteQueue=stateWriteQueue.catch(()=>{}).then(()=>invoke<void>('learning_save_state',{bookId,value}));return stateWriteQueue;
}
export function learningReadingProgress(bookId:string,book:any,cfi:string|null,pane=0):{kind:'body'|'reference';text:string;percent:number|null;position:number;total:number}|null{
  const p=packs.get(bookId);if(!p||!cfi||!book?.locations.length())return null;
  const section=book.spine.get(cfi);const href=section?.href??'';
  const currentChapter=p.chapters.find(c=>href.endsWith(c.href));
  const main=Boolean(currentChapter);if(currentChapter)lastBodyChapter.set(bookId+':'+pane,{revision:p.book_revision_sha256,chapterId:currentChapter.id});
  const size=book.locations.length();let cache=bodyLocations.get(book);
  if(!cache||cache.size!==size){
    const locations:string[]=JSON.parse(book.locations.save());const positions:number[]=[];
    for(let index=0;index<locations.length;index++){const item=book.spine.get(locations[index]);if(item&&p.chapters.some(c=>item.href.endsWith(c.href)))positions.push(index);}
    cache={size,positions};bodyLocations.set(book,cache);
  }
  if(!cache.positions.length)return null;
  if(!main)return{kind:'reference',text:'配套资料 · 不计入正文进度',percent:null,position:1,total:cache.positions.length};
  const current=book.locations.locationFromCfi(cfi);let position=0;
  while(position<cache.positions.length&&cache.positions[position]<current)position++;
  const total=cache.positions.length;return{kind:'body',text:`正文位置 ${Math.min(total,position+1)} / ${total}`,percent:total>1?Math.min(1,position/(total-1)):null,position:Math.min(total,position+1),total};
}

export function learningCfiFromBodyPosition(bookId:string,book:any,cfi:string|null,position:number):string|null{
  const progress=learningReadingProgress(bookId,book,cfi);if(!progress)return null;
  const positions=bodyLocations.get(book)!.positions;
  return book.locations.cfiFromLocation(positions[Math.min(positions.length-1,Math.max(0,Math.round(position)-1))]);
}

function handleStudyEscape():void{
  if(cancelConsent){cancelConsent();return;}
  const menu=dialog.querySelector<HTMLDetailsElement>('.study-activity-more[open]');
  if(menu){menu.open=false;menu.querySelector<HTMLElement>('summary')?.focus();return;}
  void closeStudy();
}
function trapStudyTab(event:KeyboardEvent):void{
  const scope=cancelConsent?dialog.querySelector<HTMLElement>('.study-consent'):dialog;
  if(!scope)return;
  const controls=Array.from(scope.querySelectorAll<HTMLElement>('a[href],button,input,select,textarea,summary,[tabindex]:not([tabindex="-1"])'))
    .filter(el=>!el.closest('[inert]')&&!('disabled' in el&&(el as HTMLButtonElement).disabled)&&el.getClientRects().length>0);
  if(!controls.length)return;
  const index=controls.indexOf(document.activeElement as HTMLElement);
  if(index<0||event.shiftKey&&index===0||!event.shiftKey&&index===controls.length-1){
    event.preventDefault();controls[event.shiftKey?controls.length-1:0].focus();
  }
}

export function initLearning(adapter:Host):void{
  host=adapter;
  window.addEventListener('reader-theme-change',()=>{if(!workspace?.open||!activity||activity.presentation!=='linear-classifier@1'||!latestReport)return;const step=showingTrace?trace[traceIndex]:null;drawCards(latestReport.training_rows??[],step?.weights??latestReport.trained_weights,step?.bias??latestReport.trained_bias);});
  const button=document.createElement('button');button.className='study-toggle';button.type='button';button.hidden=true;
  button.innerHTML='<span aria-hidden="true">✧</span><span class="study-toggle-label">随书学习</span>';button.title='打开当前段落的学习材料';button.setAttribute('aria-label','打开随书学习');
  document.querySelector('.active-book-title')?.after(button);
  button.addEventListener('click',async()=>{await host.settle();const current=host.current();if(current){if(!packs.has(current.bookId))await prepareLearning(current.bookId);if(packs.has(current.bookId))void openStudy(current);else host.toast(failures.get(current.bookId)||'学习资料尚未匹配；正文仍可阅读');}});
  dialog=document.createElement('section');dialog.className='study-stage';dialog.setAttribute('aria-label','随书学习');
  document.querySelector('.app-shell')?.append(dialog);
  workspace=connectStudyWorkspace(dialog,{root:document.querySelector('.app-shell'),readingRoot:document.querySelector('.reader-grid'),toolbar:document.querySelector('.top-toolbar')});
  dialog.addEventListener('cancel',e=>{e.preventDefault();void closeStudy();});
  document.addEventListener('keydown',e=>{
    if(!workspace?.open||dialog.contains(e.target as Node)||document.querySelector('dialog[open]'))return;
    if(e.key==='Escape'){e.preventDefault();e.stopPropagation();handleStudyEscape();}
    else if(e.key==='Tab')trapStudyTab(e);
  },true);
  dialog.addEventListener('keydown',e=>{
    // Reading shortcuts never steal an editor, media or stage keystroke.
    e.stopPropagation();
    if(e.key==='Escape'){e.preventDefault();handleStudyEscape();return;}
    if(e.key==='Tab')trapStudyTab(e);
    if(e.ctrlKey&&e.key==='s'){e.preventDefault();void saveDraft();}
    if(e.ctrlKey&&e.key==='Enter'&&activity){e.preventDefault();void runActivity();}
  });
  dialog.addEventListener('click',async e=>{
    const sourceLink=(e.target as Element).closest<HTMLAnchorElement>('a[href]');
    if(sourceLink&&/^https?:/.test(sourceLink.href)){
      e.preventDefault();const source=pack?.source_claims.find(s=>s.url&&new URL(s.url).href===sourceLink.href);
      if(source)void invokeBook('learning_open_source',{sourceId:source.id}).catch(showError);
      return;
    }
    const target=(e.target as Element).closest<HTMLElement>('[data-study]');if(!target)return;
    const action=target.dataset.study!;target.closest<HTMLDetailsElement>('.study-activity-more')?.removeAttribute('open');
    if(action==='backup-raw'){try{const raw=await invokeBook<any>('learning_raw_state');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([new Uint8Array(raw.bytes)],{type:'application/octet-stream'}));a.download=raw.name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),30000);one('.study-run-status').textContent='原始副本已导出。原记录保持完整，核对后再恢复。';}catch(error){showError(error);}return;}
    if(['overview','asset','activity','sources','notes','map','history','chapter','audio-group','video','run-history','saved-run'].includes(action)) await saveDraft();
    if(action==='close')void closeStudy();
    if(action==='copy-path')void invokeBook<{path:string}>('learning_resource_info',{assetId:target.dataset.id}).then(async v=>{await navigator.clipboard.writeText(v.path);status(isDesktop&&/^(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\[^\\]+\\|\/)/.test(v.path)?'已复制核对过的本机路径':'已复制书籍内资源路径');}).catch(showError);
    if(action==='overview')overview();
    if(action==='asset')void showAsset(target.dataset.id!);
    if(action==='activity')void showActivity(target.dataset.id!);
    if(action==='sources')showSources();
    if(action==='notes'){pendingMediaAnchor=null;showNotes();}
    if(action==='note-media'){const asset=assetById(target.dataset.id||currentAsset?.id||'');if(asset){pendingMediaAnchor={asset_id:asset.id,source_sha256:asset.sha256,...study.media[asset.id],time:activeMedia&&currentAsset?.id===asset.id?activeMedia.currentTime:study.media[asset.id]?.time??0};showNotes();void addNote();}}
    if(action==='return-media'){const note=study.notes.find((n:any)=>n.id===target.dataset.id);if(note?.media_anchor){study.media[note.media_anchor.asset_id]={time:note.media_anchor.time,frame:note.media_anchor.frame};void showAsset(note.media_anchor.asset_id);}}
    if(action==='map')showMap(false);
    if(action==='history')showMap(true);
    if(action==='run-history')void showRunHistory();
    if(action==='saved-run')void showSavedRun(target.dataset.id!);
    if(action==='imported-run'){const receipt=study.importedRuns?.[Number(target.dataset.index)];if(receipt){const body=resetContent();body.innerHTML=intro('跨端导入 · 只读收据','原设备的运行记录','这份记录由你选择的导出文件带入；没有在当前设备重新运行。大型产物保留在原设备。')+'<button data-study="run-history">返回运行记录</button>';renderJson(body,receipt.record);if(receipt.snapshot?.code){body.insertAdjacentHTML('beforeend','<h2>导出时保存的代码快照</h2>');appendText(body,receipt.snapshot.code);}}}
    if(action==='chapter'){chapter=pack!.chapters.find(c=>c.id===target.dataset.id)!;renderShell();overview();}
    if(action==='jump'){const href=target.dataset.href!;const pane=origin!.pane;void closeStudy().then(()=>host.jump(href,pane));}
    if(action==='run')void runActivity();
    if(action==='cancel'&&latestRun)void invokeBook('learning_cancel',{runId:latestRun.run_id}).catch(showError);
    if(action==='edit')void enableDraft(false);
    if(action==='blank')void enableDraft(true);
    if(action==='reference-code')void showReferenceCode();
    if(action==='locate-code')locateCode(target.dataset.symbol!);
    if(action==='focus-code'){
      const grid=one('.study-lab-grid'),main=one('.study-main'),editor=one<HTMLTextAreaElement>('.study-code');
      const frame=main.getBoundingClientRect(),editorBefore=editor.getBoundingClientRect();
      const anchor=editorBefore.top>=frame.top&&editorBefore.top<frame.bottom?editor:target;
      const top=anchor.getBoundingClientRect().top;
      const focused=grid.classList.toggle('study-code-focused');target.textContent=focused?'回到结果与代码':'专注看代码';
      requestAnimationFrame(()=>{if(!grid.isConnected)return;main.scrollTop+=anchor.getBoundingClientRect().top-top;const visible=editor.getBoundingClientRect(),bounds=main.getBoundingClientRect();if(visible.top>bounds.bottom-80)main.scrollTop+=visible.top-(bounds.bottom-80);else if(visible.bottom<bounds.top+80)main.scrollTop+=visible.bottom-(bounds.top+80);});
    }
    if(action==='final-result'&&latestReport){showingTrace=false;renderResult(latestReport,displayedResultIdentity);if(trace.length)renderTraceControls();}
    if(action==='error-line'){const editor=one<HTMLTextAreaElement>('.study-code');const line=Number(target.dataset.line);const lines=editor.value.split('\n');const start=lines.slice(0,line-1).reduce((n,s)=>n+s.length+1,0);editor.focus();editor.setSelectionRange(start,start+(lines[line-1]?.length??0));editor.scrollTop=Math.max(0,line-3)*22;}
    if(action==='save')void saveDraft();
    if(action==='draft-history')void showDraftHistory().catch(error=>showError(error,'draft-history'));
    if(action==='export-code')void saveDraft().then(()=>invokeBook<string|null>('learning_export',{activityId:activity!.id})).then(path=>status(path?'已导出：'+path:'已取消导出')).catch(showError);
    if(action==='reference')void showReference();
    if(action==='my-result'&&latestRun)void showRun(latestRun,true);
    if(action==='step')stepTrace(1);
    if(action==='step-back')stepTrace(-1);
    if(action==='add-note')void addNote();
    if(action==='continue-unfinished')void continueUnfinishedNote(target.dataset.importId!);
    if(action==='save-note')void persistNote();
    if(action==='export-note')void exportNotes();
    if(action==='edit-note')void editNote(target.dataset.id!);
    if(action==='archive-note')void archiveNote(target.dataset.id!);
    if(action==='restore-note')void restoreNote(target.dataset.id!);
    if(action==='audio-group')void showAudio(target.dataset.id!);
    if(action==='video')void showVideo();
    if(action==='artifact')void showArtifact(target.dataset.path!);
  });
}

export function wireLearningDocument(bookId:string,pane:number,contents:any,sectionHref:string):void{
  const p=packs.get(bookId);if(!p)return;
  const doc=contents.document as Document;
  if(doc.documentElement.dataset.learningWired===bookId)return;doc.documentElement.dataset.learningWired=bookId;
  doc.addEventListener('click',event=>{
    const a=(event.target as Element).closest<HTMLAnchorElement>('a[href]');if(!a||!event.isTrusted)return;
    const raw=a.getAttribute('href')??'';
    const normalized=raw.replace(/^.*\/OEBPS\//,'').replace(/^OEBPS\//,'');
    const id=Object.prototype.hasOwnProperty.call(p.href_assets,normalized)?p.href_assets[normalized]:undefined;
    const activityId=p.href_activities?.[normalized];
    if(!id&&!activityId&&!/^https?:\/\//.test(raw))return;
    event.preventDefault();event.stopImmediatePropagation();
    const current=host.current();if(!current)return;
    const selection=doc.getSelection();
    const quote=a.closest('p,li,figcaption')?.textContent??a.textContent??'';
    let cfi=current.cfi;
    try{const range=doc.createRange();range.selectNodeContents(a.closest('p,li,figcaption')??a);range.collapse(true);cfi=contents.cfiFromRange(range,'reader-annotation');}catch{}
    const o:Origin={...current,bookId,pane,cfi,href:sectionHref,quote,focus:a,selection:selection?.rangeCount?selection.getRangeAt(0).cloneRange():null};
    void openStudy(o,id,/^https?:\/\//.test(raw)?raw:undefined,activityId);
  },true);
  // Styling existing links changes no content tree or source EPUB bytes.
  for(const a of Array.from(doc.querySelectorAll<HTMLAnchorElement>('a[href]'))){
    const raw=a.getAttribute('href')??'';const id=p.href_assets[raw];
    if(id){const asset=p.assets.find(x=>x.id===id);a.title=`${asset?.kind==='code'?'在书中查看并实践':asset?.kind==='audio'?'听这一段':asset?.kind==='video'?'观看这段视频':'在书中查看'} · ${asset?.title??''}`;a.classList.add('study-inline-link');}
  }
  const style=doc.createElement('style');style.textContent='.study-inline-link{text-decoration-thickness:1px;text-underline-offset:.23em} .study-inline-link:focus-visible{outline:2px solid #bb8e55;outline-offset:3px}';doc.head?.append(style);
}

function showStudyLoadError(error:unknown):void{
  dialog.innerHTML=`<header class="study-header"><strong>随书学习 · 记录待核对</strong><button class="study-return" data-study="close">返回书页 <kbd>Esc</kbd></button></header><main class="study-main"><div class="study-content"><h1>学习记录暂时无法读取</h1><p>学习空间暂停写入，原有记录不会被空白状态替换。可以先另存原始副本；这不会修复或覆盖原文件，核对后再恢复。</p><button data-study="backup-raw">保存原始记录副本</button><p class="study-run-status" role="alert"></p></div></main>`;
  dialog.querySelector<HTMLElement>('.study-run-status')!.textContent=readableError(error);
  window.dispatchEvent(new CustomEvent('reader-study-open'));workspace.show();one<HTMLButtonElement>('[data-study="close"]').focus();
}
async function openStudy(o:Origin,assetId?:string,url?:string,activityId?:string):Promise<void>{
  if(workspace.open){await closeStudy();if(workspace.open)return;}
  const p=packs.get(o.bookId);if(!p)return;const opening=++studyOpenGeneration;
  origin=o;pack=p;chapter=chapterOf(o.href);latestRun=null;studyLoaded=false;study={notes:[],media:{}};
  try{
    const loaded=await invoke<any>('learning_load_state',{bookId:o.bookId});
    if(opening!==studyOpenGeneration||origin?.bookId!==o.bookId)return;
    if(!loaded||typeof loaded!=='object'||Array.isArray(loaded)||
       (loaded.notes!=null&&!Array.isArray(loaded.notes))||
       (loaded.archived_notes!=null&&!Array.isArray(loaded.archived_notes))||
       (loaded.unfinished_notes!=null&&!Array.isArray(loaded.unfinished_notes))||
       (loaded.media!=null&&(typeof loaded.media!=='object'||Array.isArray(loaded.media))))throw new Error('学习记录结构无效');
    if((loaded.notes??[]).some((note:any)=>!note||typeof note!=='object'||Array.isArray(note))||
       (loaded.archived_notes??[]).some((note:any)=>!note||typeof note!=='object'||Array.isArray(note)))throw new Error('笔记记录结构无效');
    const validUnfinished=(note:any)=>note&&typeof note==='object'&&!Array.isArray(note)&&typeof note.text==='string';
    if((loaded.unfinished_note!=null&&!validUnfinished(loaded.unfinished_note))||
       (loaded.unfinished_notes??[]).some((note:any)=>!validUnfinished(note)))throw new Error('未提交笔记结构无效');
    for(const note of loaded.unfinished_notes??[]){
      const id=await unfinishedNoteIdentity(note);
      if(note.import_id!==undefined&&note.import_id!==id)throw new Error('未提交笔记身份与内容不符');
      note.import_id=id;
    }
    if(opening!==studyOpenGeneration||origin?.bookId!==o.bookId)return;
    study=loaded;study.notes??=[];study.media??={};study.unfinished_notes??=[];studyLoaded=true;
  }catch(error){
    if(opening===studyOpenGeneration&&origin?.bookId===o.bookId)showStudyLoadError(error);return;
  }
  if(o.localSource&&o.contentDigest){let migrated=false;for(const note of study.notes){if(!note.content_digest&&note.book_revision===o.sourceSha256){note.content_digest=o.contentDigest;migrated=true;}}if(migrated){try{await persistStudy();}catch(error){studyLoaded=false;showStudyLoadError(error);return;}}}
  renderShell();window.dispatchEvent(new CustomEvent('reader-study-open'));workspace.show();
  if(activityId)await showActivity(activityId);else if(assetId)await showAsset(assetId);else if(url)showSources(url);else overview();
  one<HTMLButtonElement>('[data-study="close"]').focus();
}
function renderShell():void{
  if(!chapter||!pack)return;
  dialog.innerHTML=`<header class="study-header"><div class="study-brand"><span class="study-monogram">阅</span><div><span class="study-eyebrow">随书学习</span><strong>${enc(chapter.title)}</strong></div></div><button class="study-return" data-study="close">返回书页 <kbd>Esc</kbd><span aria-hidden="true">↗</span></button></header><div class="study-layout"><nav class="study-nav" aria-label="学习材料"><p class="study-nav-heading">此刻，顺着问题走</p><button data-study="overview"><span>01</span> 本章内容</button><button data-study="sources"><span>↗</span> 资料与依据</button><button data-study="map"><span>⌘</span> 学习地图</button><button data-study="notes"><span>✎</span> 本书笔记 <i>${study.notes.length||''}</i></button><button data-study="run-history"><span>↺</span> 运行记录</button><div class="study-nav-line"></div><p class="study-nav-heading">${chapter.number===0?'导读':`第 ${chapter.number} 章`} · 手边的材料</p><details class="study-nav-materials"><summary>本章材料</summary><div class="study-resource-nav"></div></details><div class="study-nav-foot">书页始终留在原处。<br/>按需打开，随时回来。</div></nav><main class="study-main"><div class="study-content"></div></main></div><footer class="study-footer"><span>本地保存 · 不自动运行 · 不自动上传</span><span class="study-save-state" role="status" aria-live="polite">准备就绪</span></footer>`;
  const nav=one('.study-resource-nav');
  const chapterAssets=pack.assets.filter(a=>a.root!=='derived'&&a.chapters.includes(chapter!.id));
  const codes=chapterAssets.filter(a=>a.kind==='code');
  for(const a of codes.slice(0,12)){const b=document.createElement('button');b.dataset.study='asset';b.dataset.id=a.id;b.innerHTML=`<span>⌁</span>${enc(a.title)}`;nav.append(b);}
  for(const group of pack.audio_groups.filter(g=>g.assets.some((id:string)=>chapterAssets.some(a=>a.id===id))))nav.insertAdjacentHTML('beforeend',`<button data-study="audio-group" data-id="${enc(group.id)}"><span>♪</span>${enc(group.title)}</button>`);
  if(chapterAssets.some(a=>a.kind==='video'))nav.insertAdjacentHTML('beforeend','<button data-study="video"><span>▷</span> 本章视频</button>');
}
function setStudyNavigation(tab:string):void{
  for(const button of Array.from(dialog.querySelectorAll<HTMLElement>('.study-nav>button[data-study]'))){
    if(button.dataset.study===tab)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');
  }
}
function resourceChapter(ids:string[]):void{
  if(!pack||!chapter||ids.some(id=>id.toLowerCase()===chapter!.id.toLowerCase()))return;
  const next=pack.chapters.find(c=>ids.some(id=>id.toLowerCase()===c.id.toLowerCase()));
  if(next){if(captureUnfinishedNote())void saveState();chapter=next;renderShell();}
}
function resetContent(tab='overview'):HTMLElement{
  setStudyNavigation(tab);
  if(captureUnfinishedNote())void saveState();
  pageGeneration++;resultViewRevision++;showingAuthorReference=false;disposeObject?.();disposeObject=null;disposeResponsive?.();disposeResponsive=null;
  audioAnalysisWorker?.terminate();audioAnalysisWorker=null;audioAnalyses.clear();
  if(audioGraph){void audioGraph.close();audioGraph=null;audioGain=null;}
  if(activeMedia){study.media[currentAsset?.id??'last']={time:activeMedia.currentTime};activeMedia.pause();activeMedia=null;void saveState();}
  for(const url of objectUrls)URL.revokeObjectURL(url);objectUrls=[];
  activity=null;currentAsset=null;referenceCode='';draftMode=false;
  const content=one('.study-content');content.replaceChildren();one('.study-main').scrollTop=0;return content;
}
const intro=(eyebrow:string,title:string,description:string)=>`<div class="study-intro"><p class="study-eyebrow">${enc(eyebrow)}</p><h1>${enc(title)}</h1><details class="study-intro-description" ${innerHeight>560?'open':''}><summary>说明与观察方法</summary><p class="study-lede">${enc(description)}</p></details></div>`;
function overview():void{
  const body=resetContent();if(!chapter||!pack)return;
  const items=pack.assets.filter(a=>a.root!=='derived'&&a.chapters.includes(chapter!.id));
  const activities=pack.activities.filter(a=>a.chapter===chapter!.id);
  body.innerHTML=intro(`第 ${String(chapter.number).padStart(2,'0')} 章 · 带着问题动手`,chapter.question||chapter.title,'先追一条计算，再改变一个条件。参考记录和你自己的运行，始终分别保存。');
  if(origin?.quote)body.insertAdjacentHTML('beforeend',`<blockquote class="study-origin"><span>来自你刚才读到的地方</span>${enc(origin.quote.slice(0,700))}</blockquote>`);
  if(activities.length)body.insertAdjacentHTML('beforeend',`<div class="study-feature-grid">${activities.map(a=>`<button class="study-feature" data-study="activity" data-id="${enc(a.id)}"><span class="study-eyebrow">${a.runtime==='builtin'?'内置计算':a.registered===false?'需要桌面运行环境':a.runtime==='torch'?'本机 PyTorch':'本机 Python'} · 按需实践</span><strong>${enc(a.title)}</strong><span>看计算 · 改一项 · 留下自己的结果 <b>↗</b></span></button>`).join('')}</div>`);
  if(pack.audio_groups.some(g=>g.assets.some((id:string)=>items.some(a=>a.id===id))))body.insertAdjacentHTML('beforeend',`<div class="study-feature-grid">${pack.audio_groups.filter(g=>g.assets.some((id:string)=>items.some(a=>a.id===id))).map(g=>`<button class="study-feature" data-study="audio-group" data-id="${enc(g.id)}"><span class="study-eyebrow">${g.assets.length} 段真实录音</span><strong>${enc(g.title)}</strong><span>${enc(g.description)}</span></button>`).join('')}</div>`);
  if(items.some(a=>a.kind==='video'))body.insertAdjacentHTML('beforeend','<button class="study-feature" data-study="video"><span class="study-eyebrow">本章视频</span><strong>让连续画面回答问题</strong><span>播放与逐帧观察 · 原件身份和速度分别标明 ↗</span></button>');
  body.insertAdjacentHTML('beforeend',`<div class="study-section-title"><h2>需要时，材料就在这里</h2><span>${items.length} 项</span></div><label class="study-search"><span>⌕</span><input placeholder="查找本章的程序、记录、图像或来源" aria-label="搜索本章资源"/></label><div class="study-asset-list"></div>`);
  const render=(q='')=>{one('.study-asset-list').innerHTML=items.filter(a=>[a.title,a.filename,a.relative_path].join(' ').toLowerCase().includes(q.toLowerCase())).map(a=>`<button class="study-asset-row" data-study="asset" data-id="${a.id}"><span class="study-kind">${label(a.kind)}</span><div><strong>${enc(a.title)}</strong><small>${enc(a.filename)} · ${formatBytes(a.bytes)}</small></div><span aria-hidden="true">↗</span></button>`).join('');};
  render();one<HTMLInputElement>('.study-search input').addEventListener('input',e=>render((e.target as HTMLInputElement).value));
}
async function bytesFor(a:Asset):Promise<ArrayBuffer>{return buffer(await invokeBook<ArrayBuffer|number[]>('learning_asset',{assetId:a.id}));}
async function urlFor(a:Asset,mime:string):Promise<string>{const url=URL.createObjectURL(new Blob([await bytesFor(a)],{type:mime}));objectUrls.push(url);return url;}
function details(a:Asset):string{const portable=!isDesktop||pack?.portable;return `<details class="study-details"><summary>版本、原件与资源标识</summary><dl><dt>身份</dt><dd>${a.role==='external_model'?'外部已训模型结果':'作者已有资料；不是本次运行'}</dd><dt>书籍内相对路径</dt><dd><code>${enc(a.relative_path)}</code></dd><dt>SHA-256</dt><dd><code>${enc(a.sha256)}</code></dd><dt>大小</dt><dd>${formatBytes(a.bytes)}</dd></dl><button data-study="copy-path" data-id="${a.id}">${portable?'复制书籍内资源路径':'复制资源路径'}</button></details>`;}
async function showAsset(id:string):Promise<void>{
  const a=assetById(id);if(!a)return;resourceChapter(a.chapters??[]);const chapterTargets=pack!.chapters.filter(c=>c.source_sha256===a.sha256);if(a.kind==='text'&&a.filename.endsWith('.md')&&chapterTargets.length===1){const target=chapterTargets[0].href;const pane=origin!.pane;await closeStudy();await host.jump(target,pane);return;}
  const recipe=pack!.activities.find(x=>x.entry_asset===id);if(recipe){await showActivity(recipe.id);return;}
  if(a.kind==='audio'){let g=pack!.audio_groups.find(g=>g.assets.includes(id));if(!g){g={id:'single-'+a.id,title:a.title,description:'这份音频由当前材料引用。手动播放，返回时保留位置；不会自动发声。',assets:[id]};pack!.audio_groups.push(g);}await showAudio(g.id,id);return;}
  if(a.kind==='video'){await showVideo(id);return;}
  resourceChapter(a.chapters??[]);const body=resetContent();currentAsset=a;const generation=pageGeneration;
  body.innerHTML=intro(label(a.kind),a.title,'保持原件内容，按当前问题阅读。退出后回到原句。')+'<div class="study-object">正在读取已登记原件…</div>'+details(a);
  if(['model','data'].includes(a.kind)){one('.study-object').innerHTML=`<p>这是已登记的数据或模型原件。计算活动使用自己的输入副本；这里按需显示身份，不提前把大型文件读进界面。</p><p>原件大小：${formatBytes(a.bytes)}。可以在当前章节的来源说明与参考记录中核对其用途。</p>`;return;}
  try{
    if(a.kind==='geometry'){
      const root=a.relative_path.slice(0,a.relative_path.lastIndexOf('/'));const peers=pack!.assets.filter(x=>x.kind==='geometry'&&x.root===a.root&&x.relative_path.slice(0,x.relative_path.lastIndexOf('/'))===root);
      if(a.comparison_reference)peers.sort((x,y)=>Number(y.id===a.comparison_reference)-Number(x.id===a.comparison_reference));
      const object=one('.study-object');study.geometry??={};const key=peers[0].id;
      const cleanup=await mountGeometry(object,peers,a.id,id=>bytesFor(assetById(id)!),study.geometry[key],view=>{study.geometry[key]=view;saveViewSoon();},e=>resourceUnavailable(object,e),id=>{if(generation===pageGeneration){const selected=assetById(id)!;currentAsset=selected;one('.study-intro h1').textContent=selected.title;const detail=body.querySelector(':scope > .study-details');if(detail)detail.outerHTML=details(selected);}});
      if(generation!==pageGeneration){cleanup();return;}disposeObject=cleanup;return;
    }
    const bytes=await bytesFor(a);if(generation!==pageGeneration)return;
    const object=one('.study-object');object.replaceChildren();
    if(a.kind==='image'){const url=URL.createObjectURL(new Blob([bytes]));objectUrls.push(url);study.images??={};const cleanup=await mountImage(object,url,a.title,study.images[a.id],view=>{study.images[a.id]=view;saveViewSoon();});if(generation!==pageGeneration){cleanup();return;}disposeObject=cleanup;}
    else if(a.kind==='pdf'){const cleanup=await mountPdf(object,bytes,a.title,showError);if(generation!==pageGeneration){cleanup();return;}disposeObject=cleanup;}
    else {const value=new TextDecoder().decode(bytes);if(a.filename.endsWith('.json')){try{renderJson(object,JSON.parse(value));}catch{appendText(object,value);}}else if(a.kind==='code'){appendText(object,value,'study-code-readonly');object.insertAdjacentHTML('afterbegin','<p class="study-callout">完整参考源码；保留原始代码行，可上下、左右滚动。这里不会自动执行或启动训练。</p>');}else if(/\.(csv|tsv)$/i.test(a.filename)){try{renderDelimited(object,value,/\.tsv$/i.test(a.filename)?'\t':',');}catch{appendText(object,value,'study-code-readonly');}}else renderText(object,value);}
    const footer=dialog.querySelector<HTMLElement>('.study-save-state');if(footer?.dataset.statusOwner==='resource')status('资料已载入');
  }catch(e){if(generation===pageGeneration){const object=dialog.querySelector<HTMLElement>('.study-object');if(object){object.replaceChildren();resourceUnavailable(object,e);}else showError(e);}}
}
function resourceUnavailable(parent:HTMLElement,error:unknown):void{
  parent.querySelector('.study-resource-error')?.remove();const message=document.createElement('section');message.className='study-resource-error';
  message.innerHTML='<p class="study-callout">这份原件暂时无法使用。正文与笔记仍然保留；恢复原文件后重新选择这一项即可重试。若文件已换版，请重新核对增强包。</p><details class="study-details"><summary>查看具体原因</summary><pre></pre></details>';
  message.querySelector('pre')!.textContent=readableError(error);parent.append(message);status('资料暂不可用 · 可以返回书页','resource');
}
function appendText(parent:HTMLElement,text:string,className='study-raw'):void{const pre=document.createElement('pre');pre.className=className;pre.textContent=text;parent.append(pre);}
function renderText(parent:HTMLElement,text:string):void{
  renderReadableText(parent,text,url=>{if(isDesktop)void invoke('learning_open_external',{url}).catch(showError);else window.open(url,'_blank','noopener,noreferrer');});
}
function renderJson(parent:HTMLElement,value:any):void{
  const scalar=Object.entries(value??{}).filter(([,v])=>typeof v!=='object');
  if(scalar.length){const table=document.createElement('table');table.className='study-table study-record-table';table.innerHTML='<thead><tr><th>记录字段</th><th>实际值</th></tr></thead><tbody>'+scalar.map(([k,v])=>`<tr><td>${enc(k)}</td><td>${enc(v)}</td></tr>`).join('')+'</tbody>';parent.append(table);}
  const detail=document.createElement('details');detail.className='study-details';detail.innerHTML='<summary>查看完整结构化记录</summary>';appendText(detail,JSON.stringify(value,null,2));parent.append(detail);
}

async function showBuiltinActivity(a:Activity):Promise<void>{
  const body=resetContent();activity=a;currentAsset=assetById(a.entry_asset)??null;referenceCode=builtinSource;latestRun=null;latestReport=null;
  body.innerHTML=intro('内置计算 · 按你选择的参数',a.title,a.description??'')+`<div class="study-activity-toolbar"><button class="study-primary" data-study="run">运行计算</button><button data-study="cancel" hidden>停止运行</button><button data-study="my-result" disabled>查看已保存结果</button><span class="study-runtime-badge">阅读器内置 · 无需 Python</span></div><div class="study-parameters">${(a.parameters??[]).map(p=>`<label>${enc(p.label??p.name)}<input type="number" data-parameter="${enc(p.name)}" data-parameter-type="${enc(p.type)}" value="${p.default}" min="${p.min}" max="${p.max}" step="${p.type==='integer'?'1':'any'}"/></label>`).join('')}</div><p class="study-run-status" role="status">运行时只使用本活动的输入，不安装依赖。</p><div class="study-lab-grid"><section class="study-result-panel"><div class="study-section-title"><h2>观察与结果</h2><span class="study-result-identity">尚未运行</span></div><div class="study-result"><p>改变一个参数，观察预测、误差与更新轨迹。</p></div></section><section class="study-code-panel"><div class="study-section-title"><h2>这次运行的算法</h2><button data-study="focus-code">专注看代码</button></div><p class="study-code-guide">这里是阅读器内置计算的 JavaScript 源码。书中提供的 Python 参考文件可在材料列表按需查看；它不会被网页直接执行。</p><textarea class="study-code" wrap="off" readonly spellcheck="false" aria-label="内置算法源码"></textarea><div class="study-editor-footer"><span class="study-draft-status">内置算法 · 只读</span></div></section></div><details class="study-details study-log"><summary>计算日志</summary><pre></pre></details><div class="study-artifacts"></div><p class="study-code-guide">返回书页会让本次轻量计算继续；关闭页面会中断未结束的计算。参数变化只标记旧结果，不自动运行。</p>`;
  one<HTMLTextAreaElement>('.study-code').value=builtinSource;
  dialog.querySelectorAll('[data-parameter]').forEach(el=>el.addEventListener('change',markStale));
  const previous=study.lastRuns?.[a.id];if(previous){try{latestRun=await invokeBook('learning_run_status',{runId:previous});await showRun(latestRun,true);}catch(error){showError(error);}}
}
function renderLeastSquares(body:HTMLElement,value:any):void{
  if(!Array.isArray(value.predictions)||!Array.isArray(value.trace)||!Number.isFinite(value.weights?.w)||!Number.isFinite(value.weights?.b)){renderJson(body,value);return;}
  body.innerHTML=`<canvas class="study-regression-plot" width="840" height="480" role="img" aria-label="输入记录与拟合直线"></canvas><div class="study-metrics"><div><span>运行结束的斜率</span><strong>${Number(value.weights.w).toFixed(5)}</strong></div><div><span>运行结束的截距</span><strong>${Number(value.weights.b).toFixed(5)}</strong></div></div><label class="study-replay-control">回看这次计算的步骤<input type="range" min="1" max="${value.trace.length}" value="${value.trace.length}" aria-label="回放已保存计算"/></label><p class="study-replay-value"></p><table class="study-table"><thead><tr><th>输入 x</th><th>目标 y</th><th>结束时的预测</th></tr></thead><tbody>${value.predictions.map((r:any)=>`<tr><td>${enc(r.x)}</td><td>${enc(r.target)}</td><td>${Number(r.prediction).toFixed(5)}</td></tr>`).join('')}</tbody></table>`;
  const canvas=body.querySelector<HTMLCanvasElement>('canvas')!,context=canvas.getContext('2d')!;
  const xs=value.predictions.map((r:any)=>r.x),ys=value.predictions.flatMap((r:any)=>[r.target,r.prediction]);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys,0),ymax=Math.max(...ys,1),dx=xmax-xmin||1,dy=ymax-ymin||1;
  const x=(n:number)=>65+(n-xmin)/dx*700,y=(n:number)=>405-(n-ymin)/dy*340;
  const draw=(index:number)=>{
    const step=value.trace[index],w=step.after.w,b=step.after.b;context.clearRect(0,0,840,480);context.lineWidth=2;context.strokeStyle='#91968a';context.beginPath();context.moveTo(55,45);context.lineTo(55,420);context.lineTo(790,420);context.stroke();context.font='22px system-ui';context.fillStyle='#858a80';context.fillText('x',790,450);context.fillText('y',25,35);
    context.save();context.beginPath();context.rect(55,30,735,390);context.clip();context.strokeStyle='#c58b52';context.lineWidth=3;context.beginPath();context.moveTo(x(xmin),y(w*xmin+b));context.lineTo(x(xmax),y(w*xmax+b));context.stroke();
    for(const r of value.predictions){context.fillStyle='#6e9f9b';context.beginPath();context.arc(x(r.x),y(r.target),8,0,Math.PI*2);context.fill();}context.restore();
    body.querySelector('.study-replay-value')!.textContent=`轨迹回放 ${index+1} / ${value.trace.length}：更新后的斜率 ${w.toFixed(5)}，截距 ${b.toFixed(5)}；这步更新前的均方误差 ${step.loss_before.toFixed(5)}。表格仍显示运行结束时的预测。`;
  };
  draw(value.trace.length-1);body.querySelector('input')!.addEventListener('input',e=>draw(Number((e.target as HTMLInputElement).value)-1));
}
async function showActivity(id:string):Promise<void>{
  if(activity&&draftMode)await saveDraft();
  const a=pack!.activities.find(x=>x.id===id);if(!a)return;
  if(a.runtime==='builtin'){await showBuiltinActivity(a);return;}
  const body=resetContent();activity=a;currentAsset=assetById(a.entry_asset)!;const generation=pageGeneration;
  latestReport=null;latestRun=null;trace=[];traceIndex=-1;linearVisualization=true;
  const description=a.description||'从当前材料的输入与问题开始，运行已登记的实验。结果与当前代码版本一同保存。';
  body.innerHTML=intro(a.presentation==='causal-batch@1'?'先看一批 · 完整训练另行选择':'从书中的计算，到你手里的结果',a.title,description)+`<div class="study-activity-toolbar"><button class="study-primary" data-study="run" disabled>${a.presentation==='causal-training@1'?'检查预算并开始训练':a.presentation==='causal-batch@1'?'运行真实 batch 探针':'运行本章程序'}</button><button data-study="cancel" hidden>停止运行</button><button data-study="edit" ${a.editable===false||a.presentation==='causal-batch@1'?'disabled':''}>我来改</button><button data-study="blank" ${a.editable===false||a.presentation==='causal-batch@1'?'disabled':''}>从空白写</button><button data-study="reference">先看作者记录</button><button data-study="my-result" disabled>看我的记录</button><span class="study-runtime-badge">本机 ${a.runtime==='torch'?'PyTorch':'Python'} · ${a.presentation==='causal-training@1'?'按批准的设备':'CPU'}</span></div><div class="study-run-status" role="status" aria-live="polite">正在读取源码与上次记录；就绪后才可运行。</div><div class="study-lab-grid"><section class="study-result-panel"><div class="study-section-title"><h2>观察与结果</h2><span class="study-result-identity">尚未运行</span></div><div class="study-result"><div class="study-empty-result"><span>⌁</span><h3>先在心里作一个预测</h3><p>${a.presentation==='linear-classifier@1'?'步长改变时，分界线与最终参数会怎样变化？':a.presentation==='gradient-step@1'?'算出了梯度，是否已经改动了参数？':a.presentation==='causal-batch@1'?'每个位置能看到哪些输入？它应该预测哪个 token？':'先预测一个可以核对的结果，再运行和比较。'}</p></div></div></section><section class="study-code-panel"><div class="study-section-title"><h2>计算写在哪里</h2><span class="study-code-mode">只读参考</span></div><p class="study-code-guide">${enc(a.reading_guide??'沿当前问题追踪输入、计算和输出。')}</p><textarea class="study-code" wrap="off" spellcheck="false" aria-label="Python 代码" readonly></textarea><div class="study-editor-footer"><span class="study-draft-status">正在读取原件…</span><button data-study="draft-history">草稿历史</button><button data-study="export-code">导出</button><button data-study="save" hidden>保存我的版本 <kbd>Ctrl S</kbd></button></div></section></div><details class="study-details"><summary>环境、资源预算与运行边界</summary><p>${a.presentation==='causal-training@1'?'完整训练使用上方选择的设备，每次运行单独确认时间预算。':'当前使用已核对的本机 Python 环境与 CPU。'}每次有独立目录、${a.presentation==='causal-training@1'?'本次批准的时间':a.timeout_seconds+' 秒时间'}上限、${a.presentation==='causal-training@1'?8:4} GiB 进程组已提交内存上限；${a.presentation==='causal-training@1'?'不调用外部 API。':'未启动 GPU 或外部 API。'}返回书页会继续计算，关闭应用会结束本机任务。Windows Job Objects 管理取消与退出，<strong>不是文件或网络沙箱</strong>。参考配方只使用登记的输入副本。修改代码需明确授权当前版本的高信任本机执行。</p></details><details class="study-details study-log"><summary>运行日志与错误定位（每路最多 2 MiB）</summary><pre></pre></details><div class="study-artifacts"></div>`;
  if(a.parameters?.length){const fields=document.createElement('div');fields.className='study-parameters';for(const field of a.parameters){const label=document.createElement('label');const title=document.createElement('span');title.textContent=field.label||field.name;label.append(title);const control=document.createElement(field.type==='enum'?'select':'input') as HTMLInputElement|HTMLSelectElement;control.dataset.parameter=field.name;if(/^[a-z][a-z0-9-]*$/.test(field.name))control.className='study-'+field.name;control.dataset.parameterType=field.type;if(field.type==='enum'){for(const value of field.choices??[]){const option=document.createElement('option');option.value=String(value);option.textContent=String(field.choiceLabels?.[String(value)]??value);control.append(option);}}else{const input=control as HTMLInputElement;input.type=field.type==='boolean'?'checkbox':field.type==='string'?'text':'number';if(field.min!==undefined)input.min=String(field.min);if(field.max!==undefined)input.max=String(field.max);if(field.type==='boolean')input.checked=Boolean(field.default);if(['number','integer'].includes(field.type))input.step=field.type==='integer'?'1':'any';if(field.max_bytes)input.maxLength=Math.min(field.max_bytes,10000);}control.value=String(field.default??'');control.addEventListener('input',markStale);label.append(control);fields.append(label);}one('.study-activity-toolbar').after(fields);}
  if(!isDesktop){one('.study-runtime-badge').textContent='网页 · 源码与参考材料';one<HTMLButtonElement>('[data-study="edit"]').disabled=false;one<HTMLButtonElement>('[data-study="blank"]').disabled=false;const info=one('.study-content').querySelector<HTMLElement>(':scope > .study-details p');if(info)info.textContent='这项活动需要桌面的本机解释器与经过审查的运行配置。网页可阅读源码、保存个人草稿和查看参考材料；不会安装依赖或执行 Python。可导出草稿，在桌面中核对同一书籍版本后继续。';}
  one('.study-editor-footer').insertAdjacentHTML('beforeend','<button data-study="reference-code">读参考源码</button>');
  const more=document.createElement('details');more.className='study-activity-more';more.innerHTML='<summary>参考与练习</summary>';for(const action of ['blank','reference','my-result'])more.append(one(`[data-study="${action}"]`));one('.study-activity-toolbar').insertBefore(more,one('.study-runtime-badge'));
  const focus=document.createElement('button');focus.dataset.study='focus-code';focus.textContent='专注看代码';one('.study-code-panel .study-section-title').append(focus);
  const guide=a.practice;
  if(guide){
    one('.study-code-guide').insertAdjacentHTML('afterend',`<div class="study-symbol-links" aria-label="解释与源码对应">${guide.symbols.map(([label,symbol])=>`<button data-study="locate-code" data-symbol="${enc(symbol)}">${enc(label)} ↗</button>`).join('')}</div>`);
    one('.study-code-panel').insertAdjacentHTML('beforeend',`<details class="study-details study-practice-guide"><summary>自己写：任务与按需提示</summary><p>${enc(guide.task)}</p>${guide.hints.map((hint,i)=>`<details class="study-hint"><summary>${['轻提示','进一步提示','核对更新规则'][i]}</summary><p>${enc(hint)}</p></details>`).join('')}<small>学习界面补充说明 · 0.4 版 · 不改写原书或参考程序</small></details>`);
  }
  if(a.presentation==='causal-training@1')one('.study-activity-toolbar').insertAdjacentHTML('afterend',`<div class="study-training-budget"><label>本次设备<select class="study-train-device"><option value="cuda">本机 CUDA GPU</option><option value="cpu">CPU</option></select></label><label>最长运行<select class="study-train-budget"><option value="60">1 分钟</option><option value="180" selected>3 分钟</option><option value="600">10 分钟</option></select></label><label><input type="checkbox" class="study-resume" ${study.lastRuns?.[a.id]?'':'disabled'}/> 从上次已保存的检查点恢复</label><p>实际耗时与显存占用以本次运行记录为准。主机已提交内存限制为 8 GiB，未宣称硬隔离显存。没有新下载和外部 API。只准备已登记训练/验证数据，测试集不进入训练。</p></div>`);
  try{referenceCode=new TextDecoder().decode(await bytesFor(currentAsset));if(generation!==pageGeneration)return;one<HTMLTextAreaElement>('.study-code').value=referenceCode;const keyLine=a.key_symbol?referenceCode.split('\n').findIndex(line=>line.includes(a.key_symbol!)):-1;one<HTMLTextAreaElement>('.study-code').scrollTop=Math.max(0,keyLine-2)*22;one('.study-draft-status').textContent='参考版本 · 原件只读';
    const draft=await invokeBook<any>('learning_draft',{activityId:a.id});if(generation!==pageGeneration)return;if(draft){one('.study-draft-status').textContent='已有你的独立版本 · 点“我来改”继续';}
  }catch(e){showError(e);}
  one<HTMLTextAreaElement>('.study-code').addEventListener('input',()=>{markStale();one('.study-draft-status').textContent='正在保存你的修改…';if(saveTimer)clearTimeout(saveTimer);saveTimer=window.setTimeout(()=>void saveDraft(),550);});
  one<HTMLTextAreaElement>('.study-code').addEventListener('keydown',e=>{if(e.ctrlKey&&e.code==='BracketRight'&&!(e.currentTarget as HTMLTextAreaElement).readOnly){e.preventDefault();const t=e.currentTarget as HTMLTextAreaElement;const start=t.selectionStart;t.setRangeText('    ',start,t.selectionEnd,'end');t.dispatchEvent(new Event('input'));}});
  for(const control of Array.from(dialog.querySelectorAll('.study-parameters input,.study-parameters select')))control.addEventListener('input',markStale);
  const previous=study.lastRuns?.[a.id];if(previous){try{const savedRun=await invokeBook<any>('learning_run_status',{runId:previous});if(generation!==pageGeneration)return;latestRun=savedRun;one<HTMLButtonElement>('[data-study="my-result"]').disabled=false;if(!showingAuthorReference)await showRun(latestRun,true);}catch(e){if(generation!==pageGeneration)return;one('.study-run-status').textContent=`上次运行记录暂不可用：${String(e)}`;}}
  if(generation!==pageGeneration)return;one<HTMLButtonElement>('[data-study="run"]').disabled=latestRun?.status==='running'||a.registered===false;if(a.registered===false)one('.study-run-status').textContent='此活动还未完成本机环境与配方登记。源码和参考材料仍可阅读，不会假装已经接通运行。';else if(!previous)one('.study-run-status').textContent='准备好了。点击时才计算；作者原文件保持完整。';
}
function markStale():void{if(latestReport||latestRun){one('.study-result-identity').textContent=(displayedResultIdentity||'已保存结果')+' · 输入已改变';one('.study-result-panel').classList.add('study-stale');one('.study-run-status').textContent='参数或草稿已改变。上方标明的旧结果仍然保留；点击运行才会计算新结果。';} }
function sameActivity(bookId:string,activityId:string,generation:number,editor:HTMLTextAreaElement):boolean{
  return workspace.open&&origin?.bookId===bookId&&activity?.id===activityId&&pageGeneration===generation&&dialog.querySelector('.study-code')===editor;
}
async function enableDraft(blank:boolean):Promise<void>{
  if(!activity||!origin)return;
  const a=activity,bookId=origin.bookId,generation=pageGeneration,editor=dialog.querySelector<HTMLTextAreaElement>('.study-code');
  if(!editor)return;
  try{
    const draft=await invoke<any>('learning_draft',{bookId,activityId:a.id});
    if(!sameActivity(bookId,a.id,generation,editor))return;
    if(blank&&draft?.code.trim()){
      await invoke('learning_save_draft',{bookId,activityId:a.id,code:draft.code});
      if(!sameActivity(bookId,a.id,generation,editor))return;
    }
    draftMode=true;editor.readOnly=false;
    const code=blank?a.practice?.starter??`# 我的独立练习：${a.title}\n# 先确定输入、计算与可核对的输出。\n\n`:draft?.code??referenceCode;
    editor.value=code;
    if(blank){const guide=dialog.querySelector<HTMLDetailsElement>('.study-practice-guide');if(guide)guide.open=true;}
    one('.study-code-mode').textContent='我的独立版本';one<HTMLButtonElement>('[data-study="save"]').hidden=false;editor.focus();
    await invoke('learning_save_draft',{bookId,activityId:a.id,code});
    if(!sameActivity(bookId,a.id,generation,editor))return;
    one('.study-draft-status').textContent=editor.value===code?'我的版本已保存':'正在保存你的修改…';markStale();
  }catch(error){if(sameActivity(bookId,a.id,generation,editor))showError(error);}
}
async function showReferenceCode():Promise<void>{
  if(!activity||!origin)return;const a=activity,bookId=origin.bookId,generation=pageGeneration,editor=one<HTMLTextAreaElement>('.study-code');
  await saveDraft();if(!sameActivity(bookId,a.id,generation,editor))return;
  draftMode=false;editor.value=referenceCode;editor.readOnly=true;
  one('.study-code-mode').textContent='只读参考';one('.study-draft-status').textContent='原件只读 · 我的版本仍在草稿中';one<HTMLButtonElement>('[data-study="save"]').hidden=true;markStale();
}
function locateCode(symbol:string):void{
  const editor=one<HTMLTextAreaElement>('.study-code');const escaped=symbol.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  const candidates=[new RegExp(`(?:def )?\\b${escaped}(?:\\s*=|\\s*\\(|\\s*:)`),new RegExp(`\\b${escaped}\\b`)];
  const match=candidates.map(pattern=>pattern.exec(editor.value)).find(Boolean);
  if(!match){one('.study-draft-status').textContent=`当前版本没有 ${symbol}；可主动打开只读参考核对。`;return;}
  const start=match.index+match[0].indexOf(symbol);editor.focus();editor.setSelectionRange(start,start+symbol.length);
  const line=editor.value.slice(0,start).split('\n').length-1;editor.scrollTop=Math.max(0,line-4)*parseFloat(getComputedStyle(editor).lineHeight);
  let location=dialog.querySelector<HTMLElement>('.study-code-location');if(!location){location=document.createElement('p');location.className='study-code-location';editor.after(location);}location.textContent=`源码定位：${symbol} · 第 ${line+1} 行（阅读位置，不是执行中的一行）`;
}
async function saveDraft():Promise<void>{if(!activity||!origin||!draftMode)return;const editor=dialog.querySelector<HTMLTextAreaElement>('.study-code');if(!editor)return;const bookId=origin.bookId,activityId=activity.id,generation=pageGeneration,code=editor.value;try{await invoke('learning_save_draft',{bookId,activityId,code});if(sameActivity(bookId,activityId,generation,editor))one('.study-draft-status').textContent=editor.value===code?'我的版本已保存':'正在保存你的修改…';}catch(e){if(sameActivity(bookId,activityId,generation,editor))showError(e);}}
function currentActivityParameters(a:Activity):any{
    const params:any=a.presentation==='causal-training@1'?{device:one<HTMLSelectElement>('.study-train-device').value,budget_seconds:Number(one<HTMLSelectElement>('.study-train-budget').value),resume_from:one<HTMLInputElement>('.study-resume').checked?study.lastRuns?.[a.id]??'':''}:{};
    for(const input of Array.from(dialog.querySelectorAll<HTMLInputElement|HTMLSelectElement>('[data-parameter]'))){params[input.dataset.parameter!]=input.dataset.parameterType==='boolean'?(input as HTMLInputElement).checked:['number','integer'].includes(input.dataset.parameterType!)?Number(input.value):input.value;}
    return params;
}
async function runActivity():Promise<void>{
  if(!activity||!origin)return;const a=activity;const generation=pageGeneration;
  if(showingAuthorReference){showingAuthorReference=false;resultViewRevision++;latestReport=null;trace=[];traceIndex=-1;showPersonalRunDetails(true);one('.study-result').replaceChildren();one('.study-result-identity').textContent='尚未产生本次结果';one('.study-log pre').textContent='';one('.study-artifacts').replaceChildren();dialog.querySelector('.study-run-metadata')?.remove();dialog.querySelector('.study-artifact-detail')?.remove();}
  const button=one<HTMLButtonElement>('[data-study="run"]');if(button.disabled)return;button.disabled=true;
  one('.study-run-status').textContent='正在准备当前代码版本…';
  try{
    if(a.runtime==='builtin'){
      const session=portableBooks.get(origin.bookId)!;const declared=session.book.activities.find((x:any)=>x.id===a.id);
      let ready=true;for(const key of declared.required){try{await materialize(publicStore,session.book.resources.find((r:any)=>r.id===key));}catch{ready=false;}}
      if(!ready){one('.study-run-status').textContent='先准备这个活动的输入，之后再点击运行。';await showAcquisition(origin.bookId,'activity',a.id);return;}
    }
    await saveDraft();const code=draftMode?one<HTMLTextAreaElement>('.study-code').value:referenceCode;
    if(draftMode&&code.replace(/\r\n?/g,'\n')!==referenceCode.replace(/\r\n?/g,'\n')){
      const savedDraft=await invokeBook<any>('learning_draft',{activityId:a.id});const authorized=savedDraft?.authorized||await highTrustConsent();if(!authorized){one('.study-run-status').textContent='没有运行新代码。上次结果和当前草稿都已保留。';return;}
      await invokeBook('learning_authorize_edit',{activityId:a.id,codeSha256:await digest(code)});
    }
    if(!workspace.open||generation!==pageGeneration)return;
    const params=currentActivityParameters(a);
    const requestId=crypto.randomUUID();if(a.presentation==='causal-training@1'){if(!await trainingConsent(params)){one('.study-run-status').textContent='未启动训练。预算选择已保留。';return;}await invokeBook('learning_authorize_training',{activityId:a.id,requestId,params});}one('.study-run-status').textContent=a.runtime==='builtin'?'正在执行内置计算…':'正在冻结输入并启动本机环境…';
    latestRun=await invokeBook<any>('learning_run',{activityId:a.id,requestId,params,useDraft:draftMode,expectedRecipeRevision:a.execution_revision,expectedCodeSha256:draftMode?await digest(code):currentAsset?.sha256});
    activeRun=latestRun;refreshLearningButton();study.lastRuns??={};study.lastRuns[a.id]=latestRun.run_id;await saveState();
    one<HTMLButtonElement>('[data-study="cancel"]').hidden=false;
    one('.study-result-identity').textContent=latestReport?displayedResultIdentity+' · 新计算进行中':'正在计算，尚未产生结果';one('.study-result-panel').classList.toggle('study-stale',Boolean(latestReport));
    if(poll)clearInterval(poll);
    const runId=latestRun.run_id;const bookId=origin.bookId;
    let polling=false;poll=window.setInterval(async()=>{
      if(polling)return;polling=true;try{const result=await invoke<any>('learning_run_status',{bookId,runId});activeRun=result;refreshLearningButton();if(origin?.bookId===bookId&&activity?.id===a.id){latestRun=result;if(workspace.open&&!showingAuthorReference)await showRun(result);}
        if(result.status!=='running'){if(poll)clearInterval(poll);poll=null;}
      }catch(e){if(poll)clearInterval(poll);poll=null;showError(e);}finally{polling=false;}
    },500);
  }catch(e){showError(e);}finally{if(!latestRun||latestRun.status!=='running')button.disabled=false;}
}
async function trainingConsent(params:any):Promise<boolean>{
  return new Promise(resolve=>{const box=document.createElement('div');box.className='study-consent';box.innerHTML=`<section role="alertdialog" aria-modal="true" aria-labelledby="training-title"><p class="study-eyebrow">本次完整训练 · 明确预算</p><h2 id="training-title">${params.resume_from?'从已保存的算法状态继续':'开始本次训练'}</h2><p>设备：<strong>${params.device==='cuda'?'本机 CUDA GPU':'CPU'}</strong>。最长 <strong>${params.budget_seconds/60} 分钟</strong>；达到预算或点击停止会结束进程树，保留已经写好的检查点。</p><p>占用当前设备的计算资源。原书和作者结果保持完整；没有下载、付费或远程请求。GPU 显存没有独立硬上限，8 GiB 限制适用于本机进程组已提交内存。</p><div><button class="training-no">先不运行</button><button class="study-primary training-yes">批准这次预算并运行</button></div></section>`;dialog.append(box);const finish=(v:boolean)=>{cancelConsent=null;box.remove();resolve(v);};cancelConsent=()=>finish(false);box.querySelector('.training-no')!.addEventListener('click',()=>finish(false));box.querySelector('.training-yes')!.addEventListener('click',()=>finish(true));box.querySelector<HTMLButtonElement>('.training-no')!.focus();});
}
async function highTrustConsent():Promise<boolean>{
  return new Promise(resolve=>{const box=document.createElement('div');box.className='study-consent';box.innerHTML='<section role="alertdialog" aria-modal="true" aria-labelledby="trust-title"><p class="study-eyebrow">只授权当前保存的代码版本</p><h2 id="trust-title">在本机运行你修改的代码</h2><p>它以你当前的 Windows 用户身份运行，能访问该用户有权访问的文件和网络。独立副本、时间上限和停止按钮不能隔离这些权限。</p><p>仅运行你理解并信任的代码。书籍网页或来源资料不能替你作此授权。</p><div><button class="trust-no">返回检查代码</button><button class="study-primary trust-yes">我信任这一版，运行</button></div></section>';dialog.append(box);const finish=(v:boolean)=>{cancelConsent=null;box.remove();resolve(v);};cancelConsent=()=>finish(false);box.querySelector('.trust-no')!.addEventListener('click',()=>finish(false));box.querySelector('.trust-yes')!.addEventListener('click',()=>finish(true));box.querySelector<HTMLButtonElement>('.trust-no')!.focus();});
}
function showPersonalRunDetails(visible:boolean):void{
  for(const selector of ['.study-log','.study-artifacts','.study-run-metadata','.study-artifact-detail']){
    const element=dialog.querySelector<HTMLElement>(selector);if(element)element.hidden=!visible;
  }
}
async function showRun(run:any,restored=false):Promise<void>{
  if(!activity)return;
  const generation=pageGeneration,viewRevision=++resultViewRevision;
  if(showingAuthorReference){latestReport=null;trace=[];traceIndex=-1;one('.study-result').replaceChildren();}
  showingAuthorReference=false;showPersonalRunDetails(true);
  one<HTMLButtonElement>('[data-study="my-result"]').disabled=false;
  const status=({running:'本机正在计算；返回书页后会继续。需要结束时请点击“停止运行”。',succeeded:'这次运行已完成 · 结果与代码版本一同保存',failed:'程序报错了 · 原始错误与代码保留在下方',cancelled:'进程树已停止 · 草稿和已有输出保留',interrupted:'上次运行已中断 · 没有自动重跑'} as Record<string,string>)[run.status]??run.status;
  one('.study-run-status').textContent=run.status==='running'?status:'计算已结束，正在读取这一版结果…';one('.study-log pre').textContent=(run.stdout??'')+(run.stderr?'\n'+run.stderr:'');
  one<HTMLButtonElement>('[data-study="run"]').disabled=true;one<HTMLButtonElement>('[data-study="cancel"]').hidden=run.status!=='running';
  if(run.status==='running'){
    if(activity.presentation==='causal-training@1'&&run.latest_event){const e=run.latest_event;one('.study-result').innerHTML=`<div class="study-metrics"><div><span>当前轮次</span><strong>${enc(e.epoch)}</strong></div><div><span>真实更新次数</span><strong>${enc(e.global_step)}</strong></div><div><span>已保存检查点</span><strong>${enc(run.live_checkpoint?.global_step??'尚未写出')}</strong></div></div><p class="study-callout">${e.batch_nll_before_update!==undefined?'本批更新前 NLL：'+Number(e.batch_nll_before_update).toFixed(5):'这一轮已完成验证'}。停止时只保留已经完成写入的算法检查点。</p>`;}
    return;
  }
  const files=run.artifacts??[];
  one('.study-artifacts').innerHTML=files.length?`<div class="study-section-title"><h2>这次留下的结果</h2><span>${files.length} 个独立产物</span></div>${files.map((a:any)=>`<button class="study-artifact" data-study="artifact" data-path="${enc(a.path)}">${enc(a.path.split('/').pop())}<span>${formatBytes(a.bytes)} ↗</span></button>`).join('')}`:'';
  if(run.status==='succeeded'){
    const contract=files.find((f:any)=>f.path.endsWith('visualization-contract.json'));if(contract){const info=JSON.parse(new TextDecoder().decode(buffer(await invokeBook<ArrayBuffer|number[]>('learning_artifact',{runId:run.run_id,artifactPath:contract.path}))));if(generation!==pageGeneration||viewRevision!==resultViewRevision)return;linearVisualization=info.linear_score_matches_reference===true;}
    const main=activity.result_suffix?files.find((f:any)=>f.path.endsWith(activity!.result_suffix)):files.find((f:any)=>f.extension==='json');
    if(main){const raw=buffer(await invokeBook<ArrayBuffer|number[]>('learning_artifact',{runId:run.run_id,artifactPath:main.path}));if(generation!==pageGeneration||viewRevision!==resultViewRevision)return;latestReport=JSON.parse(new TextDecoder().decode(raw));renderResult(latestReport,`${restored?'保存的结果':'本次结果'} · ${run.run_id.slice(0,8)} · 代码 ${run.code_sha256.slice(0,8)}`);}else{one('.study-result').replaceChildren();appendText(one('.study-result'),run.stdout||'程序正常结束，没有打印输出。可以在自己的代码中打印正在核对的量，再检查它是否符合预期。');one('.study-result-identity').textContent=restored?'上次运行记录':'本次程序输出';}
    const track=files.find((f:any)=>f.path.endsWith('execution_trace.json'));if(track){const raw=buffer(await invokeBook<ArrayBuffer|number[]>('learning_artifact',{runId:run.run_id,artifactPath:track.path}));if(generation!==pageGeneration||viewRevision!==resultViewRevision)return;trace=JSON.parse(new TextDecoder().decode(raw));traceIndex=-1;if(linearVisualization)renderTraceControls();}
    if(activity.presentation==='causal-batch@1'){const batch=files.find((f:any)=>f.path.endsWith('visible_batch.json'));if(batch){const raw=buffer(await invokeBook<ArrayBuffer|number[]>('learning_artifact',{runId:run.run_id,artifactPath:batch.path}));if(generation!==pageGeneration||viewRevision!==resultViewRevision)return;const data=JSON.parse(new TextDecoder().decode(raw));renderBatch(data);}}
  }else if(run.status==='failed'){
    const stderr=String(run.stderr??'');const last=stderr.trim().split(/\r?\n/).filter(Boolean).slice(-1)[0]??'进程未正常结束。';
    const reason=/OutOfMemoryError|out of memory|MemoryError/.test(stderr)?'计算所需内存未能分配。先核对本次主机内存预算与设备占用，再决定是否重试。显存空闲量与 Windows 进程内存限制是不同的量。':/ModuleNotFoundError|ImportError/.test(stderr)?'当前解释器缺少程序需要的依赖。请先核对本章环境；不会自动安装或更换环境。':/SyntaxError|IndentationError/.test(stderr)?'Python 还没能读懂这段代码。检查所示位置的语法、括号和缩进，再保存重试。':'本次计算已停止。检查出错行使用的输入、形状和条件；完整原始错误保留在下方日志中。';
    const sourceName=(currentAsset?.filename??'').replace(/[.*+?^${}()|[\]\\]/g,'\\$&');const matches=sourceName?[...stderr.matchAll(new RegExp(sourceName+'", line (\\d+)','g'))]:[];const line=matches.slice(-1)[0]?.[1];
    one('.study-result').innerHTML=`<p class="study-callout">${enc(reason)}</p><pre class="study-raw">${enc(last)}</pre>${line?`<button data-study="error-line" data-line="${enc(line)}">定位源码第 ${enc(line)} 行</button>`:''}`;
    one('.study-result-identity').textContent='这次运行未完成';
    if(run.job_memory)one('.study-result').insertAdjacentHTML('beforeend',`<p class="study-code-guide">进程组峰值：${formatBytes(run.job_memory.peak_committed_bytes)}；本次上限：${formatBytes(run.job_memory.limit_bytes)}。这是主机已提交内存，不是显存。</p>`);
  }
  if(generation!==pageGeneration||viewRevision!==resultViewRevision)return;
  one('.study-run-status').textContent=run.diagnostic??status;
  one<HTMLButtonElement>('[data-study="run"]').disabled=false;
  let metadata=dialog.querySelector<HTMLElement>('.study-run-metadata');if(!metadata){metadata=document.createElement('details');metadata.className='study-details study-run-metadata';one('.study-content').append(metadata);}metadata.innerHTML='<summary>这次运行的版本、输入与环境</summary>';appendText(metadata,JSON.stringify({run_id:run.run_id,code_sha256:run.code_sha256,adapter_sha256:run.adapter_sha256,params:run.params,environment:run.environment||run.runtime,inputs:run.inputs,started_at:run.started_at,ended_at:run.ended_at},null,2));
  if(restored){one('.study-run-status').textContent='已恢复上次运行记录。修改参数或代码后，请主动重新运行。';one('.study-result-panel').classList.add('study-stale');}
  if(activity&&run.status==='succeeded'){const current=currentActivityParameters(activity);const changed=Object.keys(run.params??{}).some(key=>JSON.stringify(current[key])!==JSON.stringify(run.params[key]));const codeHash=await digest(draftMode?one<HTMLTextAreaElement>('.study-code').value:referenceCode);if(generation===pageGeneration&&viewRevision===resultViewRevision&&(changed||codeHash!==run.code_sha256))markStale();}
}
function renderResult(value:any,identity:string):void{
  displayedResultIdentity=identity;
  showingTrace=false;one('.study-result-panel').classList.remove('study-stale');
  one('.study-result-identity').textContent=identity;const body=one('.study-result');body.replaceChildren();
  const finite=(v:any)=>typeof v==='number'&&Number.isFinite(v);const numericArray=(v:any,n?:number)=>Array.isArray(v)&&(n===undefined||v.length===n)&&v.every(finite);
  if(activity?.capability==='least-squares@1'){renderLeastSquares(body,value);return;}
  if(activity?.presentation==='linear-classifier@1'&&(!linearVisualization||!numericArray(value.trained_weights,2)||!finite(value.trained_bias)||!numericArray(value.updates_by_epoch)||!Array.isArray(value.training_rows)||value.training_rows.length>10000||!value.training_rows.every((r:any)=>numericArray(r.features,2)&&[-1,1].includes(r.label)&&finite(r.score)&&finite(r.prediction)))){body.innerHTML=linearVisualization?'<p class="study-callout">程序已产生结果，但字段不符合这幅二维卡片图的约定。完整记录保留在下方，未把它强行画成原示例。</p>':'<p class="study-callout">你改变了 score 或 predict，原来的直线图不再自动套用。下方保留本次真实输出。</p>';renderJson(body,value);return;}
  if(activity?.presentation==='linear-classifier@1'){
    body.innerHTML=`<canvas class="study-plot" width="840" height="500" role="img" aria-label="训练卡片与当前参数对应的分界线"></canvas><div class="study-metrics"><div><span>运行结束的权重 w</span><strong>${enc(value.trained_weights?.join('，'))}</strong></div><div><span>运行结束的偏置 b</span><strong>${enc(value.trained_bias)}</strong></div><div><span>整次运行的轮数</span><strong>${enc(value.updates_by_epoch?.length)} 轮</strong></div></div><div class="study-trace"></div><details class="study-details"><summary>每轮的更新次数</summary><p>${enc(value.updates_by_epoch.join(' / '))}</p></details>`;drawCards(value.training_rows??[],value.trained_weights??[0,0],value.trained_bias??0);if(value.converged_within_budget===false)body.insertAdjacentHTML('beforeend',`<p class="study-callout">达到 ${enc(value.max_epochs)} 轮预算时仍有更新。这项观察本身不证明数据不可分。</p>`);body.insertAdjacentHTML('beforeend',`<p class="study-code-guide">本次使用 ${enc(value.training_rows?.length??0)} 张卡片。下表保留运行结束时的完整分数；单步图显示轨迹中的当前状态。</p>`);const table=document.createElement('table');table.className='study-table';table.innerHTML='<thead><tr><th>卡片</th><th>标签</th><th>分数</th><th>预测</th></tr></thead><tbody>'+(value.training_rows??[]).map((r:any)=>`<tr><td>(${enc(r.features.join(', '))})</td><td>${enc(r.label)}</td><td>${enc(r.score)}</td><td>${enc(r.prediction)}</td></tr>`).join('')+'</tbody>';body.append(table);
  }else if(activity?.presentation==='causal-training@1'){body.innerHTML=`<div class="study-metrics"><div><span>完成轮数</span><strong>${enc(value.completed_epochs)} / ${enc(value.planned_epochs)}</strong></div><div><span>优化器更新</span><strong>${enc(value.global_steps)}</strong></div><div><span>最佳验证 NLL</span><strong>${Number(value.best_validation_nll).toFixed(4)}</strong></div></div><p class="study-callout">训练与验证来自原来的划分。没有读取测试集；结果只支持这个模型和本次配置的结论。</p>`;renderJson(body,value);
  }else if(activity?.presentation==='gradient-step@1'){
    const v=value.module_optimizer_cpu64;const flatten=(x:any):number[]=>['a','c','v','b'].flatMap(k=>typeof x?.[k]==='number'?[x[k]]:Array.isArray(x?.[k])?x[k].flat(Infinity):[]);
    if(!v||![v.before_step,v.gradient,v.after].every(x=>numericArray(flatten(x),9))){body.innerHTML='<p class="study-callout">本次记录与原九参数模型的表格格式不同，完整结果保留如下。</p>';renderJson(body,value);return;}
    const before=flatten(v.before_step),gradient=flatten(v.gradient),after=flatten(v.after);const names=['a₁₁','a₁₂','a₂₁','a₂₂','c₁','c₂','v₁','v₂','b'];
    body.innerHTML=`<p class="study-callout">backward 计算梯度，参数仍是“更新前”；optimizer.step 才按这些梯度改动参数。</p><div class="study-metrics"><div><span>本批损失</span><strong>${Number(v.loss).toFixed(6)}</strong></div><div><span>实际输入 shape</span><strong>${enc(value.numpy_manual?.shapes?.x?.join(' × ')??'未提供')}</strong></div><div><span>环境</span><strong>PyTorch ${enc(value.torch_version)}</strong></div></div><table class="study-table"><thead><tr><th>参数</th><th>backward 后</th><th>梯度</th><th>step 后</th></tr></thead><tbody>${before.map((x,i)=>`<tr><td>${names[i]}</td><td>${x.toFixed(5)}</td><td>${gradient[i].toFixed(5)}</td><td>${after[i].toFixed(5)}</td></tr>`).join('')}</tbody></table>`;
  }else if(activity?.presentation==='causal-batch@1'){body.innerHTML=`<div class="study-metrics"><div><span>更新前 NLL</span><strong>${Number(value.nll_before_step).toFixed(5)}</strong></div><div><span>同批更新后 NLL</span><strong>${Number(value.nll_same_batch_after_step).toFixed(5)}</strong></div><div><span>真实目标数</span><strong>${enc(value.targets)}</strong></div></div><p class="study-callout">${enc(value.scope)}。同批损失的变化不能证明对未见文章泛化。</p>`;renderJson(body,value);}
  else if(activity?.presentation==='cached-generation@1'){
    body.innerHTML=`<p class="study-callout">同一份权重、同一请求与相同贪心选择。下列文本都由本次进程生成。</p><div class="study-request-result"><span>完整计算</span><p>${enc(value.outputs.full.generated_text)}</p><span>使用 KV 缓存</span><p>${enc(value.outputs.cached.generated_text)}</p><strong>token 序列一致：${value.same_token_ids?'是':'否'}</strong></div>`;renderJson(body,value);
  }else if(activity?.presentation==='tool-observation@1'){
    body.innerHTML=`<p class="study-callout">${enc(value.question)}</p><ol class="study-execution-flow"><li><span>模型生成的动作</span><pre>${enc(value.first?.text)}</pre></li><li><span>宿主实际返回</span><pre>${enc(value.observation??value.tool_error??'这次没有调用工具')}</pre></li><li><span>模型继续生成</span><pre>${enc(value.answer??'未产生有效回答')}</pre></li></ol><p>本任务成功：${value.task_success===true?'是':value.task_success===false?'否':'未提供'}。此处保留模型的真实成功或失败。</p>`;renderJson(body,value);
  }else{
    body.innerHTML='<p class="study-callout">本次在独立教学副本中完成。完整记录保留各阶段的输入、结果与边界。</p>';renderJson(body,value);
  }
}
function drawCards(rows:any[],weights:number[],bias:number):void{
  const canvas=one<HTMLCanvasElement>('.study-plot');if(!canvas)return;
  const ctx=canvas.getContext('2d')!,w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);
  const style=getComputedStyle(dialog),muted=style.getPropertyValue('--muted').trim()||'#888',ink=style.getPropertyValue('--text').trim()||'#222';
  const valid=rows.filter(r=>Array.isArray(r.features)&&r.features.length===2&&r.features.every(Number.isFinite));
  const loX=Math.min(0,...valid.map(r=>r.features[0])),hiX=Math.max(1,...valid.map(r=>r.features[0]));
  const loY=Math.min(0,...valid.map(r=>r.features[1])),hiY=Math.max(1,...valid.map(r=>r.features[1]));
  const spanX=hiX-loX,spanY=hiY-loY,xmin=loX-.14*spanX,xmax=hiX+.14*spanX,ymin=loY-.14*spanY,ymax=hiY+.14*spanY;
  const x=(v:number)=>65+(v-xmin)/(xmax-xmin)*(w-105),y=(v:number)=>h-45-(v-ymin)/(ymax-ymin)*(h-85);
  ctx.font='24px Segoe UI';ctx.strokeStyle=muted;ctx.globalAlpha=.2;
  for(let i=0;i<=3;i++){const gx=loX+i*spanX/3,gy=loY+i*spanY/3;ctx.beginPath();ctx.moveTo(x(gx),28);ctx.lineTo(x(gx),h-35);ctx.stroke();ctx.beginPath();ctx.moveTo(45,y(gy));ctx.lineTo(w-25,y(gy));ctx.stroke();}
  ctx.globalAlpha=1;ctx.fillStyle=muted;
  for(let i=0;i<=3;i++){const gx=loX+i*spanX/3,gy=loY+i*spanY/3;ctx.fillText(String(Number(gx.toPrecision(3))),x(gx)-7,h-10);ctx.fillText(String(Number(gy.toPrecision(3))),8,y(gy)+8);}
  ctx.save();ctx.beginPath();ctx.rect(45,20,w-70,h-55);ctx.clip();ctx.strokeStyle=getComputedStyle(dialog).getPropertyValue('--accent').trim();ctx.lineWidth=3;ctx.beginPath();
  if(weights[1]!==0){ctx.moveTo(x(xmin),y((-bias-weights[0]*xmin)/weights[1]));ctx.lineTo(x(xmax),y((-bias-weights[0]*xmax)/weights[1]));}
  else if(weights[0]!==0){const at=-bias/weights[0];ctx.moveTo(x(at),20);ctx.lineTo(x(at),h-35);}
  ctx.stroke();ctx.restore();
  for(const row of valid){ctx.fillStyle=row.label>0?(document.documentElement.dataset.theme==='paper'||document.documentElement.dataset.theme==='light'?'#8a492f':'#e0a077'):(document.documentElement.dataset.theme==='paper'||document.documentElement.dataset.theme==='light'?'#286a76':'#87c5ce');ctx.beginPath();ctx.arc(x(row.features[0]),y(row.features[1]),12,0,Math.PI*2);ctx.fill();ctx.fillStyle=ink;ctx.fillText(row.label>0?'+1':'−1',x(row.features[0])+19,y(row.features[1])+8);}
  ctx.fillStyle=muted;ctx.fillText('x₁',w-32,h-10);ctx.fillText('x₂',9,24);
}

function renderTraceControls():void{const area=dialog.querySelector('.study-trace');if(!area||!trace.length)return;area.innerHTML=`<div class="study-step-controls"><button data-study="step-back">← 上一步</button><span class="study-step-label">真实执行轨迹 · ${trace.length} 个状态</span><button data-study="step">看下一步 →</button></div><p class="study-step-description">逐步回看实际 Python 的状态记录；不会重复运行或撤销原计算。</p><button data-study="final-result">查看运行结束时的结果</button>`;}
function stepTrace(delta:number):void{if(!trace.length||!latestReport)return;showingTrace=true;traceIndex=Math.min(trace.length-1,Math.max(0,traceIndex+delta));const s=trace[traceIndex];drawCards(latestReport.training_rows,s.weights,s.bias);one('.study-step-label').textContent=`轨迹回放 ${traceIndex+1} / ${trace.length} · 第 ${s.epoch} 轮 · 当时源码第 ${s.line} 行`;const metrics=dialog.querySelectorAll<HTMLElement>('.study-result > .study-metrics > div');if(metrics[0]&&metrics[1]){metrics[0].querySelector('span')!.textContent='回放时刻的权重 w';metrics[0].querySelector('strong')!.textContent=s.weights.join('，');metrics[1].querySelector('span')!.textContent='回放时刻的偏置 b';metrics[1].querySelector('strong')!.textContent=String(s.bias);}one('.study-result-identity').textContent='回放已保存的轨迹 · '+(latestRun?.run_id?.slice(0,8)??'作者记录');one('.study-step-description').textContent=s.phase==='after_update'?`更新完成：卡片 (${s.features.join(', ')})，判断使用的分数 ${s.score_before}。旧参数 (${s.old_weights?.join(', ')}, ${s.old_bias}) → 新参数 (${s.weights.join(', ')}, ${s.bias})；变化量 (${s.weight_delta?.join(', ')}, ${s.bias_delta})。`:`判断前：卡片 (${s.features.join(', ')})，标签 ${s.label}，已算出的本张分数 ${s.score_before}；当前参数 w=(${s.weights.join(', ')}), b=${s.bias}。`;}
function renderBatch(v:any):void{const area=document.createElement('details');area.className='study-details';area.open=true;area.innerHTML='<summary>这次真实 batch 的输入、目标与可见范围</summary><p>以下展示第一条样本的前 16 个位置。位置 t 的输入只可读到 t，目标为下一 token；完整数组保存在这次产物中。</p><table class="study-table"><thead><tr><th>位置</th><th>input id</th><th>label id</th><th>可见输入</th><th>参与损失</th></tr></thead><tbody>'+v.input_ids[0].slice(0,16).map((id:number,i:number)=>`<tr><td>${i}</td><td>${enc(id)}</td><td>${enc(v.labels[0][i])}</td><td>0 … ${i}</td><td>${v.attention_mask[0][i]?'是':'否'}</td></tr>`).join('')+'</tbody></table>';one('.study-result').append(area);}
async function showReference():Promise<void>{
  if(!activity||!origin)return;const a=activity,bookId=origin.bookId,generation=pageGeneration;
  const asset=a.reference_asset?assetById(a.reference_asset):undefined;
  if(!asset){showError('这项活动没有登记独立的参考结果；原材料仍可从本章列表查看。');return;}
  const viewRevision=++resultViewRevision,wasAuthor=showingAuthorReference;showingAuthorReference=true;
  one('.study-run-status').textContent='正在读取作者已有记录；个人运行详情仍独立保存。';
  try{
    const raw=await bytesFor(asset);
    if(generation!==pageGeneration||viewRevision!==resultViewRevision||origin?.bookId!==bookId||activity?.id!==a.id)return;
    linearVisualization=true;trace=[];traceIndex=-1;latestReport=JSON.parse(new TextDecoder().decode(raw));
    showPersonalRunDetails(false);renderResult(latestReport,'作者已有记录 · 不是本次运行');
    one('.study-run-status').textContent='正在查看作者已有记录；点击“看我的记录”可回到原来的运行详情。';
  }catch(error){if(generation===pageGeneration&&viewRevision===resultViewRevision){showingAuthorReference=wasAuthor;showPersonalRunDetails(!wasAuthor);showError(error);}}
}
async function showArtifact(path:string):Promise<void>{
  if(!latestRun)return;const run=latestRun,generation=pageGeneration;
  const file=run.artifacts.find((a:any)=>a.path===path);if(!file)return;
  if(file.extension==='pt'){one('.study-run-status').textContent=`二进制模型产物：${formatBytes(file.bytes)} · SHA-256 ${file.sha256}。${pack?.activities.find(a=>a.id===run.activity_id)?.presentation==='causal-training@1'?'可在完整训练中选择从此运行的检查点恢复；':''}阅读界面不会反序列化或执行模型文件。`;return;}
  try{
    const bytes=buffer(await invokeBook<ArrayBuffer|number[]>('learning_artifact',{runId:run.run_id,artifactPath:path}));if(generation!==pageGeneration||!workspace.open)return;
    let panel=dialog.querySelector<HTMLElement>('.study-artifact-detail');if(!panel){panel=document.createElement('section');panel.className='study-artifact-detail';one('.study-content').append(panel);}
    panel.querySelectorAll<HTMLMediaElement>('audio,video').forEach(m=>m.pause());panel.replaceChildren();const title=document.createElement('h2');title.textContent=path.split('/').pop()??path;panel.append(title);
    if(file.extension==='json')renderJson(panel,JSON.parse(new TextDecoder().decode(bytes)));
    else if(['png','jpg'].includes(file.extension)){const img=document.createElement('img');const url=URL.createObjectURL(new Blob([bytes]));objectUrls.push(url);img.src=url;img.alt='本次运行的图像产物：'+path;img.className='study-original-image';panel.append(img);}
    else if(['wav','mp4'].includes(file.extension)){
      const media=document.createElement(file.extension==='wav'?'audio':'video');const key=`run:${run.run_id}:${path}`;
      const url=URL.createObjectURL(new Blob([bytes],{type:file.extension==='wav'?'audio/wav':'video/mp4'}));objectUrls.push(url);media.src=url;media.controls=true;media.preload='metadata';media.style.width='100%';media.setAttribute('aria-label','本次运行的媒体产物：'+path);
      const caption=document.createElement('p');caption.className='study-callout';caption.textContent='本次程序留下的媒体产物；手动播放。它属于这次运行，不能据此断言使用了哪一种生成方法。';panel.append(caption,media);
      media.addEventListener('play',()=>{if(activeMedia&&activeMedia!==media)activeMedia.pause();activeMedia=media;currentAsset=null;});
      media.addEventListener('pause',()=>{study.media[key]={time:media.currentTime,sha256:file.sha256,run_id:run.run_id};void saveState();});
      media.addEventListener('loadedmetadata',()=>{media.currentTime=Math.min(study.media[key]?.time??0,media.duration||0);});
      media.addEventListener('error',()=>resourceUnavailable(panel!,'当前媒体未能解码；产物与运行记录已保留。'));
    }else appendText(panel,new TextDecoder().decode(bytes));
    panel.scrollIntoView({block:'nearest'});
  }catch(e){if(generation===pageGeneration)showError(e);}
}

function showSources(url?:string):void{
  const body=resetContent('sources');const claims=pack!.source_claims.filter(s=>s.chapter===chapter!.id&&(!url||s.url===url));
  body.innerHTML=intro('原典在手边','看这句话的依据','引文、书中解释与原始资料分开呈现。这里保留当前版本的论断和引用范围，不补造页码。');
  for(const s of claims){const card=document.createElement('article');card.className='study-source-card';card.innerHTML=`<div class="study-section-title"><span class="study-eyebrow">${enc(s.chapter)} · ${enc(s.year??'日期见原引用')}</span><span>书中引用</span></div><h2>${enc(s.title||new URL(s.url).hostname)}</h2><p class="study-eyebrow">书中这段解释 · 非原文摘录</p><blockquote>${enc(s.claim)}</blockquote><p class="study-source-locator">引用范围：${enc(s.source_locator||s.citation)}</p><p class="study-source-locator">作者／机构（按书中标注）：${enc(s.attribution_from_book||'当前引文未单列，需核对原件')} · 年份／版本：${enc(s.publication_year_from_book||s.year||'见原始资料')}</p><div class="study-source-actions"><a href="${enc(s.url)}" target="_blank" rel="noopener noreferrer">在系统浏览器读原文 ↗</a><button class="study-copy-source">复制原始链接</button></div><details class="study-details"><summary>原始链接与核对边界</summary><p>${enc(s.url)}</p><p>${enc(s.metadata_status)}。书中这段解释的支持范围仍需对照原文判断。</p></details>`;if(s.url)card.querySelector('.study-copy-source')!.addEventListener('click',()=>void navigator.clipboard.writeText(s.url).then(()=>status('原始链接已复制')));else{card.querySelector('.study-source-actions')?.remove();card.querySelector('.study-details')?.remove();}body.append(card);}
  if(!claims.length)body.insertAdjacentHTML('beforeend','<p class="study-callout">此处没有登记外部引文。可查看本章的本地数据来源说明。</p>');
}
function showMap(history:boolean):void{
  const body=resetContent('map');body.innerHTML=intro(history?'历史发展 · 原文中的研究线索':'学习地图 · 真实出版顺序',history?'让日期保留它的含义':'知道自己站在哪里',history?'以下按引用中明确出现的年份排列。时间相邻不表示因果，不绘制未经证实的影响箭头。':'当前章节突出显示；依赖关系来自现行单元登记中的具体使用点。点击可回到相应正文。')+`<div class="study-tabs"><button data-study="map" class="${!history?'active':''}">学习依赖</button><button data-study="history" class="${history?'active':''}">历史线索</button></div>`;
  if(history){const seen=new Set();const claims=pack!.source_claims.filter(s=>s.chapter===chapter!.id&&s.year&&!seen.has(s.url)&&seen.add(s.url)).sort((a,b)=>a.year-b.year);body.insertAdjacentHTML('beforeend',`<div class="study-timeline">${claims.map(s=>`<article><time>${enc(s.year)}</time><div><h3>${enc(s.title)}</h3><p>${enc(s.claim)}</p><a href="${enc(s.url)}" target="_blank" rel="noopener noreferrer">读对应原文 ↗</a></div></article>`).join('')}</div>`);if(!claims.length)body.insertAdjacentHTML('beforeend','<p>本章引文中没有可直接读取的年份。日期没有被推测补齐。</p>');return;}
  const deps=pack!.dependencies.filter(d=>d.to===chapter!.id);
  body.insertAdjacentHTML('beforeend',`<div class="study-dependencies">${deps.map(d=>{const c=pack!.chapters.find(c=>c.id===d.from)!;return `<button data-study="jump" data-href="${enc(c.href+'#'+d.provider_anchor)}"><span>${enc(c.title)}</span><strong>${enc(d.concept)}</strong><small>${enc(d.action)} ↗</small></button>`;}).join('')}</div><div class="study-section-title"><h2>全书目录 · ${pack!.chapters.length} 节</h2><span>教学顺序，不是历史因果链</span></div><div class="study-chapter-map">${pack!.chapters.map(c=>`<button data-study="chapter" data-id="${c.id}" class="${c.id===chapter!.id?'current':''}"><span>${String(c.number).padStart(2,'0')}</span><strong>${enc(c.title.replace(/^第.+?章\s*/,''))}</strong></button>`).join('')}</div>`);
}

async function showAudio(groupId:string,initial?:string):Promise<void>{
  const group=pack!.audio_groups.find(g=>g.id===groupId);if(!group)return;resourceChapter(group.assets.flatMap((id:string)=>assetById(id)?.chapters??[]));const body=resetContent();const files:Asset[]=group.assets.map((id:string)=>assetById(id)!);currentAsset=files.find(a=>a.id===initial)??files[0];
  body.innerHTML=intro(`${chapter?.title??'当前材料'} · 听见差别`,group.title,group.description)+`<div class="study-tabs">${pack!.audio_groups.map(g=>`<button data-study="audio-group" data-id="${enc(g.id)}" class="${g.id===groupId?'active':''}">${enc(g.title)}</button>`).join('')}</div><div class="study-audio-layout"><div class="study-audio-choices">${files.map((a,i)=>`<button class="study-audio-choice ${a.id===currentAsset!.id?'selected':''}" data-audio-id="${a.id}"><span>${String.fromCharCode(65+i)}</span><div><strong>${enc(a.title)}</strong><small>${a.media?.sample_rate?Number(a.media.sample_rate).toLocaleString()+' Hz':'原采样率待核'} · ${a.media?.duration?Number(a.media.duration).toFixed(2)+' 秒':'读取时长中'} · ${a.role==='external_model'?'外部模型生成':'作者记录'}</small></div></button>`).join('')}</div><section class="study-audio-player"><span class="study-eyebrow">${files.length===1?'单段音频 · 手动播放':group.synchronize===false?'独立音频 · 分别定位':'同一时间片切换 · 每次只播放一个音源'}</span><h2 class="study-playing-title"></h2><audio class="study-audio" controls preload="metadata"></audio><div class="study-media-settings"><label>播放速度<select class="study-speed"><option value="0.5">0.5 ×</option><option value="0.75">0.75 ×</option><option value="1" selected>1 ×</option><option value="1.25">1.25 ×</option></select></label><label>循环起点<input class="study-loop-start" type="number" min="0" value="0" step="0.1"/></label><label>终点<input class="study-loop-end" type="number" min="0" value="2" step="0.1"/></label><label><input class="study-loop" type="checkbox"/> 循环这一段</label></div><p class="study-callout">原始音量 · 改变播放速度不会改写文件的采样率。退出时暂停，回来后不会自动播放。</p><label class="study-loudness"><input type="checkbox" class="study-rms-match"/> 按均方根幅度匹配音量（RMS）</label><p class="study-rms-status">默认原始音量；匹配不会改变文件。</p><button class="study-wave-button">按需查看波形与频谱</button><button data-study="note-media">记下这一刻</button><div class="study-wave-area"></div></section></div><div class="study-audio-details"></div>`;
  const audio=one<HTMLAudioElement>('audio');activeMedia=audio;
  const choices=one<HTMLElement>('.study-audio-choices'),layout=one<HTMLElement>('.study-audio-layout'),player=one<HTMLElement>('.study-audio-player');
  const stacked=window.matchMedia('(max-width:1180px)');
  const placeChoices=()=>{
    const focused=document.activeElement as HTMLElement|null,retainFocus=focused&&choices.contains(focused);
    if(stacked.matches){if(choices.parentElement!==player)audio.after(choices);}
    else if(choices.parentElement!==layout)layout.insertBefore(choices,player);
    if(retainFocus)focused.focus({preventScroll:true});
  };
  placeChoices();stacked.addEventListener('change',placeChoices);disposeResponsive=()=>stacked.removeEventListener('change',placeChoices);
  let audioLoadRevision=0;
  const load=async(a:Asset,keep:boolean)=>{const request=++audioLoadRevision;const playing=!audio.paused;const time=keep&&group.synchronize!==false?audio.currentTime:study.media[a.id]?.time??0;audio.pause();audio.removeAttribute('src');audio.load();audio.defaultPlaybackRate=Number(one<HTMLSelectElement>('.study-speed').value);audio.playbackRate=audio.defaultPlaybackRate;currentAsset=a;one('.study-playing-title').textContent=a.title;const gen=pageGeneration;let url:string;try{url=await urlFor(a,'audio/wav');}catch(error){if(gen===pageGeneration&&request===audioLoadRevision)resourceUnavailable(one('.study-audio-player'),error);return;}if(gen!==pageGeneration||request!==audioLoadRevision)return;one('.study-audio-player').querySelector('.study-resource-error')?.remove();audio.src=url;if(audioGain)audioGain.gain.value=1;one('.study-playing-title').textContent=a.title;one('.study-audio-details').innerHTML=details(a);audio.addEventListener('loadedmetadata',()=>{if(request!==audioLoadRevision||gen!==pageGeneration)return;audio.defaultPlaybackRate=Number(one<HTMLSelectElement>('.study-speed').value);audio.playbackRate=audio.defaultPlaybackRate;audio.currentTime=Math.min(time,audio.duration||0);if(dialog.querySelector<HTMLElement>('.study-save-state')?.dataset.statusOwner==='resource')status('音频已载入');if(playing&&keep)void audio.play().catch(showError);},{once:true});for(const b of Array.from(dialog.querySelectorAll<HTMLElement>('[data-audio-id]'))){b.classList.toggle('selected',b.dataset.audioId===a.id);b.setAttribute('aria-pressed',String(b.dataset.audioId===a.id));}if(one<HTMLInputElement>('.study-rms-match').checked)await setAmplitudeMatch(audio);};
  for(const b of Array.from(dialog.querySelectorAll<HTMLElement>('[data-audio-id]')))b.addEventListener('click',()=>void load(assetById(b.dataset.audioId!)!,true).catch(showError));
  one<HTMLSelectElement>('.study-speed').addEventListener('change',e=>{audio.defaultPlaybackRate=Number((e.target as HTMLSelectElement).value);audio.playbackRate=audio.defaultPlaybackRate;});
  audio.addEventListener('timeupdate',()=>{const ratio=audio.duration>0?audio.currentTime/audio.duration:0;for(const cursor of Array.from(dialog.querySelectorAll<HTMLElement>('.study-play-cursor')))cursor.style.left=`${Number(cursor.dataset.start||0)+ratio*Number(cursor.dataset.span||100)}%`;if(one<HTMLInputElement>('.study-loop').checked){const start=Number(one<HTMLInputElement>('.study-loop-start').value),end=Number(one<HTMLInputElement>('.study-loop-end').value);if(end>start&&audio.currentTime>=Math.min(audio.duration,end)){audio.currentTime=start;}}});
  one('.study-wave-button').addEventListener('click',()=>void showWaveform());one('.study-rms-match').addEventListener('change',()=>void setAmplitudeMatch(audio).catch(showError));await load(currentAsset,false);
}
async function analyseAudio(a:Asset):Promise<any>{
  if(audioAnalyses.has(a.id))return audioAnalyses.get(a.id);
  const bytes=await bytesFor(a);
  const decoder=new OfflineAudioContext(a.media?.channels||1,a.media?.frames||1,a.media?.sample_rate||48000);
  const decoded=await decoder.decodeAudioData(bytes);
  const worker=new Worker(new URL('./audio-analysis.ts',import.meta.url),{type:'module'});
  audioAnalysisWorker=worker;
  const result=await new Promise<any>((resolve,reject)=>{worker.onmessage=e=>resolve(e.data);worker.onerror=e=>reject(new Error(e.message));worker.postMessage({samples:decoded.getChannelData(0),sampleRate:decoded.sampleRate});});
  worker.terminate();if(audioAnalysisWorker===worker)audioAnalysisWorker=null;audioAnalyses.set(a.id,result);return result;
}
async function setAmplitudeMatch(audio:HTMLAudioElement):Promise<void>{
  const checkbox=one<HTMLInputElement>('.study-rms-match');const generation=pageGeneration;
  if(!checkbox.checked){if(audioGain)audioGain.gain.value=1;one('.study-rms-status').textContent='原始音量 · 没有对齐幅度';return;}
  if(!audioGraph){audioGraph=new AudioContext();audioGain=audioGraph.createGain();audioGraph.createMediaElementSource(audio).connect(audioGain);audioGain.connect(audioGraph.destination);}
  await audioGraph.resume();const a=currentAsset!;const analysed=await analyseAudio(a);if(currentAsset?.id!==a.id||!workspace.open||generation!==pageGeneration||!checkbox.checked)return;
  const gain=analysed.rms>1e-8?Math.min(8,.12/analysed.rms,.98/Math.max(analysed.peak,1e-8)):1;
  if(audioGain)audioGain.gain.value=gain;
  one('.study-rms-status').textContent=`第一声道 RMS ${analysed.rms.toFixed(4)} · 播放增益 ${gain.toFixed(2)} 倍。为避免削波保留峰值余量；这不是 LUFS 听感响度测量。`;
}
async function showWaveform():Promise<void>{
  if(!currentAsset)return;const generation=pageGeneration;const a=currentAsset;
  one('.study-wave-area').innerHTML='<p>正在后台分析当前片段…</p>';
  try{
    const v=await analyseAudio(a);if(generation!==pageGeneration||currentAsset?.id!==a.id)return;
    one('.study-wave-area').innerHTML='<h3>波形 · 第一声道</h3><div class="study-signal-plot"><canvas width="960" height="220" class="study-wave" role="img" aria-label="横轴为时间，纵轴为原始幅度"></canvas><span class="study-play-cursor" aria-hidden="true"></span></div><h3>短时频谱</h3><div class="study-signal-plot"><canvas width="960" height="340" class="study-spectrum" role="img" aria-label="横轴时间，纵轴频率，颜色表示谱幅度"></canvas><span class="study-play-cursor" data-start="6.25" data-span="92.7" aria-hidden="true"></span></div><p class="study-analysis-description"></p>';
    const wave=one<HTMLCanvasElement>('.study-wave'),spectrum=one<HTMLCanvasElement>('.study-spectrum');disposeObject?.();disposeObject=mountAudioPlots(dialog,wave,spectrum,v);
    one('.study-analysis-description').textContent=`分析率 ${v.sampleRate} Hz；${v.fftSize} 点 Hann 窗，频率由低到高，颜色对应 −80 至 0 dB 的相对谱幅度。RMS=${v.rms.toFixed(5)}，峰值=${v.peak.toFixed(5)}。横轴覆盖整段 ${v.duration.toFixed(3)} 秒，源文件未改变。`;
  }catch(e){showError(e);}
}
async function showVideo(selectedId?:string):Promise<void>{
  const body=resetContent();const generation=pageGeneration;
  const files=pack!.assets.filter(a=>a.kind==='video'&&(a.chapters.includes(chapter!.id)||a.id===selectedId));
  body.innerHTML=intro(`${chapter?.title??'当前材料'} · 视频`,'在当前问题旁观看原视频','保留原视频与播放位置；只有登记了实际解码帧，才开放精确的逐帧查看。各段素材的身份和时间关系以书中说明为准。')+'<div class="study-video-grid"></div>';
  for(const a of files){
    const frames:any[]=a.media?.decoded_frames??[];
    const panel=document.createElement('section');panel.className='study-video-card';
    panel.innerHTML=`<span class="study-eyebrow">${enc(a.media?.identity??'当前材料引用的原视频')}</span><h2>${enc(a.title)}</h2><div class="study-video-display"><video controls preload="metadata" aria-label="${enc(a.title)}"></video><img hidden alt="从原视频实际解码的所选帧"/></div><div class="study-video-controls"><button class="prev-frame" ${frames.length?'':'disabled'}>← 前一帧</button><span class="frame-label">${a.media?.playback_fps?'播放设置 '+enc(a.media.playback_fps)+' fps':'播放速率以原件为准'}</span><button class="next-frame" ${frames.length?'':'disabled'}>后一帧 →</button></div><input class="frame-range" type="range" ${frames.length?'':'disabled'} min="0" max="${Math.max(0,frames.length-1)}" value="0" aria-label="选择实际解码帧"/><button class="video-mode">回到视频播放</button><button class="video-enlarge">放大这段画面</button><button data-study="note-media" data-id="${a.id}">记下这一帧</button><p class="frame-context">${frames.length?'逐帧显示来自原文件的 PNG 解码帧；原始视频始终保留。播放速率不代表原拍摄帧率。':'未登记解码帧：当前使用播放器时间轴定位，不宣称帧精确。'}</p>${details(a)}`;
    one('.study-video-grid').append(panel);
    const video=panel.querySelector<HTMLVideoElement>('video')!;
    const image=panel.querySelector<HTMLImageElement>('.study-video-display img')!;
    try{video.src=await urlFor(a,'video/mp4');}catch(error){if(generation!==pageGeneration)return;panel.querySelectorAll<HTMLButtonElement|HTMLInputElement>('button,input').forEach(el=>el.disabled=true);resourceUnavailable(panel,error);continue;}if(generation!==pageGeneration)return;
    video.addEventListener('play',()=>{for(const v of Array.from(dialog.querySelectorAll('video')))if(v!==video)v.pause();activeMedia=video;currentAsset=a;});
    video.addEventListener('pause',()=>{study.media[a.id]={time:video.currentTime};void saveState();});
    let frame=0,frameGeneration=0;
    const show=async(n:number)=>{
      video.pause();currentAsset=a;frame=Math.max(0,Math.min(frames.length-1,n));const selected=frames[frame];if(!selected)return;
      const request=++frameGeneration;const decoded=assetById(selected.asset_id);if(!decoded)return;
      const url=await urlFor(decoded,'image/png');if(request!==frameGeneration||generation!==pageGeneration)return;
      image.src=url;image.hidden=false;video.hidden=true;
      panel.querySelector('.frame-label')!.textContent=`解码帧 ${frame+1} / ${frames.length} · ${selected.timestamp.toFixed(3)} 秒`;
      panel.querySelector<HTMLInputElement>('.frame-range')!.value=String(frame);
      if(selected.description)panel.querySelector('.frame-context')!.textContent=selected.description;
      study.media[a.id]={time:selected.timestamp,frame};void saveState();
    };
    panel.querySelector('.prev-frame')!.addEventListener('click',()=>void show(frame-1).catch(showError));
    panel.querySelector('.next-frame')!.addEventListener('click',()=>void show(frame+1).catch(showError));
    panel.querySelector<HTMLInputElement>('.frame-range')!.addEventListener('input',e=>void show(Number((e.target as HTMLInputElement).value)).catch(showError));
    panel.querySelector('.video-mode')!.addEventListener('click',()=>{image.hidden=true;video.hidden=false;video.currentTime=frames[frame]?.timestamp??0;video.focus();});
    panel.querySelector('.video-enlarge')!.addEventListener('click',()=>{panel.classList.toggle('enlarged');});
    video.addEventListener('loadedmetadata',()=>{video.currentTime=Math.min(study.media[a.id]?.time??0,video.duration||0);const savedFrame=study.media[a.id]?.frame;if(Number.isInteger(savedFrame))void show(savedFrame).catch(showError);});
  }
}

async function showRunHistory():Promise<void>{
  const body=resetContent('run-history'),generation=pageGeneration;
  body.innerHTML=intro('按保存版本回看','运行记录','每次结果都属于当时的代码、参数和输入。打开记录不会重新执行，也不会替换当前草稿。')+'<label class="study-search"><span>⌕</span><input class="study-run-search" aria-label="查找运行记录" placeholder="按活动、日期或运行编号查找"/></label><label class="study-code-guide"><input class="study-include-validation" type="checkbox"/> 包含功能验收记录</label><div class="study-run-list">正在读取记录…</div>';
  if(study.importedRuns?.length)body.insertAdjacentHTML('beforeend',`<details class="study-details"><summary>从其他设备导入的运行收据 · ${study.importedRuns.length}</summary>${study.importedRuns.map((r:any,i:number)=>`<button class="study-asset-row" data-study="imported-run" data-index="${i}">${enc(r.record.activity_id)} · ${enc(r.record.run_id)}</button>`).join('')}</details>`);
  try{
    const rows=await invokeBook<any[]>('learning_run_history');if(generation!==pageGeneration)return;
    const render=()=>{const q=one<HTMLInputElement>('.study-run-search').value.trim().toLowerCase(),include=one<HTMLInputElement>('.study-include-validation').checked;
      const filtered=rows.filter(r=>include||r.purpose!=='technical_validation').filter(r=>[r.run_id,pack!.activities.find(a=>a.id===r.activity_id)?.title??r.activity_id,new Date((r.started_at??0)*1000).toLocaleString()].join(' ').toLowerCase().includes(q));
      const list=one('.study-run-list');list.replaceChildren();if(!filtered.length){list.textContent='还没有符合条件的记录。';return;}
      let shown=0;const more=document.createElement('button');more.textContent='继续显示记录';
      const append=()=>{for(const r of filtered.slice(shown,shown+40)){const button=document.createElement('button');button.className='study-asset-row';button.dataset.study='saved-run';button.dataset.id=r.run_id;
        const title=pack!.activities.find(a=>a.id===r.activity_id)?.title??r.activity_id;const state=({succeeded:'已完成',failed:'未完成',cancelled:'已停止',interrupted:'已中断',running:'运行中，打开核对',preparing:'准备中，打开核对'} as Record<string,string>)[r.status]??r.status;
        button.innerHTML=`<span class="study-kind">${enc(state)}</span><div><strong>${enc(title)}</strong><small>${new Date((r.started_at??0)*1000).toLocaleString()} · ${enc(r.run_id)}${r.purpose==='technical_validation'?' · 功能验收':''}</small></div><span>↗</span>`;list.insertBefore(button,more.parentElement?more:null);}
        shown+=40;if(shown<filtered.length){if(!more.parentElement)list.append(more);}else more.remove();};more.addEventListener('click',append);append();};
    one('.study-run-search').addEventListener('input',render);one('.study-include-validation').addEventListener('change',render);render();
  }catch(e){if(generation===pageGeneration)resourceUnavailable(one('.study-run-list'),e);}
}
async function showSavedRun(id:string):Promise<void>{
  const body=resetContent(),generation=pageGeneration;latestRun=null;latestReport=null;trace=[];
  body.innerHTML=intro('历史结果 · 只读回看','正在读取这次实验','不会重新计算或替换你的草稿。')+'<p class="study-run-status" role="status">正在核对运行身份…</p>';
  try{
    const run=await invokeBook<any>('learning_run_status',{runId:id});if(generation!==pageGeneration)return;latestRun=run;
    const name=pack!.activities.find(a=>a.id===run.activity_id)?.title??run.activity_id;
    body.innerHTML=intro('历史结果 · '+(run.purpose==='technical_validation'?'功能验收':'按原版本保存'),name,'下方源码来自该次冻结快照；参数、输入和产物均按保存版本回看。这里不会重新执行。')+'<button data-study="run-history">← 所有运行记录</button><p class="study-run-status" role="status"></p><div class="study-lab-grid"><section class="study-result-panel"><h2>当时的结果</h2><div class="study-result"></div></section><section class="study-code-panel"><h2>当时的代码 · 只读</h2><textarea class="study-code" wrap="off" readonly aria-label="历史运行代码" spellcheck="false"></textarea></section></div><div class="study-artifacts"></div><details class="study-details study-log"><summary>完整 stdout / stderr</summary><pre></pre></details><details class="study-details study-history-metadata"><summary>参数、输入、环境和版本</summary></details>';
    one('.study-run-status').textContent=`${({succeeded:'已完成',failed:'未完成',cancelled:'已停止',interrupted:'已中断',running:'查询时仍在运行'} as Record<string,string>)[run.status]??run.status} · ${new Date((run.started_at??0)*1000).toLocaleString()} · ${run.run_id}`;
    one('.study-log pre').textContent=(run.stdout??'')+'\n'+(run.stderr??'');
    appendText(one('.study-history-metadata'),JSON.stringify({params:run.params,inputs:run.inputs,environment:run.environment,code_sha256:run.code_sha256,adapter_sha256:run.adapter_sha256,started_at:run.started_at,ended_at:run.ended_at},null,2));
    const files=run.artifacts??[];one('.study-artifacts').innerHTML=files.map((f:any)=>`<button class="study-artifact" data-study="artifact" data-path="${enc(f.path)}">${enc(f.path.split('/').pop())}<span>${formatBytes(f.bytes)} ↗</span></button>`).join('');
    const primary=files.find((f:any)=>/(?:_run|_train|_probe|one_step|cached_request|tool_request|\/report)\.json$/.test(f.path));
    if(primary){const raw=buffer(await invokeBook<ArrayBuffer|number[]>('learning_artifact',{runId:id,artifactPath:primary.path}));if(generation!==pageGeneration)return;renderJson(one('.study-result'),JSON.parse(new TextDecoder().decode(raw)));}
    else appendText(one('.study-result'),run.diagnostic||run.stdout||'没有完成结果，可查看已保存产物与原始错误。');
    try{const snapshot=await invokeBook<any>('learning_run_snapshot',{runId:id});if(generation===pageGeneration){one<HTMLTextAreaElement>('.study-code').value=snapshot.code;if(snapshot.adapter_code){const adapter=document.createElement('details');adapter.className='study-details';adapter.innerHTML='<summary>当时的随书运行适配 · 只读快照</summary>';appendText(adapter,snapshot.adapter_code);body.append(adapter);}}}
    catch(e){if(generation===pageGeneration)one<HTMLTextAreaElement>('.study-code').value='无法核对当时的代码快照：'+String(e);}
  }catch(e){if(generation===pageGeneration)resourceUnavailable(body,e);}
}
function unfinishedLocation(draft:any):string{
  return pack?.chapters.find(c=>c.id===draft.chapter)?.title??String(draft.chapter||draft.href||'原章节未标明');
}
function showNotes():void{
  const body=resetContent('notes');body.innerHTML=intro('本书笔记 · 跨章保留','你的思考，和它原来的位置','这里列出本书各章的笔记。新记录会标明当前段落或章节，并能回到对应的实验和原句。')+'<button class="study-primary" data-study="add-note">写一条思考</button><button data-study="export-note">导出本书笔记</button><div class="study-notes-list"></div>';
  if(study.unfinished_note){
    const draft=study.unfinished_note,editor=document.createElement('section');editor.className='study-note-editor';
    if(draft.editId)editor.dataset.editId=draft.editId;
    editor.dataset.restoredAnchor=JSON.stringify(draft);
    editor.innerHTML=`<p class="study-eyebrow">当前未提交草稿 · ${enc(unfinishedLocation(draft))}</p>${draft.quote?`<blockquote>${enc(String(draft.quote).slice(0,240))}</blockquote>`:''}<textarea aria-label="继续当前草稿"></textarea><button class="study-primary" data-study="save-note">保存这条思考</button>`;
    editor.querySelector('textarea')!.value=draft.text;one('.study-notes-list').append(editor);
  }
  if(study.unfinished_notes?.length){
    const heading=document.createElement('h2');heading.textContent=`待继续的草稿 · ${study.unfinished_notes.length} 份`;
    one('.study-notes-list').append(heading);
    for(const draft of study.unfinished_notes){
      const card=document.createElement('article');card.className='study-note-card';
      card.innerHTML=`<span class="study-eyebrow">${enc(unfinishedLocation(draft))} · 待继续的草稿</span>${draft.quote?`<blockquote>${enc(String(draft.quote).slice(0,240))}</blockquote>`:''}<p>${enc(String(draft.text).slice(0,240)).replace(/\n/g,'<br/>')}</p>${draft.content_digest&&draft.content_digest!==origin?.contentDigest?'<p class="study-code-guide">原内容版本不同，保存前请核对原位置。</p>':''}`;
      const button=document.createElement('button');button.dataset.study='continue-unfinished';button.dataset.importId=draft.import_id;button.textContent='继续这份草稿';card.append(button);one('.study-notes-list').append(card);
    }
  }
  for(const note of [...study.notes].reverse()){const el=document.createElement('article');el.className='study-note-card';el.innerHTML=`<span class="study-eyebrow">${enc(note.chapter)} · ${new Date(note.created_at).toLocaleString()}</span><blockquote>${enc(note.quote)}</blockquote><p>${enc(note.text).replace(/\n/g,'<br/>')}</p>${note.media_anchor?`<button data-study="return-media" data-id="${enc(note.id)}">回到媒体 ${Number(note.media_anchor.time||0).toFixed(2)} 秒${note.media_anchor.frame!==undefined?` · 解码帧 ${note.media_anchor.frame+1}`:''} ↗</button>`:''}${note.run_id?`<button data-study="saved-run" data-id="${enc(note.run_id)}">查看当时的实验 ↗</button>`:''}${note.content_digest&&note.content_digest!==origin?.contentDigest?'<span class="study-code-guide">旧内容版本的笔记 · 保留待核对</span>':`<button data-study="jump" data-href="${enc(note.cfi||note.href)}">回到这条思考的原句 ↗</button>`}<button data-study="edit-note" data-id="${enc(note.id)}">修改</button><button data-study="archive-note" data-id="${enc(note.id)}">移到回收站</button>`;one('.study-notes-list').append(el);}
  if(study.archived_notes?.length)one('.study-notes-list').insertAdjacentHTML('beforeend',`<details class="study-details"><summary>回收站 · ${study.archived_notes.length} 条，可恢复</summary>${study.archived_notes.map((n:any)=>`<p>${enc(n.text.slice(0,160))}<button data-study="restore-note" data-id="${enc(n.id)}">恢复</button></p>`).join('')}</details>`);
}
async function addNote():Promise<void>{const existing=dialog.querySelector<HTMLTextAreaElement>('.study-note-editor textarea');if(existing){existing.focus();return;}const el=document.createElement('section');el.className='study-note-editor';if(pendingMediaAnchor)el.dataset.restoredAnchor=JSON.stringify({chapter:chapter?.id,quote:origin?.quote,href:origin?.href,cfi:origin?.cfi,content_digest:origin?.contentDigest,media_anchor:pendingMediaAnchor});el.innerHTML=`<blockquote>${enc(origin?.quote||chapter?.question)}</blockquote><textarea aria-label="我的思考" placeholder="我改了什么？结果支持什么？还有什么没有弄清？"></textarea><button class="study-primary" data-study="save-note">保存这条思考</button>`;one('.study-notes-list').prepend(el);el.querySelector('textarea')!.focus();}
async function continueUnfinishedNote(importId:string):Promise<void>{
  if(!studyLoaded||noteTransition)return;
  captureUnfinishedNote();
  const generation=pageGeneration,bookId=origin?.bookId,visible=dialog.querySelector<HTMLTextAreaElement>('.study-note-editor textarea');
  const previousCurrent=structuredClone(study.unfinished_note??null);
  const previousQueue=structuredClone(study.unfinished_notes??[]);
  const previousMedia=pendingMediaAnchor;
  const previousCommitted=visible?.dataset.committed;
  let marked=false;
  noteTransition=true;
  try{
    const index=previousQueue.findIndex((note:any)=>note.import_id===importId);
    if(index<0)throw new Error('这份草稿已不在待继续队列中');
    const selected=structuredClone(previousQueue[index]);
    if(await unfinishedNoteIdentity(selected)!==importId)throw new Error('草稿身份与内容不符；原有草稿未改动');
    if(generation!==pageGeneration||origin?.bookId!==bookId||!workspace.open)return;
    const next=previousQueue.filter((_:any,i:number)=>i!==index);
    if(previousCurrent?.text?.trim()){
      const currentId=await unfinishedNoteIdentity(previousCurrent);
      if(currentId!==importId&&!next.some((note:any)=>note.import_id===currentId))next.push({...previousCurrent,import_id:currentId});
    }
    if(generation!==pageGeneration||origin?.bookId!==bookId||!workspace.open)return;
    delete selected.import_id;
    if(visible){visible.dataset.committed='true';marked=true;}
    study.unfinished_note=selected;study.unfinished_notes=next;
    await persistStudy();
    pendingMediaAnchor=selected.media_anchor??null;
    if(generation===pageGeneration&&origin?.bookId===bookId&&workspace.open){showNotes();one<HTMLTextAreaElement>('.study-note-editor textarea').focus();status('已切换草稿；其他未提交内容仍保留');}
  }catch(error){if(origin?.bookId===bookId){study.unfinished_note=previousCurrent;study.unfinished_notes=previousQueue;pendingMediaAnchor=previousMedia;if(marked&&visible?.isConnected){if(previousCommitted===undefined)delete visible.dataset.committed;else visible.dataset.committed=previousCommitted;}showError(error);}}
  finally{noteTransition=false;}
}
async function persistNote():Promise<void>{
  const input=dialog.querySelector<HTMLTextAreaElement>('.study-note-editor textarea');if(!studyLoaded||!origin||noteTransition||!input?.value.trim())return;
  const editor=input.closest<HTMLElement>('.study-note-editor');
  const editId=editor?.dataset.editId;
  const savedAnchor=editor?.dataset.restoredAnchor;
  let restored:any=null;
  try{restored=savedAnchor?JSON.parse(savedAnchor):null;}catch{showError('草稿的原位置记录无法读取；内容仍在编辑框中，未保存到错误位置');return;}
  noteTransition=true;
  const generation=pageGeneration,bookId=origin.bookId,oldCommitted=input.dataset.committed;
  const original=(key:string,fallback:unknown)=>restored&&Object.prototype.hasOwnProperty.call(restored,key)?restored[key]:fallback;
  const oldNotes=structuredClone(study.notes),oldDraft=structuredClone(study.unfinished_note??null),oldMedia=pendingMediaAnchor;
  const existing=editId?study.notes.find((n:any)=>n.id===editId):null;
  input.dataset.committed='true';
  if(existing){existing.previous_versions??=[];existing.previous_versions.push({text:existing.text,at:Date.now()});existing.text=input.value;}
  else study.notes.push({id:crypto.randomUUID(),book_uuid:original('book_uuid',pack!.book_uuid),book_revision:original('book_revision',pack!.book_revision_sha256),content_digest:original('content_digest',origin!.contentDigest),chapter:original('chapter',chapter!.id),cfi:original('cfi',origin!.href.endsWith(chapter!.href)?origin!.cfi:null),href:original('href',chapter!.href),quote:original('quote',origin!.href.endsWith(chapter!.href)?origin!.quote||chapter!.question:chapter!.question),text:input.value,media_anchor:original('media_anchor',pendingMediaAnchor),created_at:Date.now(),run_id:original('run_id',pack!.activities.find(a=>a.id===latestRun?.activity_id)?.chapter===chapter!.id?latestRun?.run_id??null:null)});
  study.unfinished_note=null;
  try{await persistStudy();if(origin?.bookId===bookId){pendingMediaAnchor=null;if(generation===pageGeneration&&workspace.open){status('已保存到本机');showNotes();}}}
  catch(error){if(origin?.bookId===bookId){study.notes=oldNotes;study.unfinished_note=oldDraft;pendingMediaAnchor=oldMedia;if(input.isConnected){if(oldCommitted===undefined)delete input.dataset.committed;else input.dataset.committed=oldCommitted;}showError(error);}}
  finally{noteTransition=false;}
}
async function exportNotes():Promise<void>{
  await saveState();try{const path=await invokeBook<string|null>('learning_export',{activityId:null});status(path?'已导出：'+path:'已取消导出');}catch(e){showError(e);}
}
async function showDraftHistory():Promise<void>{
  if(!activity||!origin)return;
  const bookId=origin.bookId,activityId=activity.id,generation=pageGeneration;
  const editor=one<HTMLTextAreaElement>('.study-code');
  await saveDraft();if(!sameActivity(bookId,activityId,generation,editor))return;
  const versions=await invokeBook<any[]>('learning_draft_versions',{activityId});
  if(!sameActivity(bookId,activityId,generation,editor))return;
  let box=dialog.querySelector<HTMLElement>('.study-version-list');if(box)box.remove();box=document.createElement('div');box.className='study-version-list';
  box.innerHTML='<h3>你的独立草稿版本</h3>'+(!versions.length?'<p>还没有保存过自己的版本。点“我来改”即可开始。</p>':'');
  for(const version of versions){
    const b=document.createElement('button');
    b.textContent=`${new Date(version.updated_at*1000).toLocaleString()} · ${formatBytes(version.bytes)} · ${version.sha256.slice(0,10)} · 恢复`;
    b.addEventListener('click',()=>void (async()=>{
      if(!sameActivity(bookId,activityId,generation,editor))return;
      b.disabled=true;
      try{
        const restored=await invokeBook<any>('learning_restore_draft',{activityId,revision:version.sha256});
        if(!sameActivity(bookId,activityId,generation,editor))return;
        draftMode=true;editor.readOnly=false;editor.value=restored.code;
        one('.study-code-mode').textContent='我的独立版本';one<HTMLButtonElement>('[data-study="save"]').hidden=false;box?.remove();
        const message=dialog.querySelector<HTMLElement>('.study-run-status');
        if(message?.dataset.errorScope==='draft-history'){message.classList.remove('study-error');delete message.dataset.statusOwner;delete message.dataset.errorScope;message.textContent='已恢复核对过的草稿版本；点击运行才会计算。';}
        markStale();status('已恢复这个版本；其他版本仍在');
      }catch(error){if(sameActivity(bookId,activityId,generation,editor))showError(error,'draft-history');}
      finally{if(b.isConnected)b.disabled=false;}
    })());
    box.append(b);
  }
  one('.study-code-panel').append(box);
}
async function editNote(id:string):Promise<void>{
  if(!studyLoaded||!origin||noteTransition)return;
  const note=study.notes.find((n:any)=>n.id===id);if(!note)return;
  const visible=dialog.querySelector<HTMLElement>('.study-note-editor');
  if(visible?.dataset.editId===id){visible.querySelector<HTMLTextAreaElement>('textarea')?.focus();return;}
  const generation=pageGeneration,bookId=origin.bookId,oldCommitted=visible?.querySelector<HTMLTextAreaElement>('textarea')?.dataset.committed;
  captureUnfinishedNote();
  const previousCurrent=structuredClone(study.unfinished_note??null),previousQueue=structuredClone(study.unfinished_notes??[]);
  let marked=false;
  noteTransition=true;
  try{
    if(previousCurrent?.text?.trim()){
      const importId=await unfinishedNoteIdentity(previousCurrent);
      if(generation!==pageGeneration||origin?.bookId!==bookId||!workspace.open)return;
      study.unfinished_notes??=[];
      if(!study.unfinished_notes.some((draft:any)=>draft.import_id===importId))study.unfinished_notes.push({...previousCurrent,import_id:importId});
      const input=visible?.querySelector<HTMLTextAreaElement>('textarea');if(input){input.dataset.committed='true';marked=true;}
      study.unfinished_note=null;await persistStudy();pendingMediaAnchor=null;
    }
    if(generation!==pageGeneration||origin?.bookId!==bookId||!workspace.open)return;
    visible?.remove();showNotes();
    const editor=document.createElement('section');editor.className='study-note-editor';editor.dataset.editId=id;editor.dataset.restoredAnchor=JSON.stringify({chapter:note.chapter,cfi:note.cfi,href:note.href,quote:note.quote,content_digest:note.content_digest,editId:id});editor.innerHTML='<textarea aria-label="修改这条思考"></textarea><button class="study-primary" data-study="save-note">保存修改</button>';editor.querySelector('textarea')!.value=note.text;one('.study-notes-list').prepend(editor);editor.querySelector('textarea')!.focus();
  }catch(error){
    if(origin?.bookId===bookId){study.unfinished_note=previousCurrent;study.unfinished_notes=previousQueue;const input=visible?.querySelector<HTMLTextAreaElement>('textarea');if(marked&&input?.isConnected){if(oldCommitted===undefined)delete input.dataset.committed;else input.dataset.committed=oldCommitted;}showError(error);}
  }
  finally{noteTransition=false;}
}
async function archiveNote(id:string):Promise<void>{const note=study.notes.find((n:any)=>n.id===id);if(!note)return;study.archived_notes??=[];study.archived_notes.push({...note,archived_at:Date.now()});study.notes=study.notes.filter((n:any)=>n.id!==id);await saveState();showNotes();}
async function restoreNote(id:string):Promise<void>{const note=study.archived_notes?.find((n:any)=>n.id===id);if(!note)return;study.notes.push(note);study.archived_notes=study.archived_notes.filter((n:any)=>n.id!==id);await saveState();showNotes();}

async function saveState():Promise<void>{if(!origin||!studyLoaded)return;try{await persistStudy();status('已保存到本机');}catch(e){showError(e);}}
function saveViewSoon():void{if(saveTimer!==null)clearTimeout(saveTimer);saveTimer=window.setTimeout(()=>{saveTimer=null;void saveState();},350);}
function status(message:string,owner='general'):void{const el=dialog?.querySelector<HTMLElement>('.study-save-state');if(el){el.textContent=message;el.dataset.statusOwner=owner;el.classList.remove('study-error');}}
function showError(error:unknown,scope='general'):void{
  let text=readableError(error);
  if(/Python 环境不可用|解释器版本已改变|缺少必要字段 python/.test(text))text+='。请恢复本章已登记的解释器，或核对后重新绑定环境，再重试；正文、草稿和原有结果保留。';
  const footer=dialog?.querySelector<HTMLElement>('.study-save-state');
  const el=dialog?.querySelector<HTMLElement>('.study-run-status')??footer;
  if(el){el.textContent=text;el.dataset.statusOwner='error';el.dataset.errorScope=scope;el.classList.add('study-error');}else host.toast(text);
  if(scope==='draft-history'&&footer&&footer!==el){footer.textContent=text;footer.dataset.statusOwner='error';footer.dataset.errorScope=scope;footer.classList.add('study-error');}
}
async function closeStudy():Promise<void>{
  if(!workspace?.open)return;studyOpenGeneration++;pendingMediaAnchor=null;cancelConsent?.();try{await flushLearningBeforeClose();}catch(error){showError('尚未保存成功，请保留当前窗口重试：'+String(error));return;}if(activeMedia){study.media[currentAsset?.id??'last']={time:activeMedia.currentTime};activeMedia.pause();activeMedia=null;}
  for(const media of Array.from(dialog.querySelectorAll<HTMLMediaElement>('audio,video')))media.pause();
  if(audioGraph){void audioGraph.close();audioGraph=null;audioGain=null;}audioAnalysisWorker?.terminate();audioAnalysisWorker=null;audioAnalyses.clear();if(studyLoaded)await saveState();pageGeneration++;disposeObject?.();disposeObject=null;disposeResponsive?.();disposeResponsive=null;for(const url of objectUrls)URL.revokeObjectURL(url);objectUrls=[];workspace.close();
  // The reader canvas was never resized or navigated by the stage.
  if(origin?.selection){try{const selection=origin.selection.startContainer.ownerDocument?.getSelection();selection?.removeAllRanges();selection?.addRange(origin.selection);}catch{}}
  if(origin?.focus&&!origin.focus.isConnected&&origin.cfi){try{await host.jump(origin.cfi,origin.pane);}catch(e){host.toast(`原句暂未恢复：${String(e)}`);}}
  if(origin?.focus?.isConnected)origin.focus.focus({preventScroll:true});else document.querySelector<HTMLButtonElement>('.study-toggle')?.focus();
  studyLoaded=false;
}

export async function closeLearning():Promise<void>{await closeStudy();}
