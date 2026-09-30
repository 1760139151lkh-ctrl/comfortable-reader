import {portableBooks,portableAsset,portableLearningPack} from './portable-books';
import {parsePairs} from '../../site/engine.mjs';
import engineSource from '../../site/engine.mjs?raw';
import {sha256,materialize} from '../../site/core.mjs';
import {publicStore} from './portable-books';

const prefix='comfortable-reader-personal-v1:';
function read(key:string,fallback:any):any{const value=localStorage.getItem(prefix+key);if(value===null)return structuredClone(fallback);try{return JSON.parse(value);}catch{throw new Error('本机保存的数据未能解析。原数据仍在，请先导出或检查，不会自动清空。');}}
function save(key:string,value:any):void{const json=JSON.stringify(value);if(json.length>4*1024*1024)throw new Error('这份记录超过单次保存范围');try{localStorage.setItem(prefix+key,json);}catch{throw new Error('浏览器保存空间不足；请保留当前页面并导出记录。');}}
const emptySession={paneCount:1,paneBookIds:[null,null,null,null],activePane:0,theme:'paper',fontScale:100,readerFont:'serif',contentWidth:44,lineHeight:null};
function saveEdition(bookId:string,progress:any):void{const key=progress?.contentDigest??progress?.sourceSha256;if(typeof key==='string'&&/^[a-f0-9]{64}$/.test(key))save(`edition:${bookId}:${key}`,progress);}
const runWorkers=new Map<string,Worker>();
const liveRuns=new Map<string,any>();
export const builtinSource=engineSource;
export function activeBuiltin(id:string):any{return liveRuns.get(id);}
let nativeCall:((command:string,args:any)=>Promise<any>)|null=null;
export function useNativePersonalStore(call:(command:string,args:any)=>Promise<any>):void{nativeCall=call;}
async function getRun(bookId:string,id:string):Promise<any>{return liveRuns.get(id)??(nativeCall?await nativeCall('learning_run_status',{bookId,runId:id}):read(`run:${bookId}:${id}`,null));}
async function putRun(bookId:string,run:any,result?:any):Promise<void>{if(nativeCall)Object.assign(run,await nativeCall('learning_store_builtin_run',{bookId,record:run,result:result??null,code:engineSource}));else{save(`run:${bookId}:${run.run_id}`,run);save(`snapshot:${bookId}:${run.run_id}`,{code:engineSource,code_sha256:run.code_sha256,run_id:run.run_id});if(result!==undefined)save(`result:${bookId}:${run.run_id}`,result);}}
export async function runBuiltin(args:any):Promise<any>{
  const session=portableBooks.get(args.bookId);if(!session)throw new Error('书籍尚未准备');
  const activity=session.book.activities.find((a:any)=>a.id===args.activityId);if(activity?.capability!=='least-squares@1')throw new Error('此活动需要已绑定的桌面运行环境。浏览器不会执行书中代码。');
  if(args.useDraft)throw new Error('这份 Python 草稿没有在浏览器执行。可导出到本机，或切回内置计算。');
  const existing=await getRun(args.bookId,args.requestId).catch(()=>null);if(existing)return existing;
  const input=session.book.resources.find((r:any)=>r.id===activity.required[0]);
  let data:ArrayBuffer;try{data=await materialize(publicStore,input);}catch{throw new Error('本次输入尚未准备，请先取得这个活动需要的材料。');}
  const rows=parsePairs(new TextDecoder('utf-8',{fatal:true}).decode(data));
  const codeHash=await sha256(new TextEncoder().encode(engineSource).buffer);
  const record:any={run_id:args.requestId,book_id:args.bookId,activity_id:args.activityId,chapter:activity.chapter,status:'running',params:args.params,code_sha256:codeHash,identity:'browser_builtin',security_mode:'trusted_builtin_worker',runtime:'least-squares@1',started_at:Math.floor(Date.now()/1000),inputs:[{path:input.logicalPath,sha256:input.sha256}],artifacts:[],stdout:'',stderr:''};
  await putRun(args.bookId,record);
  const worker=new Worker(new URL('../../site/worker.mjs',import.meta.url),{type:'module'});runWorkers.set(record.run_id,worker);liveRuns.set(record.run_id,record);
  const finish=async(error:string|null,result?:any)=>{if(!runWorkers.has(record.run_id))return;runWorkers.delete(record.run_id);worker.terminate();record.status=error?'failed':'succeeded';record.ended_at=Math.floor(Date.now()/1000);record.stderr=error??'';if(result){const bytes=new TextEncoder().encode(JSON.stringify(result));record.artifacts=[{path:'work/results/report.json',extension:'json',bytes:bytes.length,sha256:await sha256(bytes.buffer),role:'builtin_result'}];}await putRun(args.bookId,record,result);};
  const timer=window.setTimeout(()=>void finish('本次内置计算超过 4 秒，已结束；旧结果保留。'),4000);
  worker.addEventListener('message',event=>{if(event.data.requestId!==record.run_id)return;clearTimeout(timer);void finish(event.data.ok?null:String(event.data.error),event.data.result);});
  worker.addEventListener('error',event=>{void finish(event.message);});worker.postMessage({requestId:record.run_id,rows,parameters:args.params});return record;
}
function download(name:string,body:string,mime='application/json'):void{const url=URL.createObjectURL(new Blob([body],{type:mime}));const link=document.createElement('a');link.href=url;link.download=name;link.click();window.setTimeout(()=>URL.revokeObjectURL(url),1000);}
export async function webInvoke(command:string,args:any={}):Promise<any>{
  const bookId=args.bookId;
  switch(command){
    case 'bootstrap':case 'refresh_library':return{books:read('books',[]),progress:read('progress',{}),session:read('session',emptySession),libraryRoots:[],storagePath:'当前浏览器 · 不自动同步',scan:{rootsScanned:0,missingRoots:0,filesSeen:0,epubCandidates:0,loadedCandidates:0,booksLoaded:read('books',[]).length,added:0,updated:0,duplicates:0,unreadable:0,otherBookFiles:0,ignoredTrees:0,issues:[]}};
    case 'register_catalog_book':{const books=read('books',[]);const old=books.findIndex((b:any)=>b.id===args.record.id);if(old<0)books.push(args.record);else books[old]=args.record;save('books',books);return args.record;}
    case 'save_session':save('session',args.session);return;
    case 'save_progress':{const progress=read('progress',{});saveEdition(bookId,progress[bookId]);saveEdition(bookId,args.progress);progress[bookId]=args.progress;save('progress',progress);return;}
    case 'activate_progress_edition':{const all=read('progress',{}),previous=all[bookId];saveEdition(bookId,previous);const source=args.sourceSha256,key=args.contentDigest??source;let chosen=read(`edition:${bookId}:${key}`,null)??read(`edition:${bookId}:${source}`,null);chosen??={cfi:null,page:0,totalPages:0,percent:0,annotations:[]};chosen={...chosen,sourceSha256:source,contentDigest:args.contentDigest,bookUuid:portableBooks.get(bookId)?.book.id,pageMode:previous?.pageMode??0,readingMode:previous?.readingMode??'scroll',updatedAt:Math.floor(Date.now()/1000)};all[bookId]=chosen;saveEdition(bookId,chosen);save('progress',all);return chosen;}
    case 'progress_editions':{const key=prefix+`edition:${bookId}:`;return Object.keys(localStorage).filter(name=>name.startsWith(key)).map(name=>({key:name.slice(key.length),progress:JSON.parse(localStorage.getItem(name)!)})).sort((a,b)=>b.progress.updatedAt-a.progress.updatedAt);}
    case 'learning_pack':return portableLearningPack(bookId);
    case 'learning_asset':return await portableAsset(bookId,args.assetId);
    case 'learning_load_state':return read(`study:${bookId}`,{version:1,notes:[],media:{},drafts:{}});
    case 'learning_raw_state':{const value=localStorage.getItem(prefix+`study:${bookId}`);if(value===null)throw new Error('没有可备份的原始记录');const data=new TextEncoder().encode(value);return{name:'原始学习记录-'+bookId+'.json',sha256:await sha256(data.buffer),bytes:Array.from(data)};}
    case 'learning_save_state':save(`study:${bookId}`,args.value);return;
    case 'learning_save_draft':{const key=`draft:${bookId}:${args.activityId}`;const value={code:args.code,sha256:await sha256(new TextEncoder().encode(args.code).buffer),updated_at:Math.floor(Date.now()/1000)};save(key,value);save(key+':'+value.sha256,value);return value;}
    case 'learning_draft':return read(`draft:${bookId}:${args.activityId}`,null);
    case 'learning_import_draft':{if(typeof args.code!=='string'||args.code.length>300000||!/^[a-z][a-z0-9-]{0,63}$/.test(args.activityId))throw new Error('导入草稿无效');const key=`draft:${bookId}:${args.activityId}`,value={code:args.code,sha256:await sha256(new TextEncoder().encode(args.code).buffer),updated_at:Math.floor(Date.now()/1000),provenance:'explicit_personal_import'};save(key+':'+value.sha256,value);if(!read(key,null))save(key,value);return;}
    case 'learning_draft_activities':{const key=prefix+`draft:${bookId}:`;const ids=new Set<string>();for(const name of Object.keys(localStorage)){if(!name.startsWith(key))continue;const id=name.slice(key.length).split(':')[0];if(!/^[A-Za-z0-9_-]{1,160}$/.test(id))throw new Error('代码活动标识无效；原记录未改动');ids.add(id);if(ids.size>2000)throw new Error('代码活动数量超过单次导出范围');}return [...ids].sort();}
    case 'learning_draft_versions':{const key=prefix+`draft:${bookId}:${args.activityId}:`;return Object.keys(localStorage).filter(k=>k.startsWith(key)).map(k=>JSON.parse(localStorage.getItem(k)!)).map(v=>({sha256:v.sha256,updated_at:v.updated_at,bytes:new TextEncoder().encode(v.code).length}));}
    case 'learning_draft_revision':{
      if(typeof args.activityId!=='string'||!/^[a-z][a-z0-9-]{0,63}$/.test(args.activityId)||typeof args.revision!=='string'||!/^[a-f0-9]{64}$/.test(args.revision))throw new Error('无效的代码版本');
      const value=read(`draft:${bookId}:${args.activityId}:${args.revision}`,null);
      if(!value)throw new Error('该草稿版本不存在');
      const bytes=typeof value.code==='string'?new TextEncoder().encode(value.code):null;
      if(!bytes||bytes.byteLength>300000||value.sha256!==args.revision||await sha256(bytes.buffer)!==args.revision||!Number.isSafeInteger(value.updated_at)||value.updated_at<0)throw new Error('草稿版本内容或身份校验失败；原数据未改动');
      return {code:value.code,sha256:value.sha256,updated_at:value.updated_at};
    }
    case 'learning_restore_draft':{const value=await webInvoke('learning_draft_revision',args);return await webInvoke('learning_save_draft',{bookId,activityId:args.activityId,code:value.code});}
    case 'learning_run':return await runBuiltin(args);
    case 'learning_run_status':{const run=read(`run:${bookId}:${args.runId}`,null);if(!run)throw new Error('这条运行记录不存在');if(run.status==='running'&&!runWorkers.has(args.runId)){run.status='interrupted';run.diagnostic='浏览器已重开，上次计算已中断；没有自动重跑';save(`run:${bookId}:${args.runId}`,run);}return run;}
    case 'learning_run_history':{const key=prefix+`run:${bookId}:`;return Object.keys(localStorage).filter(k=>k.startsWith(key)).map(k=>JSON.parse(localStorage.getItem(k)!)).sort((a,b)=>b.started_at-a.started_at);}
    case 'learning_run_snapshot':{const run=read(`run:${bookId}:${args.runId}`,null),snapshot=read(`snapshot:${bookId}:${args.runId}`,null);if(!run||!snapshot||snapshot.code_sha256!==run.code_sha256||run.code_sha256!==await sha256(new TextEncoder().encode(snapshot.code).buffer))throw new Error('历史代码快照尚未核对，当前源码不会冒充历史源码');return snapshot;}
    case 'learning_artifact':{if(args.artifactPath!=='work/results/report.json')throw new Error('产物未登记');const value=read(`result:${bookId}:${args.runId}`,null);if(value===null)throw new Error('结果不存在');return new TextEncoder().encode(JSON.stringify(value)).buffer;}
    case 'learning_cancel':{const worker=runWorkers.get(args.runId);if(worker){worker.terminate();runWorkers.delete(args.runId);const run=await getRun(bookId,args.runId);run.status='cancelled';run.ended_at=Math.floor(Date.now()/1000);await putRun(bookId,run);}return;}
    case 'learning_open_source':{const pack=await portableLearningPack(bookId),source=pack.source_claims.find((s:any)=>s.id===args.sourceId);if(!source?.url||!/^https:\/\//.test(source.url))throw new Error('来源地址不可用');window.open(source.url,'_blank','noopener,noreferrer');return;}
    case 'learning_resource_info':{const book=portableBooks.get(bookId),asset=book?.book.resources.find((r:any)=>r.id===args.assetId);if(!asset)throw new Error('资源未登记');return{path:asset.logicalPath,sha256:asset.sha256,bytes:asset.bytes};}
    case 'learning_export':{if(args.activityId){const draft=read(`draft:${bookId}:${args.activityId}`,null);if(!draft)throw new Error('还没有自己的代码');download('我的代码-'+args.activityId+'.py',draft.code,'text/plain');}else{const study=read(`study:${bookId}`,{notes:[]});download('本书笔记.md',study.notes.map((n:any)=>`## ${n.chapter}\n\n${n.text}\n\n`).join(''),'text/markdown');}return '浏览器下载';}
    case 'learning_authorize_edit':case 'learning_authorize_training':throw new Error('浏览器不能授予本机执行权限，请在桌面阅读器中选择运行环境。');
    default:throw new Error('当前网页不支持这项本机操作。已有内容与记录保持完整。');
  }
}
