/** Read the retired prototype's current-origin store without changing it. */
import {sha256} from '../../site/core.mjs';
export type LegacyRecords={bookUuid:string;notes:Array<{key:string;value:any}>;runs:any[];reading:any};
async function openExisting():Promise<IDBDatabase|null>{
  if(indexedDB.databases&&!((await indexedDB.databases()).some(db=>db.name==='comfortable-open-library-v1')))return null;
  return await new Promise((resolve,reject)=>{
    const request=indexedDB.open('comfortable-open-library-v1');let created=false;
    request.onupgradeneeded=()=>{created=true;request.transaction?.abort();};
    request.onerror=()=>created?resolve(null):reject(request.error);
    request.onsuccess=()=>resolve(request.result);
  });
}
export async function legacyRecords(bookUuid:string):Promise<LegacyRecords|null>{
  const db=await openExisting();if(!db)return null;
  const result:LegacyRecords={bookUuid,notes:[],runs:[],reading:null};let bytes=0;
  try{
    for(const table of ['notes','runs']){
      if(!db.objectStoreNames.contains(table))continue;
      await new Promise<void>((resolve,reject)=>{
        const tx=db.transaction(table,'readonly'),cursor=tx.objectStore(table).openCursor();
        cursor.onsuccess=()=>{const row=cursor.result;if(!row)return;const key=String(row.key),value=row.value;
          if(table==='notes'&&key===`reading:${bookUuid}`)result.reading=value;
          if(value?.bookId===bookUuid){bytes+=JSON.stringify(value).length;if(bytes>4*1024*1024||result.notes.length+result.runs.length>10000){tx.abort();reject(new Error('旧网页记录超过本次迁移范围，请先在原页面导出'));return;}
            if(table==='runs')result.runs.push(value);else if(key.startsWith('note:')||key.startsWith('draft:'))result.notes.push({key,value});}
          row.continue();};
        tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error??new Error('读取已中断'));
      });
    }
  }finally{db.close();}
  return result.notes.length||result.runs.length||result.reading?result:null;
}
export async function mergeLegacyRecords(state:any,old:LegacyRecords):Promise<any>{
  const result=structuredClone(state);result.notes??=[];result.importedRuns??=[];
  for(const {key,value} of old.notes){
    if(typeof value.text!=='string'||!value.text.trim())continue;
    const id='legacy-'+(await sha256(new TextEncoder().encode(JSON.stringify([key,value])).buffer)).slice(0,24);
    if(result.notes.some((note:any)=>note.id===id))continue;
    result.notes.push({id,book_uuid:old.bookUuid,chapter:value.chapterId,book_revision:value.revision,content_digest:'legacy-unmapped@1',quote:key.startsWith('draft:')?'旧网页尚未提交的思考草稿':'旧网页保存的思考',text:value.text,created_at:value.updatedAt??0,legacy_anchor:value.anchor,legacy_origin:'comfortable-open-library-v1'});
  }
  for(const run of old.runs){
    if(typeof run.id!=='string')continue;const id='legacy-'+run.id;
    if(!result.importedRuns.some((row:any)=>row.record.run_id===id))result.importedRuns.push({record:{run_id:id,activity_id:run.activityId,identity:'legacy_browser_receipt',book_revision:run.bookRevision,started_at:Math.floor((run.createdAt??0)/1000),params:run.parameters,input_sha256:run.inputSha256,engine_build:run.engineBuild,result:run.result,provenance:'旧网页原记录；没有在当前版本重算，也没有可核对的历史代码快照'},snapshot:null});
  }
  if(old.reading){result.legacyReading??=[];if(!result.legacyReading.some((r:any)=>JSON.stringify(r)===JSON.stringify(old.reading)))result.legacyReading.push(old.reading);}
  if(JSON.stringify(result).length>4*1024*1024)throw new Error('合并后的记录超过保存范围，旧网页记录没有改变');
  return result;
}
