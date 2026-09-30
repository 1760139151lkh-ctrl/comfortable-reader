/** User-owned browser notes/runs stay separate from removable public bytes. */
export class LocalStore {
  constructor(name='comfortable-open-library-v1'){this.name=name;this.database=null;}
  async open(){
    if(this.database)return this;
    this.database=await new Promise((resolve,reject)=>{
      const request=indexedDB.open(this.name,1);
      request.onupgradeneeded=()=>{const db=request.result;for(const name of ['chunks','pins','manifests','notes','runs'])if(!db.objectStoreNames.contains(name))db.createObjectStore(name);};
      request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);
    });
    return this;
  }
  async action(table,mode,kind,key,value){
    await this.open();return await new Promise((resolve,reject)=>{
      const transaction=this.database.transaction(table,mode),store=transaction.objectStore(table);
      const req=kind==='put'?store.put(value,key):kind==='delete'?store.delete(key):store.get(key);
      let result;
      req.onsuccess=()=>{result=req.result;};req.onerror=()=>reject(req.error);
      transaction.oncomplete=()=>resolve(result);transaction.onerror=()=>reject(transaction.error);transaction.onabort=()=>reject(transaction.error);
    });
  }
  async getChunk(hash){return await this.action('chunks','readonly','get',hash);}
  // Presence is not integrity; acquisition checks the declared length and digest.
  async hasChunk(hash){return Boolean(await this.getChunk(hash));}
  async removeChunkIfUnchanged(hash,observed){
    await this.open();return await new Promise((resolve,reject)=>{
      const transaction=this.database.transaction('chunks','readwrite'),store=transaction.objectStore('chunks');
      const request=store.get(hash);let removed=false;
      request.onsuccess=()=>{
        const current=request.result;
        let unchanged=!(observed instanceof ArrayBuffer) && current!==undefined && !(current instanceof ArrayBuffer);
        if(observed instanceof ArrayBuffer && current instanceof ArrayBuffer && observed.byteLength===current.byteLength){
          unchanged=true;const before=new Uint8Array(observed),now=new Uint8Array(current);
          for(let i=0;i<before.length;i++)if(before[i]!==now[i]){unchanged=false;break;}
        }
        if(unchanged){store.delete(hash);removed=true;}
      };
      request.onerror=()=>reject(request.error);
      transaction.oncomplete=()=>resolve(removed);transaction.onerror=()=>reject(transaction.error);transaction.onabort=()=>reject(transaction.error);
    });
  }
  async putChunk(hash,buffer){
    if(!/^[a-f0-9]{64}$/.test(hash)||!(buffer instanceof ArrayBuffer))throw new Error('下载分块身份无效');
    try{await this.action('chunks','readwrite','put',hash,buffer);}
    catch(error){if(error?.name==='QuotaExceededError')throw new Error('浏览器可用空间不足；已核分块仍保留，可清理未使用资源后重试');throw error;}
  }
  async removeChunk(hash){await this.action('chunks','readwrite','delete',hash);}
  async pin(owner,hashes){if(!Array.isArray(hashes)||hashes.some(h=>!/^[a-f0-9]{64}$/.test(h)))throw new Error('固定范围无效');await this.action('pins','readwrite','put',owner,[...new Set(hashes)]);}
  async unpin(owner){await this.action('pins','readwrite','delete',owner);}
  async getPins(){
    await this.open();return await new Promise((resolve,reject)=>{
      const rows=new Map(),tx=this.database.transaction('pins','readonly'),cursor=tx.objectStore('pins').openCursor();
      cursor.onsuccess=()=>{const c=cursor.result;if(c){rows.set(c.key,c.value);c.continue();}};
      tx.oncomplete=()=>resolve(rows);tx.onerror=()=>reject(tx.error);
    });
  }
  async unpinBook(bookId){for(const key of (await this.getPins()).keys())if(key.startsWith(`${bookId}:`))await this.unpin(key);}
  async unusedChunks(){
    const pinned=new Set([...await this.getPins().then(x=>x.values())].flat());
    await this.open();return await new Promise((resolve,reject)=>{
      const rows=[],tx=this.database.transaction('chunks','readonly'),cursor=tx.objectStore('chunks').openCursor();
      cursor.onsuccess=()=>{const c=cursor.result;if(c){if(!pinned.has(c.key))rows.push({sha256:c.key,bytes:c.value?.byteLength??0});c.continue();}};
      tx.oncomplete=()=>resolve(rows);tx.onerror=()=>reject(tx.error);
    });
  }
  async clearUnused(){const files=await this.unusedChunks();for(const row of files)await this.action('chunks','readwrite','delete',row.sha256);return files;}
  async putManifest(bookId,manifest){await this.action('manifests','readwrite','put',bookId,manifest);}
  async getManifest(bookId){return await this.action('manifests','readonly','get',bookId);}
  async putNote(key,note){await this.action('notes','readwrite','put',key,note);}
  async getNote(key){return await this.action('notes','readonly','get',key);}
  async deleteNote(key){await this.action('notes','readwrite','delete',key);}
  async putRun(id,run){await this.action('runs','readwrite','put',id,run);}
  async getRun(id){return await this.action('runs','readonly','get',id);}
  async listRuns(bookId){
    await this.open();return await new Promise((resolve,reject)=>{
      const rows=[],tx=this.database.transaction('runs','readonly'),cursor=tx.objectStore('runs').openCursor();
      cursor.onsuccess=()=>{const c=cursor.result;if(c){if(c.value.bookId===bookId)rows.push(c.value);c.continue();}};
      tx.oncomplete=()=>resolve(rows.sort((a,b)=>b.createdAt-a.createdAt));tx.onerror=()=>reject(tx.error);
    });
  }
}
