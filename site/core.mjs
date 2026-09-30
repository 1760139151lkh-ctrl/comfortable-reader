/** Portable, book-neutral identities, planning and verified acquisition. */
const id = /^[a-z][a-z0-9-]{0,63}$/;
const hash = /^[a-f0-9]{64}$/;
const MAX_MANIFEST = 2 * 1024 * 1024;
const MAX_CHAPTER = 16 * 1024 * 1024;
const MAX_CHUNK = 4 * 1024 * 1024;
export const CAPABILITIES = Object.freeze({'least-squares@1': '浏览器内置的确定性回归计算'});
export class BookFormatError extends Error {}
const fail = message => {throw new BookFormatError(message)};
const same = (a,b) => a === b;
const unique = values => new Set(values).size === values.length;
export function validPath(path) {
  if (typeof path !== 'string' || !path || /[\\:%?#]/.test(path) || path.startsWith('/')) return false;
  return path.split('/').every(part => part && part !== '.' && part !== '..' && !/[. ]$/.test(part) && !/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part));
}
export function resolveBookPath(baseUrl, path) {
  if (!validPath(path)) fail(`资源地址无效：${path}`);
  const base = new URL('.', baseUrl);
  const target = new URL(path.split('/').map(encodeURIComponent).join('/'), base);
  if (target.origin !== base.origin || !target.pathname.startsWith(base.pathname)) fail('资源越过这本书的目录');
  return target.href;
}
export function resolveCatalogPath(catalogUrl, path) {
  if (!validPath(path)) fail('目录地址无效');
  const base = new URL('.', catalogUrl);
  const target = new URL(path.split('/').map(encodeURIComponent).join('/'), base);
  if (target.origin !== base.origin || !target.pathname.startsWith(base.pathname)) fail('书籍地址越过目录');
  return target.href;
}
export function entryBytes(value, max = MAX_CHUNK) {
  if (!value || !Number.isSafeInteger(value.bytes) || value.bytes < 0 || value.bytes > max || !hash.test(value.sha256)) fail('资源大小或 SHA-256 身份无效');
  return value;
}
export function validateCatalog(catalog) {
  if (catalog?.schemaVersion !== 1 || !Array.isArray(catalog.books) || catalog.books.length > 500) fail('书目版本或书籍列表无效');
  const identities = [];
  for (const row of catalog.books) {
    if (!/^urn:uuid:[0-9a-f-]{36}$/.test(row.id) || !id.test(row.slug) || !row.title || !validPath(row.manifest?.path)) fail('书目中的身份或地址无效');
    entryBytes(row.manifest, MAX_MANIFEST);identities.push(row.id);
    if(row.epub){if(!validPath(row.epub.path))fail('EPUB 发行地址无效');entryBytes(row.epub,128*1024*1024);}
  }
  if (!unique(identities)) fail('书目中有重复的书 ID');
  return catalog;
}
export function validateBook(book) {
  if (book?.schemaVersion !== 1 || !/^urn:uuid:[0-9a-f-]{36}$/.test(book.id) || !/^\d+\.\d+\.\d+$/.test(book.revision) || !id.test(book.slug)) fail('书籍身份或格式版本无效');
  if (!Array.isArray(book.chapters) || !Array.isArray(book.resources) || !Array.isArray(book.activities) || !Array.isArray(book.sources) || book.chapters.length > 500 || book.resources.length > 5000) fail('章节或资源数量无效');
  const chapters = new Map(), resources = new Map(), activities = new Map();
  for (const chapter of book.chapters) {
    if (!id.test(chapter.id) || chapters.has(chapter.id) || !validPath(chapter.body?.path)) fail('章节 ID、正文或地址无效');
    entryBytes(chapter.body, MAX_CHAPTER);chapters.set(chapter.id, chapter);
  }
  for (const res of book.resources) {
    if (!id.test(res.id) || resources.has(res.id) || !Number.isSafeInteger(res.bytes) || res.bytes < 0 || !hash.test(res.sha256)) fail('资源身份无效');
    if (!['code','data','image','audio','video','pdf','model','geometry','text'].includes(res.kind) || !validPath(res.logicalPath)) fail('资源类型或逻辑路径无效');
    if((res.kind==='video'&&!['video/webm','video/mp4'].includes(res.mediaType))||(res.kind==='audio'&&res.mediaType!=='audio/wav')||(res.kind==='image'&&!['image/png','image/jpeg','image/webp'].includes(res.mediaType)))fail('媒体资源类型与此阅读器的能力不符');
    if(res.kind==='image'&&(!Number.isInteger(res.width)||!Number.isInteger(res.height)||res.width<1||res.height<1||res.width>8192||res.height>8192||res.width*res.height>32000000))fail('图片尺寸超出受控解码预算');
    if (res.chunks) {
      if (!Array.isArray(res.chunks) || !res.chunks.length || res.chunks.length > 2048 || res.chunks.reduce((sum, c) => sum + c.bytes, 0) !== res.bytes) fail('资源分块不完整');
      for (const c of res.chunks) {entryBytes(c);if (!validPath(c.path)) fail('分块路径无效');}
    } else {entryBytes(res,128*1024*1024);if (!validPath(res.path)) fail('资源路径无效');}
    resources.set(res.id,res);
  }
  for(const res of book.resources) {
    if(!Array.isArray(res.dependsOn??[]) || (res.dependsOn??[]).some(key=>!resources.has(key)||key===res.id))fail('资源必需依赖未登记');
  }
  for(const chapter of book.chapters){
    if(!Array.isArray(chapter.essential??[])||!unique(chapter.essential??[])||(chapter.essential??[]).some(key=>!resources.has(key)))fail('章节必需资源未登记或重复');
  }
  function visitResource(key,trail=new Set()){
    if(trail.size>32)fail('资源依赖超过当前深度预算');
    if(trail.has(key))fail('资源依赖形成循环');
    const next=new Set(trail);next.add(key);for(const dep of resources.get(key).dependsOn??[])visitResource(dep,next);
  }
  for(const key of resources.keys())visitResource(key);
  for (const activity of book.activities) {
    if (!id.test(activity.id) || activities.has(activity.id) || !chapters.has(activity.chapter) || !/^[a-z][a-z0-9-]*@[1-9]\d*$/.test(activity.capability)) fail('活动身份、章节或能力版本无效');
    if(['shell','command','cwd','pythonPath','python_path','env','trusted','script','permissions'].some(key=>Object.hasOwn(activity,key)))fail('书籍活动不能声明本机执行权');
    if (!Array.isArray(activity.required) || !activity.required.length || !Array.isArray(activity.optional) || !unique([...activity.required,...activity.optional]) || [...activity.required,...activity.optional].some(x=>!resources.has(x))) fail('活动依赖有重复或缺失');
    if (activity.capability === 'least-squares@1' && (activity.required.length !== 1 || resources.get(activity.required[0]).kind !== 'data')) fail('这个计算需要一份明确的输入数据');
    for (const [name,spec] of Object.entries(activity.parameters??{})) if (!id.test(name) || !['number','integer'].includes(spec.type) || !Number.isFinite(spec.default) || !(spec.min<=spec.default && spec.default<=spec.max)) fail('活动参数规格无效');
    if (activity.capability==='least-squares@1'&&!same(Object.keys(activity.parameters??{}).sort().join(','),'rate,steps')) fail('回归计算须说明步长和更新次数');
    activities.set(activity.id,activity);
  }
  const sourceIds = book.sources.map(s=>s.id);
  if (!unique(sourceIds) || sourceIds.some(x=>!id.test(x))) fail('来源 ID 重复或无效');
  for (const src of book.sources) if (src.url !== null && src.url !== undefined && (!/^https:\/\/[^@/]+\//.test(src.url) || src.url.includes('@') || src.url.includes('\\'))) fail('来源链接需要无凭据的 HTTPS 地址');
  return book;
}
export function validateChapter(doc,book,chapterId) {
  if (doc?.schemaVersion !== 1 || !Array.isArray(doc.nodes) || !doc.nodes.length || doc.nodes.length>8000 || !Array.isArray(doc.anchors) || !book.chapters.some(c=>c.id===chapterId)) fail('章节正文结构无效');
  const activityIds = new Set(book.activities.filter(a=>a.chapter===chapterId).map(a=>a.id));
  const resourceIds = new Set(book.resources.map(r=>r.id));const sourceIds=new Set(book.sources.map(s=>s.id));const chapters = new Map(book.chapters.map(c=>[c.id,c]));
  const headingIds=[];
  const inline=parts=>{
    if(!Array.isArray(parts)||parts.length>2000)fail('章节行内内容无效或过长');
    for(const part of parts){
      if(!part||!['text','strong','resource','source','chapter','activity'].includes(part.type))fail('书籍包含未知行内元素');
      if(part.type==='text'||part.type==='strong'){
        if(typeof part.text!=='string'||part.text.length>200000)fail('章节文字内容无效或过长');
      }else if(!id.test(part.id)||
        (part.type==='resource'&&!resourceIds.has(part.id))||
        (part.type==='source'&&!sourceIds.has(part.id))||
        (part.type==='chapter'&&!chapters.has(part.id))||
        (part.type==='activity'&&!activityIds.has(part.id)))fail('书籍行内引用没有登记');
      if(part.anchor!==undefined && (part.type!=='chapter'||!id.test(part.anchor)))fail('跨章锚点无效');
    }
  };
  for (const node of doc.nodes) {
    if(!node||!['heading','paragraph','table','code','list','activity','image'].includes(node.type))fail('章节含未经审核的组件');
    if(node.type==='heading'){
      if(!Number.isInteger(node.level)||node.level<1||node.level>3||!id.test(node.id))fail('章节标题层级或锚点无效');
      headingIds.push(node.id);inline(node.content);
    }else if(node.type==='paragraph')inline(node.content);
    else if(node.type==='activity'){
      if(!activityIds.has(node.id))fail('章节含未登记活动');
    }else if(node.type==='image'){
      const chapter=chapters.get(chapterId);
      if(!id.test(node.id)||!resourceIds.has(node.id)||!chapter.essential?.includes(node.id)||book.resources.find(r=>r.id===node.id)?.kind!=='image')fail('正文插图未作为本章必需资源登记');
    }else if(node.type==='code'){
      if(typeof node.text!=='string'||node.text.length>200000||typeof node.language!=='string'||!/^[a-zA-Z0-9_-]{0,24}$/.test(node.language))fail('代码块超出预算或语言标记无效');
    }else if(node.type==='list'){
      if(!Array.isArray(node.items)||node.items.length>2000)fail('列表结构无效');for(const item of node.items)inline(item);
    }else if(node.type==='table'){
      if(!Array.isArray(node.head)||!node.head.length||node.head.length>16||!Array.isArray(node.rows)||node.rows.length>1000)fail('表格尺寸无效');
      for(const cell of node.head)inline(cell);
      for(const row of node.rows){if(!Array.isArray(row)||row.length!==node.head.length)fail('表格行列数不一致');for(const cell of row)inline(cell);}
    }
  }
  if(doc.nodes[0].type!=='heading'||!unique(headingIds)||!unique(doc.anchors)||doc.anchors.some(a=>!id.test(a))||doc.anchors.length!==headingIds.length||doc.anchors.some(a=>!headingIds.includes(a)))fail('章节标题和锚点登记不一致');
  return doc;
}
export async function sha256(bytes) {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
}
export async function fetchVerified(url, identity, signal, maxBytes=MAX_CHUNK) {
  const controller=new AbortController();const cancel=()=>controller.abort(signal?.reason);
  if(signal?.aborted)cancel();else signal?.addEventListener('abort',cancel,{once:true});
  const timer=setTimeout(()=>controller.abort(new DOMException('取得资源超过 30 秒；已核分块保留，可重试','TimeoutError')),30000);
  try{return await fetchVerifiedBody(url,identity,controller.signal,maxBytes);}
  finally{clearTimeout(timer);signal?.removeEventListener('abort',cancel);}
}
async function fetchVerifiedBody(url, identity, signal, maxBytes) {
  entryBytes(identity, maxBytes);
  let reply;
  try{reply=await fetch(url,{signal,cache:'no-store',credentials:'omit',redirect:'error'});}
  catch(error){if(error?.name==='AbortError')throw error;throw new Error('目前无法连接这份资源；已保存的分块仍在，连接恢复后可重试');}
  if (!reply.ok || reply.type==='opaque') throw new Error(`资源暂不可用（HTTP ${reply.status}）；没有标成已保存`);
  const reader=reply.body?.getReader();let bytes;
  if(reader){const parts=[];let length=0;while(true){const {value,done}=await reader.read();if(done)break;length+=value.byteLength;if(length>identity.bytes||length>maxBytes){await reader.cancel();throw new Error('实际资源超过声明大小，已停止取得');}parts.push(value);}const joined=new Uint8Array(length);let at=0;for(const part of parts){joined.set(part,at);at+=part.byteLength;}bytes=joined.buffer;}
  else bytes=await reply.arrayBuffer();
  if (bytes.byteLength!==identity.bytes || await sha256(bytes)!==identity.sha256) throw new Error('资源版本或字节身份不符；没有替换已核内容');
  return bytes;
}
export function resourceUnits(res) {return res.chunks??[res];}
async function verifiedCachedChunk(store,unit) {
  entryBytes(unit,128*1024*1024);
  const value=await store.getChunk(unit.sha256);
  if(value===undefined)return null;
  if(value instanceof ArrayBuffer && value.byteLength===unit.bytes && await sha256(value)===unit.sha256)return value;
  if(store.removeChunkIfUnchanged)await store.removeChunkIfUnchanged(unit.sha256,value);
  else await store.removeChunk(unit.sha256);
  return null;
}
export function selection(book, kind, selectedId, optionals=[]) {
  validateBook(book);
  let chosen=[];
  if (kind==='book') {
    chosen=book.chapters.map(c=>({id:`chapter:${c.id}`,title:`正文 · ${c.title}`,kind:'chapter',...(c.readingDocument??c.body),required:true,reason:'完整书籍的正文'}));
    for(const key of new Set(book.chapters.flatMap(c=>c.essential??[]))){const resource=book.resources.find(r=>r.id===key);chosen.push({...resource,required:true,reason:'正文所需插图'});}
  }
  else if (kind==='chapter') {
    const c=book.chapters.find(c=>c.id===selectedId);if(!c)fail('找不到这章');
    chosen=[{id:`chapter:${c.id}`,title:`正文 · ${c.title}`,kind:'chapter',...(c.readingDocument??c.body),required:true,reason:'所选章节正文'}];
    for(const key of c.essential??[]){const a=book.resources.find(r=>r.id===key);if(!a)fail('章节必需插图尚未登记');chosen.push({...a,required:true,reason:'正文所需资源'});}
  } else if (kind==='resource') {
    const a=book.resources.find(r=>r.id===selectedId);if(!a)fail('找不到这份资源');chosen=[{...a,required:true,reason:'明确选择的资源'}];
  } else if(kind==='activity') {
    const a=book.activities.find(a=>a.id===selectedId);if(!a)fail('找不到这个活动');
    const c=book.chapters.find(c=>c.id===a.chapter);
    chosen=[{id:`chapter:${c.id}`,title:`正文 · ${c.title}`,kind:'chapter',...(c.readingDocument??c.body),required:true,reason:'实验对应的正文'}];
    for(const key of c.essential??[]){const r=book.resources.find(r=>r.id===key);chosen.push({...r,required:true,reason:'对应正文所需插图'});}
    for(const key of a.required){const r=book.resources.find(r=>r.id===key);chosen.push({...r,required:true,reason:'这次计算的必需输入'});}
    for(const key of optionals){if(!a.optional.includes(key))fail('所选额外资源不属于此活动');const r=book.resources.find(r=>r.id===key);chosen.push({...r,required:false,reason:'你额外选择的资源'});}
  } else fail('未知选择范围');
  if(book.reader&&['book','chapter','activity'].includes(kind)){
    for(const entry of book.reader.support??[]){
      const type=entry.path===book.reader.package?.path?'package':/\.css$/i.test(entry.path)?'style':/\.ncx$/i.test(entry.path)?'legacy-navigation':'navigation';
      const labels={package:['书籍结构信息','保持章节顺序、文件身份与阅读入口'],style:['正文排版样式','离线时保持文字、公式与图表排版'],'legacy-navigation':['目录兼容信息','保持电子书目录在不同阅读环境中的对应关系'],navigation:['全书目录','显示章节和配套资料入口；不包含其他章的正文']};
      const [title,reason]=labels[type];chosen.push({id:'reader:'+entry.path,title,kind:'metadata',...entry,required:true,reason});
    }
    if(book.studyDescriptor)chosen.push({id:'study-descriptor',title:'随书学习说明',kind:'metadata',...book.studyDescriptor,required:true,reason:'来源、活动和资源的说明，不含运行环境'});
    for(const item of [...chosen])if(item.kind==='chapter'){
      const chapter=book.chapters.find(c=>'chapter:'+c.id===item.id);
      for(const entry of chapter?.readingDependencies??[])chosen.push({id:'reader:'+entry.path,title:'本章插图',kind:'metadata',...entry,required:true,reason:'此章正文引用的插图'});
    }
  }
  chosen=Array.from(new Map(chosen.map(item=>[item.id,item])).values());
  const byId=new Map(book.resources.map(r=>[r.id,r]));
  const visited=new Set(chosen.map(x=>x.id));
  function includeDependencies(item){
    for(const key of item.dependsOn??[]){
      if(visited.has(key))continue;
      const dependency=byId.get(key);if(!dependency)fail('资源必需依赖未登记');
      visited.add(key);const row={...dependency,required:true,reason:`${item.title} 的必需依赖`};chosen.push(row);includeDependencies(row);
    }
  }
  for(const item of [...chosen])includeDependencies(item);
  const ids=chosen.map(x=>x.id);if(!unique(ids))fail('取得计划出现重复资源');
  return chosen;
}
export async function planSelection(store,book,kind,selectedId,optionals=[]) {
  const items=selection(book,kind,selectedId,optionals);
  const checked=new Map();let addedBytes=0,alreadyPresentBytes=0;
  for(const item of items){
    let ready=true;
    for(const unit of resourceUnits(item)) {
      let cached=checked.get(unit.sha256);
      if(cached && cached.bytes!==unit.bytes)fail('同一分块的大小声明不一致');
      if(!cached){
        cached={bytes:unit.bytes,ready:!!await verifiedCachedChunk(store,unit)};
        checked.set(unit.sha256,cached);
        if(cached.ready)alreadyPresentBytes+=unit.bytes;else addedBytes+=unit.bytes;
      }
      if(!cached.ready)ready=false;
    }
    item.cached=ready;
  }
  return {items,addedBytes,alreadyPresentBytes};
}
export async function acquire(store,bookUrl,item,{signal,onProgress}={}) {
  const units=resourceUnits(item);let finished=0;
  for(const unit of units){
    if(signal?.aborted) throw new DOMException('下载已取消；已核分块保留以便继续','AbortError');
    if(await verifiedCachedChunk(store,unit)){finished++;onProgress?.(finished,units.length);continue;}
    const url=resolveBookPath(bookUrl,unit.path);
    const bytes=await fetchVerified(url,unit,signal,item.kind==='chapter'||item.kind==='metadata'?64*1024*1024:128*1024*1024);
    await store.putChunk(unit.sha256,bytes);finished++;onProgress?.(finished,units.length);
  }
  return {ready:true,logicalBytes:item.bytes,verifiedUnits:units.length};
}
export async function materialize(store,item) {
  const units=resourceUnits(item),parts=[];let length=0;
  for(const unit of units){
    const value=await verifiedCachedChunk(store,unit);
    if(!value)throw new Error('已保存分块缺失或损坏；请重新准备这份资源');
    parts.push(new Uint8Array(value));length+=value.byteLength;
  }
  if(length!==item.bytes || length>128*1024*1024)throw new Error('资源总长度或读取预算不符');
  const bytes=new Uint8Array(length);let offset=0;for(const part of parts){bytes.set(part,offset);offset+=part.length;}
  if(await sha256(bytes.buffer)!==item.sha256)throw new Error('资源发布清单与分块身份不一致；不要用于实验');
  return bytes.buffer;
}
