import {LocalStore} from '../../site/store.mjs';
import {validateCatalog,validateBook,resolveBookPath,resolveCatalogPath,fetchVerified,sha256,materialize,entryBytes,validPath} from '../../site/core.mjs';

export const publicStore=new LocalStore('comfortable-reader-content-v1');
export const portableBooks=new Map<string,PortableBook>();
export type CatalogRecord={id:string;title:string;author:string;path:string;addedAt:number;lazyPages:null;bookUuid:string;catalogSource:{url:string;sha256:string;bytes:number;slug:string;revision:string;epub?:any}};
type PortableBook={record:CatalogRecord;book:any;url:string;entries:Map<string,any>;memory:Map<string,ArrayBuffer>;urls:Map<string,string>;study?:any};
const enc=(value:unknown)=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
export async function portableId(uuid:string):Promise<string>{return (await sha256(new TextEncoder().encode('epub-identifier-v1:'+uuid.toLowerCase()).buffer)).slice(0,24);}
export async function fetchCatalog(url=new URL('catalog.json',location.href).href):Promise<{catalog:any;records:CatalogRecord[];url:string}>{
  const address=new URL(url);if(address.username||address.password||address.protocol!=='https:'&&!(address.protocol==='http:'&&['127.0.0.1','localhost'].includes(address.hostname)))throw new Error('书目需要无凭据的 HTTPS 地址或本机预览地址');
  let catalog:any;
  try{const response=await fetch(url,{credentials:'omit',cache:'no-store',redirect:'error'});if(!response.ok)throw new Error(`HTTP ${response.status}`);const reader=response.body?.getReader();if(!reader)throw new Error('当前浏览器未提供有界读取');let length=0;const chunks:Uint8Array[]=[];while(true){const {value,done}=await reader.read();if(done)break;length+=value.byteLength;if(length>1000000){await reader.cancel();throw new Error('书目超过读取范围');}chunks.push(value);}const joined=new Uint8Array(length);let at=0;for(const chunk of chunks){joined.set(chunk,at);at+=chunk.length;}catalog=validateCatalog(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(joined)));await publicStore.putManifest('catalog:'+url,catalog);}
  catch(e){catalog=await publicStore.getManifest('catalog:'+url);if(!catalog)throw e;validateCatalog(catalog);}
  const records:CatalogRecord[]=[];
  for(const row of catalog.books){records.push({id:await portableId(row.id),bookUuid:row.id,title:row.title,author:row.author||'作者信息见书籍说明',path:resolveCatalogPath(url,row.manifest.path),addedAt:0,lazyPages:null,catalogSource:{url:resolveCatalogPath(url,row.manifest.path),sha256:row.manifest.sha256,bytes:row.manifest.bytes,slug:row.slug,revision:row.revision,epub:row.epub?{...row.epub,url:resolveCatalogPath(url,row.epub.path)}:undefined}});}
  return{catalog,records,url};
}
export async function preparePortableBook(record:CatalogRecord):Promise<PortableBook>{
  const current=portableBooks.get(record.id);if(current?.record.catalogSource.sha256===record.catalogSource.sha256)return current;
  const source=record.catalogSource;let raw:ArrayBuffer=await publicStore.getChunk(source.sha256);
  if(!raw||raw.byteLength!==source.bytes||await sha256(raw)!==source.sha256){
    if(raw)await publicStore.removeChunk(source.sha256);
    raw=await fetchVerified(source.url,source,undefined,2*1024*1024);await publicStore.putChunk(source.sha256,raw);
  }
  const book=validateBook(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(raw)));
  if(book.id!==record.bookUuid||book.revision!==source.revision)throw new Error('书目与内容版本不一致，原记录保留待核对');
  const reader=book.reader;if(reader?.format!=='epub-stream@1'||!Array.isArray(reader.entries)||reader.entries.length>15000||!Array.isArray(reader.locations)||reader.locations.length>100000)throw new Error('此书尚未提供受支持的按章阅读结构');
  if(reader.epubSha256&&(!/^[a-f0-9]{64}$/.test(reader.epubSha256)||record.catalogSource.epub?.sha256!==reader.epubSha256))throw new Error('流式章节与可下载电子书的版本声明不一致');
  const entries=new Map<string,any>();
  for(const entry of reader.entries){if(!validPath(entry.path))throw new Error('章节地址无效');entryBytes(entry,64*1024*1024);const url=resolveBookPath(source.url,entry.path);if(entries.has(url))throw new Error('章节地址重复');entries.set(url,entry);}
  const encoder=new TextEncoder();
  const identities=reader.entries.map((entry:any)=>({row:[entry.path.replace(/^reader\//,''),entry.bytes,entry.sha256],order:encoder.encode(entry.path.replace(/^reader\//,''))}));
  identities.sort((a:any,b:any)=>{for(let i=0;i<Math.min(a.order.length,b.order.length);i++){if(a.order[i]!==b.order[i])return a.order[i]-b.order[i];}return a.order.length-b.order.length;});
  const contentDigest=await sha256(encoder.encode(JSON.stringify(identities.map((item:any)=>item.row))).buffer);
  if(reader.contentDigest!==contentDigest)throw new Error('内容文件清单的身份不一致；未应用旧位置或批注');
  if(!reader.locations.every((cfi:any)=>typeof cfi==='string'&&cfi.length<1000&&cfi.startsWith('epubcfi(')))throw new Error('阅读位置表无效');
  if(!entries.has(resolveBookPath(source.url,reader.package.path)))throw new Error('阅读入口未登记');
  for(const entry of [...(reader.support??[]),...book.chapters.flatMap((c:any)=>c.readingDependencies??[])]){const found=entries.get(resolveBookPath(source.url,entry.path));if(!found||found.sha256!==entry.sha256||found.bytes!==entry.bytes)throw new Error('阅读依赖没有绑定到当前书籍版本');}
  for(const chapter of book.chapters){const declared=chapter.readingDocument;if(!declared||entries.get(resolveBookPath(source.url,declared.path))?.sha256!==declared.sha256)throw new Error('章节与按需阅读文件不一致');}
  let study:any;
  if(book.studyDescriptor){
    const entry=book.studyDescriptor;entryBytes(entry,2*1024*1024);if(!validPath(entry.path))throw new Error('学习描述地址无效');
    let data=await publicStore.getChunk(entry.sha256);
    if(!data||await sha256(data)!==entry.sha256){data=await fetchVerified(resolveBookPath(source.url,entry.path),entry,undefined,2*1024*1024);await publicStore.putChunk(entry.sha256,data);}
    study=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(data));
    if(study.schema_version!==1||study.book_uuid!==book.id||!Array.isArray(study.assets)||!Array.isArray(study.chapters)||!Array.isArray(study.activities))throw new Error('学习描述与这本书不匹配');
    for(const asset of study.assets){const r=book.resources.find((r:any)=>r.id===asset.id);if(!r||r.sha256!==asset.sha256||r.bytes!==asset.bytes)throw new Error('学习材料与当前书籍版本不一致');}
    study={...study,portable:true,activities:study.activities.map((a:any)=>({...a,registered:false,editable:false}))};
  }
  const session={record,book,url:source.url,entries,memory:new Map<string,ArrayBuffer>(),urls:new Map<string,string>(),study};portableBooks.set(record.id,session);return session;
}
export async function portableEntry(session:PortableBook,url:string):Promise<ArrayBuffer>{
  const resolved=new URL(url,session.url);resolved.hash='';const entry=session.entries.get(resolved.href);if(!entry)throw new Error('这份章节资源未在当前书籍登记');
  if(session.memory.has(entry.sha256))return session.memory.get(entry.sha256)!;
  let data=await publicStore.getChunk(entry.sha256);
  if(data&&await sha256(data)!==entry.sha256){await publicStore.removeChunk(entry.sha256);data=null;}
  if(!data){try{data=await fetchVerified(resolved.href,entry,undefined,64*1024*1024);}catch{throw new Error('这部分尚未保存，当前网络也未取得。可以保留原书页，联网后重试或选择保存所需章节。');}}
  session.memory.set(entry.sha256,data);return data;
}
export function portableRequest(session:PortableBook){
  return async(url:string,type:string)=>{
    const data=await portableEntry(session,url);if(type==='binary'||type==='arraybuffer')return data;if(type==='blob')return new Blob([data]);const text=new TextDecoder('utf-8',{fatal:true}).decode(data);
    if(['xml','xhtml','html','opf','ncx'].includes(type)||/\.(?:opf|xhtml|xml|ncx)$/.test(new URL(url,session.url).pathname)){
      const doc=new DOMParser().parseFromString(text,'application/xhtml+xml');if(doc.querySelector('parsererror'))throw new Error('章节 XML 解析失败，内容未被替换');
      validateReadingDocument(doc);return doc;
    }
    if(type==='json')return JSON.parse(text);return text;
  };
}
function validateReadingDocument(doc:Document):void{
  if(doc.querySelector('script,iframe,object,embed,form,input,textarea,button,foreignObject,meta[http-equiv="refresh"]'))throw new Error('书籍含不能进入阅读视图的执行或表单元素');
  for(const element of Array.from(doc.querySelectorAll('*'))){for(const attr of Array.from(element.attributes)){if(/^on/i.test(attr.name)||/^(?:javascript|vbscript|file):/i.test(attr.value.trim())||attr.name==='srcset')throw new Error('章节含不安全的事件或地址');if(attr.name==='style'&&/(?:url\s*\(|@import|expression\s*\()/i.test(attr.value))throw new Error('行内样式不能引入额外网络资源');}}
  for(const style of Array.from(doc.querySelectorAll('style')))if(/(?:url\s*\(|@import|expression\s*\()/i.test(style.textContent??''))throw new Error('书内样式不能自行引入网络资源');
}
export async function rewritePortableResources(bookId:string,text:string,sectionUrl:string):Promise<string>{
  const session=portableBooks.get(bookId);if(!session)return text;
  const doc=new DOMParser().parseFromString(text,'application/xhtml+xml');validateReadingDocument(doc);
  const policy=doc.createElementNS('http://www.w3.org/1999/xhtml','meta');policy.setAttribute('http-equiv','Content-Security-Policy');policy.setAttribute('content',"default-src 'none'; img-src blob: data:; style-src 'unsafe-inline' blob:; font-src blob: data:; media-src blob:; connect-src 'none'; base-uri 'none'; form-action 'none'");doc.querySelector('head')?.prepend(policy);
  for(const element of Array.from(doc.querySelectorAll<HTMLElement>('img[src],link[rel="stylesheet"][href]'))){
    const attribute=element.tagName.toLowerCase()==='img'?'src':'href',relative=element.getAttribute(attribute)!;
    if(relative.startsWith('data:image/png;')||relative.startsWith('blob:')&&[...session.urls.values()].includes(relative))continue;
    const url=new URL(relative,sectionUrl).href,entry=session.entries.get(url);if(!entry)throw new Error('插图或样式不属于当前登记版本');
    let blob=session.urls.get(url);if(!blob){const data=await portableEntry(session,url);const mime=attribute==='src'?'image/png':'text/css';if(attribute==='href'&&/(?:url\s*\(|@import|expression\s*\()/i.test(new TextDecoder().decode(data)))throw new Error('此样式包含尚未登记适配的外部依赖，未发出额外请求');blob=URL.createObjectURL(new Blob([data],{type:mime}));session.urls.set(url,blob);}
    element.setAttribute(attribute,blob);
  }
  return new XMLSerializer().serializeToString(doc);
}
export async function portableAsset(bookId:string,id:string):Promise<ArrayBuffer>{
  const session=portableBooks.get(bookId);if(!session)throw new Error('书籍尚未准备');const asset=session.book.resources.find((a:any)=>a.id===id);if(!asset)throw new Error('资源未登记');
  try{return await materialize(publicStore,asset);}catch{}
  const units=asset.chunks??[asset],parts:Uint8Array[]=[];for(const unit of units){parts.push(new Uint8Array(await fetchVerified(resolveBookPath(session.url,unit.path),unit,undefined,64*1024*1024)));}
  const data=new Uint8Array(parts.reduce((n,p)=>n+p.length,0));let offset=0;for(const part of parts){data.set(part,offset);offset+=part.length;}if(await sha256(data.buffer)!==asset.sha256)throw new Error('资源内容与版本不一致');return data.buffer;
}
export function portableLearningPack(bookId:string):any{
  const session=portableBooks.get(bookId);if(!session)throw new Error('书籍尚未准备');const b=session.book;
  if(session.study)return session.study;
  const usage=(id:string)=>b.chapters.filter((c:any)=>(c.essential??[]).includes(id)||(c.resourceIds??[]).includes(id)||b.resources.find((r:any)=>r.id===id)?.chapters?.includes(c.id)||b.activities.some((a:any)=>a.chapter===c.id&&[...a.required,...a.optional].includes(id))).map((c:any)=>c.id);
  return{schema_version:1,book_uuid:b.id,book_revision_sha256:session.record.catalogSource.sha256,
    chapters:b.chapters.map((c:any,i:number)=>({id:c.id,number:i+1,title:c.title,question:c.question??'',href:c.readingDocument.path.split('/').pop(),assets:b.resources.filter((r:any)=>usage(r.id).includes(c.id)).map((r:any)=>r.id)})),
    assets:b.resources.map((r:any)=>({id:r.id,relative_path:r.logicalPath,sha256:r.sha256,bytes:r.bytes,kind:r.kind,title:r.title,filename:r.logicalPath.split('/').pop(),role:r.role??'author_recorded',chapters:usage(r.id),media:r.media})),
    href_assets:Object.fromEntries(b.resources.map((r:any)=>['resources.xhtml#resource-'+r.id,r.id])),
    href_activities:Object.fromEntries(b.activities.flatMap((a:any)=>[['chapter-'+a.chapter+'.xhtml#activity-'+a.id,a.id],['resources.xhtml#activity-'+a.id,a.id]])),
    activities:b.activities.map((a:any)=>({id:a.id,chapter:a.chapter,title:a.title,description:a.question,entry_asset:a.optional.find((id:string)=>b.resources.find((r:any)=>r.id===id)?.kind==='code')??a.required[0],runtime:a.capability==='least-squares@1'?'builtin':'native',capability:a.capability,editable:false,registered:a.capability==='least-squares@1',timeout_seconds:30,parameters:Object.entries(a.parameters??{}).map(([name,p])=>({name,...p as any})),identity:'declared_book_activity'})),
    source_claims:b.sources.flatMap((s:any)=>b.chapters.filter((c:any)=>(c.sourceIds??[]).includes(s.id)).map((c:any)=>({id:s.id+'-'+c.id,chapter:c.id,title:s.title,url:s.url,claim:s.supports,citation:s.locator,source_locator:s.locator,attribution_from_book:s.creator,publication_year_from_book:s.year,metadata_status:s.kind==='teaching_construction'?'教学构造，非外部测量':'书籍声明的来源信息'}))),
    audio_groups:[],dependencies:[],history:[],portable:true};
}
export function readerPackageUrl(bookId:string):string{return resolveBookPath(portableBooks.get(bookId)!.url,portableBooks.get(bookId)!.book.reader.package.path);}
export function escapePortableText(value:unknown):string{return enc(value);}
