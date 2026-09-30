import {invoke as nativeInvoke} from '@tauri-apps/api/core';
import {getCurrentWindow as nativeWindow} from '@tauri-apps/api/window';
import {open as nativeOpen} from '@tauri-apps/plugin-dialog';
import {webInvoke,runBuiltin,useNativePersonalStore,activeBuiltin} from './web-backend';
import {portableBooks,portableLearningPack,portableAsset} from './portable-books';
export const isDesktop=Boolean((window as any).__TAURI_INTERNALS__);
const nativeLearningBooks=new Set<string>();
if(isDesktop)useNativePersonalStore((command,args)=>nativeInvoke(command,args));
export async function invoke<T=unknown>(command:string,args:Record<string,any>={}):Promise<T>{
  if(command==='learning_run_status'&&activeBuiltin(args.runId))return structuredClone(activeBuiltin(args.runId));
  if(command==='learning_cancel'&&activeBuiltin(args.runId))return await webInvoke(command,args);
  const portable=args.bookId&&portableBooks.has(args.bookId);
  if(portable){
    if(command==='learning_pack'){
      nativeLearningBooks.delete(args.bookId);
      if(isDesktop){try{const pack=await nativeInvoke<T>(command,args);nativeLearningBooks.add(args.bookId);return pack;}catch(error){if(!/没有登记学习增强包|没有.*绑定|不存在|找不到|No such|os error 2/.test(String(error)))throw error;}}
      return portableLearningPack(args.bookId);
    }
    if(nativeLearningBooks.has(args.bookId)&&command.startsWith('learning_'))return await nativeInvoke<T>(command,args);
    if(command==='learning_asset')return await portableAsset(args.bookId,args.assetId) as T;
    if(command==='learning_run'&&portableBooks.get(args.bookId)!.book.activities.find((a:any)=>a.id===args.activityId)?.capability==='least-squares@1')return await runBuiltin(args);
    if(command==='learning_open_source'&&isDesktop){const source=(await portableLearningPack(args.bookId)).source_claims.find((s:any)=>s.id===args.sourceId);if(!source?.url)throw new Error('来源地址未登记');return await nativeInvoke('learning_open_external',{url:source.url});}
    if(['learning_open_source','learning_resource_info'].includes(command))return await webInvoke(command,args);
  }
  return isDesktop?await nativeInvoke<T>(command,args):await webInvoke(command,args);
}
export const open:typeof nativeOpen=isDesktop?nativeOpen:async()=>{throw new Error('网页仅访问你选择的书籍内容。本机文件可在桌面阅读器中导入。');};
const browserWindow={
  async isFullscreen(){return Boolean(document.fullscreenElement);},
  async setFullscreen(full:boolean){if(full)await document.documentElement.requestFullscreen();else if(document.fullscreenElement)await document.exitFullscreen();},
  async onCloseRequested(callback:(event:{preventDefault:()=>void})=>Promise<void>){const listener=()=>{void callback({preventDefault:()=>{}});};window.addEventListener('pagehide',listener);document.addEventListener('visibilitychange',()=>{if(document.hidden)listener();});return()=>window.removeEventListener('pagehide',listener);},
};
export function getCurrentWindow():ReturnType<typeof nativeWindow>{return isDesktop?nativeWindow():browserWindow as ReturnType<typeof nativeWindow>;}
