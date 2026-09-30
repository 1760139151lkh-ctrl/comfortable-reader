const CACHE='comfortable-shell-__SHELL_VERSION__';
const FILES=__SHELL_FILES__;
self.addEventListener('install',event=>event.waitUntil((async()=>{
  const cache=await caches.open(CACHE);await cache.addAll(FILES.map(file=>new URL(file,self.registration.scope).href));await self.skipWaiting();
})()));
self.addEventListener('activate',event=>event.waitUntil((async()=>{
  for(const key of await caches.keys())if(key.startsWith('comfortable-shell-')&&key!==CACHE)await caches.delete(key);
  await self.clients.claim();
})()));
self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET')return;
  const url=new URL(event.request.url),scope=self.registration.scope;
  if(url.origin!==new URL(scope).origin)return;
  // A navigation request seen by Chrome's service worker can retain a #/book
  // fragment even though its cached shell response belongs to the same path.
  url.hash='';
  if(event.request.mode==='navigate')url.search='';
  const shell=new Set(FILES.map(file=>new URL(file,scope).href));
  if(!shell.has(url.href))return; // Chapters/assets only enter the explicit resource store.
  event.respondWith((async()=>{
    const cache=await caches.open(CACHE),cached=await cache.match(url.href,{ignoreSearch:false});
    if(url.pathname.endsWith('/catalog.json')){
      try{const live=await fetch(event.request);if(live.ok)await cache.put(url.href,live.clone());return live;}
      catch{if(cached)return cached;throw new Error('目录尚未取得，且本机没有离线快照');}
    }
    if(cached)return cached;
    return fetch(event.request);
  })());
});
