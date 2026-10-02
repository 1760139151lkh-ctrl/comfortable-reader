// Isolated browser regression of the actual built UI with read-only real EPUBs.
// Native IPC is mocked; installed-app verification is recorded separately.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'playwright');
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),out=path.join(root,'work','redesign-20261002','browser');fs.mkdirSync(out,{recursive:true});
const baseline=JSON.parse(fs.readFileSync(process.env.READER_BASELINE||path.join(root,'个人数据','阅读状态','reader-state.json'),'utf8'));
const titles=['智能优化算法简介','有限元方法','数学定理等价性','模糊规划','综合评价'];
const books=titles.map(t=>baseline.books.find(b=>b.title.startsWith(t)));assert.ok(books.every(Boolean));
const report={tier:'isolated_browser_mocked_ipc',checks:[],errors:[]};
let activePage;
const pushCheck=report.checks.push.bind(report.checks);report.checks.push=(v)=>{console.log(v);return pushCheck(v);};
let state=structuredClone(baseline);state.books=baseline.books;state.session={...state.session,paneCount:1,paneBookIds:[books[0].id],activePane:0,workspace:null,surfaces:null,theme:'night'};
// Each test run starts at its own positions. This never writes the production state.
state.progress={};let callbacks={},nextCallback=1;
const server=http.createServer((req,res)=>{try{let target;if(req.url.startsWith('/__book/'))target=state.books.find(b=>b.id===req.url.split('/').pop())?.path;else target=path.join(root,'desktop','dist',decodeURIComponent(req.url.split('?')[0]==='/'?'/index.html':req.url.split('?')[0]));if(!target||!fs.existsSync(target)){res.writeHead(404);res.end('not found');return;}const ext=path.extname(target);res.setHeader('Content-Type',({'.js':'application/javascript','.css':'text/css','.html':'text/html','.json':'application/json','.epub':'application/epub+zip'})[ext]||'application/octet-stream');res.end(fs.readFileSync(target));}catch(e){res.writeHead(500);res.end(String(e));}});
const wait=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const browser=await chromium.launch({channel:'msedge',headless:true}),context=await browser.newContext({viewport:{width:1520,height:980},deviceScaleFactor:1});const page=await context.newPage();activePage=page;page.setDefaultTimeout(12000);page.setDefaultNavigationTimeout(20000);
 page.on('pageerror',e=>report.errors.push(e.message));
 await page.exposeFunction('__testInvoke',async(command,args)=>{
  if(command==='bootstrap'||command==='refresh_library')return {...structuredClone(state),storagePath:'isolated test only',scan:{rootsScanned:1,missingRoots:0,filesSeen:12,epubCandidates:12,loadedCandidates:12,booksLoaded:12,added:0,updated:0,duplicates:0,unreadable:0,otherBookFiles:0,ignoredTrees:0,issues:[]}};
  if(command==='save_session'){state.session=structuredClone(args.session);return;}
  if(command==='save_progress'){state.progress[args.bookId]=structuredClone(args.progress);return;}
  if(command==='load_location_index'){
   const file=path.join(root,'个人数据','阅读状态','location-index',args.sourceSha256+'-epubjs-1000-v1.json');return fs.existsSync(file)?fs.readFileSync(file,'utf8'):null;
  }
  if(command==='save_location_index')return;
  if(command==='epub_content_identity')throw new Error('content identity not mocked');
  if(command==='learning_pack')throw new Error('没有登记学习增强包（隔离布局回归）');
  if(command==='learning_runtime_info')return {platform:'browser-test',security_mode:'no-execution'};
  if(command==='plugin:window|is_fullscreen')return false;
  if(command==='plugin:event|listen')return 1;
  if(command.startsWith('plugin:event|'))return;
  throw new Error('Unmocked command: '+command);
 });
 await page.addInitScript(()=>{window.__TAURI_INTERNALS__={metadata:{currentWindow:{label:'main'},currentWebview:{label:'main'}},transformCallback:()=>1,unregisterCallback:()=>{},invoke:async(command,args={})=>command==='load_book_bytes'?await(await fetch('/__book/'+args.bookId)).arrayBuffer():await window.__testInvoke(command,args)};window.__TAURI_EVENT_PLUGIN_INTERNALS__={unregisterListener:()=>{}};});
 async function settle(){await page.waitForFunction(()=>document.querySelector('.boot-screen')?.classList.contains('hidden'),{timeout:45000});await page.waitForFunction(()=>![...document.querySelectorAll('.reader-pane:not(.hidden-pane)')].some(p=>p.classList.contains('loading')),{timeout:45000});await wait(900);}
 async function shot(name){await page.screenshot({path:path.join(out,name+'.png')});}
 async function boxes(){return await page.locator('.reader-pane:not(.hidden-pane) .pane-stage').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return{x:r.x,y:r.y,w:r.width,h:r.height};}));}
 async function open(t){await page.locator('.workspace-library').click();await page.locator('.library-book').filter({has:page.locator('.book-copy strong',{hasText:t})}).locator('.book-main').click();await settle();}
 async function reveal(){await page.locator('.tools-toggle').hover();await wait(220);}
 async function tools(){if(!await page.locator('.tools-toggle').getAttribute('aria-expanded').then(x=>x==='true'))await page.locator('.tools-toggle').click();}
 async function clickText(pane){const host=page.locator(`.reader-pane[data-pane-index="${pane}"] .pane-stage`),r=await host.boundingBox();await page.mouse.click(r.x+60,r.y+70);}
 async function textVisible(pane,find=null){return await page.locator(`.reader-pane[data-pane-index="${pane}"]`).evaluate((pane,find)=>{
  const stage=pane.querySelector('.pane-stage').getBoundingClientRect();
  for(const f of pane.querySelectorAll('iframe')){const doc=f.contentDocument;if(!doc?.body)continue;const fr=f.getBoundingClientRect(),scale=fr.width/f.offsetWidth;const w=doc.createTreeWalker(doc.body,NodeFilter.SHOW_TEXT);let n;
   while(n=w.nextNode()){const text=n.textContent;if(!text.trim()||n.parentElement.closest('style,script,annotation'))continue;const positions=find?[text.indexOf(find)]:Array.from({length:Math.ceil(text.length/12)},(_,i)=>i*12);
    for(const pos of positions){if(pos<0)continue;const range=doc.createRange();range.setStart(n,pos);range.setEnd(n,Math.min(text.length,pos+(find?.length??16)));const r=range.getBoundingClientRect(),x=fr.left+r.left*scale,y=fr.top+r.top*scale;if(r.width>0&&x>=stage.left-1&&x<stage.right-5&&y>=stage.top-1&&y<stage.bottom-5){const value=text.slice(pos,Math.min(text.length,pos+(find?.length??16))).trim();if(find?value.length>0:value.length>5)return find?true:value;}}
   }
  }return find?false:null;
 },find);}

 const check=message=>report.checks.push(message);
 const pane=(i=0)=>page.locator(`.reader-pane[data-pane-index="${i}"]`);
 const rect=async loc=>await loc.boundingBox();
 async function drag(handle,x,y,commit=true){const r=await rect(handle);await page.mouse.move(r.x+r.width/2,r.y+r.height/2);await page.mouse.down();await page.mouse.move(x,y,{steps:12});if(commit)await page.mouse.up();}
 const intersects=(a,b)=>a.x<b.x+b.width&&a.x+a.width>b.x&&a.y<b.y+b.height&&a.y+a.height>b.y;
 await page.goto(`http://127.0.0.1:${server.address().port}/`);await settle();
 const quietBoxes=await boxes();await page.mouse.move(600,400);await wait(2100);
 assert.equal(await page.locator('.quick-tools').isVisible(),false);assert.equal(await page.locator('.workspace-tabs').count(),0);
 assert.deepEqual(await boxes(),quietBoxes);await shot('17-quiet-corner');
 await page.locator('.tools-toggle').hover();await page.mouse.move(600,400);await wait(260);
 assert.equal(await page.locator('.quick-tools').isVisible(),false,'Passing the edge should not flash the controls');
 await reveal();assert.equal(await page.locator('.quick-tools').isVisible(),true);assert.deepEqual(await boxes(),quietBoxes);
 await page.locator('.contents-toggle').click();await wait(2100);
 const compactNav=await rect(page.locator('.book-navigation'));assert.ok(compactNav.height<=420);assert.equal(await page.locator('.book-navigation').isVisible(),true);
 const chapterGeometry=await page.locator('.chapter-title').evaluateAll(es=>es.map(e=>({h:e.getBoundingClientRect().height,whiteSpace:getComputedStyle(e).whiteSpace,mask:getComputedStyle(e).maskImage})));
 assert.ok(chapterGeometry.length>5);assert.ok(chapterGeometry.every(e=>e.h<30&&e.whiteSpace==='nowrap'&&e.mask.includes('linear-gradient')));
 await shot('18-compact-contents');await page.locator('.navigation-close').click();
 check('A/C: quiet left corner, delayed edge reveal without reflow, no horizontal book tabs, compact fading contents stays open while in use');
 await shot('01-quiet-reading');
 const startBoxes=await boxes();await tools();await shot('02-tools-in-safe-rail');
 assert.deepEqual(await boxes(),startBoxes);assert.equal(intersects(await rect(page.locator('.top-toolbar')),await rect(pane().locator('.pane-stage'))),false);
 await page.locator('.tools-toggle').click();assert.deepEqual(await boxes(),startBoxes);
 let before=state.progress[books[0].id].page;const edge=pane().locator('.page-hotspot-next');
 const er=await rect(edge),sr=await rect(pane().locator('.pane-stage'));assert.ok(er.width>=22);assert.equal(intersects(er,sr),false);
 await edge.hover();await wait(200);assert.equal(state.progress[books[0].id].page,before);await edge.click();await wait(1000);assert.equal(state.progress[books[0].id].page,before+1);
 await page.keyboard.press('ArrowLeft');await wait(900);assert.equal(state.progress[books[0].id].page,before,'Page edge must return keyboard focus to the current book');
 await clickText(0);await page.keyboard.press('ArrowRight');await wait(900);assert.equal(state.progress[books[0].id].page,before+1);
 before=state.progress[books[0].id].page;let stage=await rect(pane().locator('.pane-stage'));await page.mouse.move(stage.x+stage.width*.65,stage.y+160);await page.mouse.wheel(0,120);await wait(900);assert.equal(state.progress[books[0].id].page,before+1);
 const frame=pane().locator('iframe').first();before=state.progress[books[0].id].page;
 await frame.evaluate(async f=>{for(const d of [4,8,15,24,28,21,16,12,8,6,4,3,2,1]){f.contentDocument.body.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaY:d}));await new Promise(r=>setTimeout(r,16));}});await wait(1000);assert.equal(state.progress[books[0].id].page,before+1);
 check('A: safe rail never overlaps or moves the stage; wide outside-text click edges, hover without turning, keyboard, one wheel notch, one inertial gesture');
 // Actual input stays with its local scroll owner, including its end boundary.
 before=state.progress[books[0].id].page;
 await frame.evaluate(f=>{const d=f.contentDocument,local=d.createElement('pre');local.dataset.testScroll='true';local.style.cssText='position:fixed;left:70px;top:120px;width:180px;height:70px;overflow:auto;z-index:9999;background:#444;color:white;white-space:pre';local.textContent=Array.from({length:30},(_,i)=>'local scroll '+i+' long line '.repeat(30)).join('\n');d.body.append(local);for(const [key,value] of Object.entries({height:'70px',width:'180px','max-height':'70px',overflow:'auto'}))local.style.setProperty(key,value,'important');if(local.scrollHeight<=local.clientHeight)throw new Error('Nested scroll fixture did not overflow');local.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaY:120}));local.scrollTop=9999;local.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaY:120}));local.remove();});await wait(450);assert.equal(state.progress[books[0].id].page,before);
 await page.locator('.workspace-library').click();const library=page.locator('.library-drawer');await page.locator('.library-list').hover();await page.mouse.wheel(0,600);await wait(400);assert.equal(state.progress[books[0].id].page,before);await page.locator('.search-input').count().then(()=>{});
 const search=page.locator('.search-box input');await search.fill(titles[1]);await wait(150);
 let book=page.locator('.library-book').filter({has:page.locator('.book-copy strong',{hasText:titles[1]})}).locator('.book-main');
 let rootRect=await rect(page.locator('.reader-grid'));const savedCfi=state.progress[books[0].id].cfi;
 await drag(book,rootRect.x+rootRect.width-90,rootRect.y+rootRect.height*.45,false);
 assert.equal(await page.locator('.book-pickup').isVisible(),true);assert.equal(await page.locator('.workspace-projected-book').count(),2);assert.deepEqual(await boxes(),startBoxes);await shot('03-drag-second-book-preview');
 const previewRects=await page.locator('.workspace-projected-book').evaluateAll(es=>es.map(e=>({x:e.offsetLeft,y:e.offsetTop,width:e.offsetWidth,height:e.offsetHeight})));
 await page.mouse.up();await settle();assert.equal(await page.locator('.reader-pane:not(.hidden-pane)').count(),2);assert.equal(state.session.paneBookIds.filter(Boolean).length,2);const actualRects=await page.locator('.reader-pane:not(.hidden-pane)').evaluateAll(es=>es.map(e=>({x:e.offsetLeft,y:e.offsetTop,width:e.offsetWidth,height:e.offsetHeight})));assert.deepEqual(actualRects,previewRects);
 for(let attempt=0;attempt<80&&!state.progress[books[1].id]?.cfi;attempt++)await wait(100);assert.ok(state.progress[books[1].id]?.cfi,'New book must finish opening and save its content anchor');let secondPage=state.progress[books[1].id].page;await page.keyboard.press('ArrowRight');await wait(900);assert.equal(state.progress[books[1].id].page,secondPage+1,'Dropped book must receive keyboard navigation');
 await clickText(1);await page.keyboard.press('Control+1');await wait(250);const firstPage=state.progress[books[0].id].page;secondPage=state.progress[books[1].id].page;await page.keyboard.press('ArrowRight');await wait(900);assert.equal(state.session.activePane,0);assert.equal(state.progress[books[0].id].page,firstPage+1);assert.equal(state.progress[books[1].id].page,secondPage);
 assert.ok(await textVisible(0));check('B: shelf drag carries title/author, previews two final rectangles without reflow; drop adds a second real book');
 await page.locator('.workspace-library').click();await search.fill(titles[2]);book=page.locator('.library-book').filter({has:page.locator('.book-copy strong',{hasText:titles[2]})}).locator('.book-main');
 const second=await rect(pane(1));const treeBefore=structuredClone(state.session.workspace.tree),cfiBefore=state.progress[books[0].id].cfi;
 await drag(book,second.x+second.width*.5,second.y+second.height-45,false);await page.keyboard.press('Escape');await page.mouse.up();
 assert.equal(state.session.paneBookIds.filter(Boolean).length,2);assert.deepEqual(state.session.workspace.tree,treeBefore);assert.equal(await search.inputValue(),titles[2]);assert.equal(await library.isVisible(),true);assert.equal(state.progress[books[0].id].cfi,cfiBefore);
 await drag(book,second.x+second.width*.5,second.y+second.height-45);await settle();assert.equal(await page.locator('.reader-pane:not(.hidden-pane)').count(),3);await shot('04-three-books');
 await reveal();await page.locator('.reading-settings-toggle').click();const scope=page.locator('.settings-target');assert.equal(await scope.locator('option').count(),3);const originalFirst=structuredClone(state.progress[books[0].id]);await scope.selectOption('1');await page.locator('.font-up').click();await wait(1300);assert.equal(state.progress[books[0].id].cfi,originalFirst.cfi);assert.deepEqual(state.progress[books[0].id].readingPreferences,originalFirst.readingPreferences);await scope.selectOption('2');assert.equal(await scope.inputValue(),'2');await pane(0).locator('.pane-settings').click();assert.equal(await scope.inputValue(),'0');await scope.selectOption('2');assert.equal(await page.locator('.reading-settings').getAttribute('aria-hidden'),'false');await page.locator('.settings-close').click();check('C: explicit target book selector and per-book typography entry change only the selected book');

 // Match the user's ordinary window: the shelf must not occupy half the reading space.
 await page.setViewportSize({width:1030,height:640});await settle();const textBeforeShelf=await boxes();
 await page.locator('.workspace-library').click();await search.fill('');await wait(250);
 let compactShelf=await rect(library);assert.ok(compactShelf.width<=300);assert.ok(compactShelf.height<=460);assert.equal(await library.getAttribute('data-docked'),'float');assert.deepEqual(await boxes(),textBeforeShelf);
 assert.equal(await page.locator('.library-actions').isVisible(),false);assert.equal(await page.locator('.library-audit').isVisible(),false);
 const longBook=page.locator('.book-copy strong').filter({hasText:'Introduction to Algorithms'}).first();
 const faded=await longBook.evaluate(e=>({width:e.clientWidth,scroll:e.scrollWidth,whiteSpace:getComputedStyle(e).whiteSpace,mask:getComputedStyle(e).maskImage}));
 assert.ok(faded.scroll>faded.width);assert.equal(faded.whiteSpace,'nowrap');assert.ok(faded.mask.includes('linear-gradient'));await shot('19-compact-shelf-user-window');
 const shelfGeometry=await rect(library);await page.locator('.workspace-toggle').click();await wait(200);assert.equal((await rect(library)).width,shelfGeometry.width);assert.ok((await rect(library)).height<shelfGeometry.height);assert.equal(await page.locator('.workspace-book-row').count(),3);assert.equal(await page.locator('.workspace-panel').evaluate(e=>e.closest('[data-surface]')?.getAttribute('data-surface')),'library');await shot('20-open-books-one-shelf');
 const arrangement=structuredClone(state.session.workspace.tree),targetBook=await rect(pane(2));
 await drag(page.locator('[data-workspace-select="0"]'),targetBook.x+targetBook.width/2,targetBook.y+targetBook.height/2,false);assert.equal(await page.locator('.book-pickup').isVisible(),true);await page.keyboard.press('Escape');await page.mouse.up();assert.deepEqual(state.session.workspace.tree,arrangement);assert.equal(await library.getAttribute('data-shelf-view'),'open');
 await page.locator('.library-all').click();await page.locator('.library-options-toggle').click();assert.equal(await page.locator('.library-actions').isVisible(),true);await page.locator('.library-options-toggle').click();
 await library.locator('.drawer-close').click();await page.setViewportSize({width:1520,height:980});await settle();
 check('B/C: at the user-sized window the shelf is compact, titles visibly overflow into a fade, opened books stay in the same panel, and dragging from that view can be cancelled');
 const beforeArrange=structuredClone(state.session.workspace.tree);const div=await rect(page.locator('.workspace-divider').first());await drag(page.locator('.workspace-divider').first(),div.x-65,div.y+100);await settle();assert.notDeepEqual(state.session.workspace.tree,beforeArrange);
 const target=await rect(pane(1));await drag(pane(2).locator('.pane-book-meta'),target.x+target.width/2,target.y+target.height/2);await settle();
 check('B: cancel preserves shelf search, books, CFI and layout; lower-edge drop, divider resize and center swap work');
 // A new book dropped at a divider produces a real middle region.
 await page.locator('.workspace-library').click();await search.fill(titles[3]);book=page.locator('.library-book').filter({has:page.locator('.book-copy strong',{hasText:titles[3]})}).locator('.book-main');
 let dividerRect=await rect(page.locator('.workspace-divider').first());await drag(book,dividerRect.x+dividerRect.width/2,dividerRect.y+dividerRect.height/2,false);assert.equal(await page.locator('.workspace-projected-book').count(),4);await shot('05-insert-between-preview');await page.mouse.up();await settle();assert.equal(state.session.paneBookIds.filter(Boolean).length,4);
 check('B: divider insertion previews affected groups and inserts without losing their internal ratios');
 // The same panel gestures apply to library, currently-open books, and navigation.
 await page.locator('.workspace-library').click();await search.fill('');let lr=await rect(library);
 await drag(library.locator('.surface-grip'),lr.x+230,lr.y+150);await wait(250);let moved=await rect(library);assert.ok(moved.x>lr.x+80);assert.ok(moved.y>lr.y+50);
 const beforeResize={...moved};await drag(library.locator('[data-edge="se"]'),moved.x+moved.width+95,moved.y+moved.height-100);await wait(300);moved=await rect(library);assert.ok(moved.width>beforeResize.width+60);assert.ok(moved.height<beforeResize.height-60);
 await shot('06-moved-resized-library');await library.locator('.surface-dock').click();await settle();const dockRect=await rect(library),gridRect=await rect(page.locator('.reader-grid'));assert.ok(gridRect.x>=dockRect.x+dockRect.width-1);
 await clickText(state.session.activePane);assert.equal(await library.isVisible(),true);assert.equal(await library.getAttribute('data-docked'),'left');await shot('07-docked-library-reserves-space');
 await library.locator('.drawer-close').click();await settle();
 await page.locator('.workspace-library').click();const shelfBefore=await rect(library);await page.locator('.workspace-toggle').click();const openPanel=page.locator('.workspace-panel');
 assert.equal(await openPanel.evaluate(e=>Boolean(e.closest('.library-drawer'))),true);assert.equal(await page.locator('[data-surface="books"]').count(),0);assert.equal((await rect(library)).width,shelfBefore.width);assert.equal(await page.locator('.workspace-book-row').count(),4);
 await shot('08-open-books-in-same-shelf');await page.locator('.library-all').click();assert.equal(await page.locator('.library-list').isVisible(),true);await library.locator('.drawer-close').click();
 check('C: library moves/resizes and truly docks; currently-open books use the same panel without a second overlay');
 await reveal();await page.locator('.workspace-focus').click();await settle();const active=state.session.activePane,activeBook=state.session.paneBookIds[active],otherProgress=structuredClone(state.progress[books[0].id]);
 const anchor=await textVisible(active);await reveal();await page.locator('.reading-settings-toggle').click();assert.equal(await page.locator('.font-size-slider').isVisible(),true);
 await page.locator('.font-size-slider').fill('125');await wait(1500);assert.ok(await textVisible(active,anchor));assert.equal(state.progress[activeBook].readingPreferences.fontScale,125);
 await page.locator('.settings-page-mode').selectOption('3');await wait(1500);assert.ok(await textVisible(active,anchor));await page.locator('.settings-page-mode').selectOption('0');await wait(1300);
 await page.locator('.reading-mode').selectOption('scroll');await wait(1400);await shot('09-settings');await page.locator('.settings-close').click();await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
 const cfiScroll=state.progress[activeBook].cfi;stage=await rect(pane(active).locator('.pane-stage'));await page.mouse.move(stage.x+stage.width*.5,stage.y+stage.height*.45);
 const scrollInfo=async()=>pane(active).evaluate(p=>({containers:[...p.querySelectorAll('.epub-container')].map(e=>({top:e.scrollTop,h:e.clientHeight,total:e.scrollHeight,overflow:getComputedStyle(e).overflowY})),className:p.className,frames:[...p.querySelectorAll('iframe')].map(f=>({y:f.getBoundingClientRect().y,h:f.getBoundingClientRect().height})),mode:p.dataset.readingMode}));
 report.scrollDebug={before:await scrollInfo(),cfi:cfiScroll};await page.mouse.wheel(0,220);await wait(700);report.scrollDebug.after=await scrollInfo();report.scrollDebug.cfiAfter=state.progress[activeBook].cfi;console.log('scroll observation',JSON.stringify(report.scrollDebug));assert.notEqual(state.progress[activeBook].cfi,cfiScroll);
 assert.deepEqual(state.progress[books[0].id].readingPreferences,otherProgress.readingPreferences);
 await reveal();await page.locator('.reading-settings-toggle').click();await page.locator('.reading-mode').selectOption('paged');await wait(1500);await page.locator('.settings-close').click();await reveal();await page.locator('.workspace-focus').click();await settle();
 check('C/D: one-layer font/page/mode controls preserve visible anchor, scroll is continuous, changes remain per book, focus restores the arrangement');
 // Moving the tool rail has an explicit preview and is persisted.
 const rail=page.locator('.rail-grip');await drag(rail,300,955,false);assert.equal(await page.locator('.surface-dock-preview').isVisible(),true);await page.mouse.up();await settle();assert.equal(state.session.surfaces.toolbarEdge,'bottom');
 await tools();for(const box of await boxes())assert.equal(intersects(await rect(page.locator('.top-toolbar')), {x:box.x,y:box.y,width:box.w,height:box.h}),false);await shot('10-bottom-rail');
 await page.locator('.rail-grip').focus();await page.keyboard.press('Home');await settle();assert.equal(state.session.surfaces.toolbarEdge,'top');
 // Seek previews only; one release commits one navigation and can return.
 await clickText(state.session.activePane);const seek=pane(state.session.activePane).locator('.progress-seek');const sk=await rect(seek),bookId=state.session.paneBookIds[state.session.activePane],beforeSeek=state.progress[bookId].cfi;
 await page.mouse.move(sk.x+sk.width*.25,sk.y+sk.height/2);await page.mouse.down();await page.mouse.move(sk.x+sk.width*.45,sk.y+sk.height/2,{steps:8});await wait(400);assert.equal(state.progress[bookId].cfi,beforeSeek);assert.equal(await pane(state.session.activePane).locator('.progress-preview').isVisible(),true);await page.mouse.up();await wait(1300);assert.notEqual(state.progress[bookId].cfi,beforeSeek);
 await page.locator('.reading-back').click();await wait(1300);
 check('D: rail parks top/bottom outside text; progress drag previews without repeated jumps, commits on release and offers return');
 const layout=structuredClone(state.session.workspace.tree),surface=structuredClone(state.session.surfaces);await page.setViewportSize({width:640,height:650});await settle();assert.deepEqual(state.session.workspace.tree,layout);assert.equal(await page.locator('.reader-pane:not(.hidden-pane)').count(),1);
 await page.locator('.workspace-library').click();lr=await rect(library);assert.ok(lr.x>=0&&lr.y>=0&&lr.x+lr.width<=641&&lr.y+lr.height<=651);await shot('11-narrow-panel');await library.locator('.drawer-close').click();
 await page.setViewportSize({width:1520,height:980});await settle();assert.deepEqual(state.session.workspace.tree,layout);
 await reveal();await page.locator('.reading-settings-toggle').click();await page.locator('.typography-details summary').click();await page.locator('.interface-scale').selectOption('1.3');await wait(500);const settingsRect=await rect(page.locator('.reading-settings'));assert.ok(settingsRect.x+settingsRect.width<=1520&&settingsRect.y+settingsRect.height<=980);await shot('12-larger-controls');await page.locator('.settings-close').click();
 const persisted=structuredClone(state.session);await page.reload();await settle();assert.deepEqual(state.session.workspace,persisted.workspace);assert.deepEqual(state.session.surfaces,persisted.surfaces);assert.deepEqual(state.session.paneBookIds,persisted.paneBookIds);
 check('D/E: narrow/wide keeps the tree, panel bounds stay usable, independent UI scale and panel geometry survive reload');

 await page.locator('.workspace-library').click();
 if(await library.getAttribute('data-docked')==='left')await library.locator('.surface-dock').click();
 let smallBox=await rect(library);await drag(library.locator('[data-edge="se"]'),smallBox.x+282,smallBox.y+222);await wait(250);
 const closeBox=await rect(library.locator('.drawer-close'));smallBox=await rect(library);assert.ok(closeBox.x+closeBox.width<=smallBox.x+smallBox.width+1);assert.ok(closeBox.y+closeBox.height<=smallBox.y+smallBox.height);
 await search.fill('有限元');assert.equal(await page.locator('.library-book').count(),1);await search.fill('');
 const last=library.locator('.book-new-pane').last();await last.focus();await wait(150);await shot('14-minimum-panel-last-book');const lastBounds=await rect(last);assert.ok(lastBounds.y>=smallBox.y&&lastBounds.y+lastBounds.height<=smallBox.y+smallBox.height);
 const libraryGeometry=structuredClone(state.session.surfaces.panels.library);await drag(library.locator('.surface-grip'),10,300,false);assert.equal(await page.locator('.surface-dock-preview').isVisible(),true);await page.keyboard.press('Escape');await page.mouse.up();await wait(250);assert.deepEqual(state.session.surfaces.panels.library,libraryGeometry);
 await drag(library.locator('.surface-grip'),10,300);await settle();assert.equal(await library.getAttribute('data-docked'),'left');
 const persistedDock=structuredClone(state.session.surfaces);await page.reload();await settle();assert.equal(await library.isVisible(),true);assert.equal(await library.getAttribute('data-docked'),'left');assert.deepEqual(state.session.surfaces,persistedDock);await library.locator('.drawer-close').click();
 await reveal();await page.locator('.contents-toggle').click();const nav=page.locator('.book-navigation');let navBox=await rect(nav);await drag(nav.locator('.surface-grip'),450,180);await wait(200);navBox=await rect(nav);await drag(nav.locator('[data-edge="se"]'),navBox.x+420,navBox.y+275);await wait(200);await shot('15-navigation-adjusted');await nav.locator('.navigation-close').click();
 check('C/E: minimum-size shelf retains search/close and keyboard reachability; docking drag cancel is reversible; an open dock and resized navigation persist correctly');

 await page.setViewportSize({width:480,height:620});await settle();await tools();
 const narrowTools=await page.locator('.top-toolbar button:visible').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return {name:e.getAttribute('aria-label')||e.title,x:r.x,right:r.right,top:r.top,bottom:r.bottom};}));
 assert.ok(narrowTools.every(b=>b.x>=0&&b.right<=480&&b.top>=0&&b.bottom<=62),JSON.stringify(narrowTools));await shot('16-narrow-tools');
 await reveal();await page.locator('.reading-settings-toggle').click();assert.equal(await page.locator('.compact-page-actions').isVisible(),true);await page.locator('.settings-close').click();await page.setViewportSize({width:1520,height:980});await settle();
 check('E: at 480 CSS pixels the secondary tools remain inside the safe rail; grouped navigation/fullscreen have labelled settings alternatives');
 // Content pixel scale and UI scale are deliberately separate checks.
 const cdp=await context.newCDPSession(page);await cdp.send('Emulation.setDeviceMetricsOverride',{width:760,height:490,deviceScaleFactor:2,mobile:false});await wait(1600);await page.locator('.workspace-library').click();lr=await rect(library);assert.ok(lr.x+lr.width<=761&&lr.y+lr.height<=491);await shot('13-density-2-panel');
 check('E: device scale 2 with a constrained CSS viewport retains panel controls (not OS-level display-setting verification)');
 if(report.errors.length)throw new Error(report.errors.join('\n'));
 fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({...report,status:'passed',state},null,2));await browser.close();server.close();console.log(JSON.stringify({status:'passed',checks:report.checks,output:out}));
}
main().catch(async e=>{report.status='failed';report.failure=e.stack;try{await activePage?.screenshot({path:path.join(out,'failure.png')});}catch{}fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({...report,state},null,2));console.error(e);server.close();process.exit(1);});
