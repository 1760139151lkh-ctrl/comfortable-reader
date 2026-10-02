// Real built reader and EPUB; native window operations and persistence are isolated mocks.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'playwright');
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),out=path.join(root,'work','redesign-20261002','window');fs.mkdirSync(out,{recursive:true});
const baseline=JSON.parse(fs.readFileSync(process.env.READER_BASELINE||path.join(root,'个人数据','阅读状态','reader-state.json'),'utf8'));
const book=baseline.books.find(b=>b.title.startsWith('智能优化算法简介'));assert.ok(book);
const state=structuredClone(baseline);state.progress={};state.session={...state.session,paneBookIds:[book.id],paneCount:1,activePane:0,workspace:null,surfaces:null};
const report={tier:'browser_real_ui_mocked_native_window',checks:[],errors:[],calls:[]};
let page,full=false,maximized=false,position=null,minimized=false,destroyed=false,outerSize={width:2700,height:1750};const listeners=[];
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const server=http.createServer((req,res)=>{const file=req.url==='/book.epub'?book.path:path.join(root,'desktop','dist',req.url==='/'?'index.html':decodeURIComponent(req.url.split('?')[0]));if(!fs.existsSync(file)){res.writeHead(404);res.end();return;}res.setHeader('Content-Type',({'.js':'application/javascript','.css':'text/css','.html':'text/html','.json':'application/json','.epub':'application/epub+zip'})[path.extname(file)]||'application/octet-stream');res.end(fs.readFileSync(file));});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));const browser=await chromium.launch({channel:'msedge',headless:true});const context=await browser.newContext({viewport:{width:1280,height:860},deviceScaleFactor:2});page=await context.newPage();page.setDefaultTimeout(12000);page.on('pageerror',e=>report.errors.push(e.message));
 await page.exposeFunction('__windowInvoke',async(command,args={})=>{
  report.calls.push({command,args});
  if(command==='bootstrap'||command==='refresh_library')return{...state,storagePath:'isolated',scan:{rootsScanned:1,missingRoots:0,epubCandidates:baseline.books.length,loadedCandidates:baseline.books.length,booksLoaded:baseline.books.length,issues:[]}};
  if(command==='save_session'){state.session=structuredClone(args.session);return;}
  if(command==='save_progress'){state.progress[args.bookId]=structuredClone(args.progress);return;}
  if(command==='load_location_index')return null;
  if(command==='save_location_index')return;
  if(command==='learning_pack'||command==='epub_content_identity')throw Error('not registered in isolated window test');
  if(command==='learning_runtime_info')return{platform:'test',security_mode:'no-execution'};
  if(command==='plugin:event|listen'){listeners.push(args);return listeners.length;}
  if(command.startsWith('plugin:event|'))return;
  const op=command.split('|')[1];
  if(op==='is_fullscreen')return full;
  if(op==='set_fullscreen'){full=args.value;return;}
  if(op==='is_maximized')return maximized;
  if(op==='unmaximize'){maximized=false;return;}
  if(op==='cursor_position')return{x:1700,y:240};
  if(op==='current_monitor')return{name:'mock 2x',position:{x:0,y:0},size:{width:2560,height:1720},workArea:{position:{x:0,y:0},size:{width:2560,height:1640}},scaleFactor:2};
  if(op==='outer_size')return outerSize;
  if(op==='inner_size')return{width:outerSize.width-16,height:outerSize.height-56};
  if(op==='set_size'){const size=args.value.size;outerSize={width:size.width+16,height:size.height+56};return;}
  if(op==='outer_position')return{x:100,y:100};
  if(op==='inner_position')return{x:108,y:156};
  if(op==='set_position'){position=args.value;return;}
  if(op==='minimize'){minimized=true;return;}
  if(op==='close'){setTimeout(()=>void page.evaluate(async entries=>{for(const e of entries)await window.__windowDispatch(e.handler,{event:e.event,id:1,payload:null});},listeners.filter(e=>e.event==='tauri://close-requested')),0);return;}
  if(op==='destroy'){destroyed=true;return;}
  throw Error('Unexpected mock command '+command);
 });
 await page.addInitScript(()=>{const callbacks=new Map();let id=0;window.__windowDispatch=async(id,event)=>await callbacks.get(id)?.(event);window.__TAURI_INTERNALS__={metadata:{currentWindow:{label:'main'},currentWebview:{label:'main'}},transformCallback:fn=>{callbacks.set(++id,fn);return id;},unregisterCallback:()=>{},invoke:async(command,args={})=>command==='load_book_bytes'?await(await fetch('/book.epub')).arrayBuffer():await window.__windowInvoke(command,args)};window.__TAURI_EVENT_PLUGIN_INTERNALS__={unregisterListener:()=>{}};});
 await page.goto(`http://127.0.0.1:${server.address().port}/`);await page.waitForFunction(()=>document.querySelector('.boot-screen')?.classList.contains('hidden'));await wait(1700);
 const edge=page.locator('.window-edge'),grip=page.locator('.window-grab');assert.equal(await edge.isVisible(),false);
 const cfi=state.progress[book.id].cfi;
 await page.keyboard.press('F11');await wait(300);assert.equal(full,true);assert.equal(await edge.isVisible(),true);assert.equal(await page.locator('.window-grab-line').count(),0);
 assert.equal(await page.locator('.window-grab-hint').evaluate(e=>getComputedStyle(e).opacity),'0');await page.screenshot({path:path.join(out,'01-empty-top-edge.png')});
 let r=await grip.boundingBox();await page.mouse.move(r.x+r.width/2,r.y+15);await page.mouse.down();await page.mouse.move(r.x+r.width/2,r.y+75,{steps:8});assert.equal(await edge.evaluate(e=>e.classList.contains('pulling')),true);assert.equal(full,true);await page.keyboard.press('Escape');await page.mouse.up();assert.equal(full,true);assert.equal(position,null);report.checks.push('Empty top edge has no visible handle; drag previews and Escape cancellation preserve full-screen and book position');
 r=await grip.boundingBox();maximized=true;await page.mouse.move(r.x+r.width/2,r.y+15);await page.mouse.down();await page.mouse.move(r.x+r.width/2,r.y+90,{steps:10});await page.mouse.up();await wait(400);assert.equal(full,false);assert.equal(maximized,false);assert.ok(position);assert.equal(await edge.isVisible(),false);assert.equal(state.progress[book.id].cfi,cfi);report.checks.push('Downward release restores a normal window and uses physical monitor bounds without changing the CFI');
 await page.keyboard.press('F11');await wait(250);await grip.dblclick();await wait(300);assert.equal(full,false);report.checks.push('Double-clicking the blank edge restores the window');
 await page.keyboard.press('F11');await wait(250);const actions=page.locator('.window-edge-actions');await actions.hover();await wait(220);assert.equal(await actions.evaluate(e=>getComputedStyle(e).opacity),'1');await page.screenshot({path:path.join(out,'02-window-actions.png')});await page.locator('[data-window-action="minimize"]').click();assert.equal(minimized,true);
 await page.locator('[data-window-action="restore"]').click();await wait(250);assert.equal(full,false);report.checks.push('Window actions appear at the right edge; minimize and explicit restore call the native operations');
 await page.keyboard.press('F11');await wait(250);await page.locator('.rail-grip').focus();await page.keyboard.press('ArrowDown');await wait(800);const stage=await page.locator('.reader-pane:not(.hidden-pane) .pane-stage').boundingBox(),cap=await edge.boundingBox();assert.ok(stage.y>=cap.y+cap.height);report.checks.push('Moving the reading tools to the bottom still keeps the window edge outside the text');
 await page.setViewportSize({width:480,height:640});await wait(800);const buttons=await page.locator('.window-edge-actions button').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return{left:r.left,right:r.right};}));assert.ok(buttons.every(r=>r.left>=0&&r.right<=480));
 await actions.hover();await wait(200);const closeAt=report.calls.length;await page.locator('[data-window-action="close"]').click();for(let i=0;i<40&&!destroyed;i++)await wait(100);assert.equal(destroyed,true);const closeCalls=report.calls.slice(closeAt).map(c=>c.command);assert.ok(closeCalls.includes('save_session'));assert.ok(closeCalls.indexOf('save_session')<closeCalls.indexOf('plugin:window|destroy'));report.checks.push('At narrow width the controls stay reachable; closing follows the existing save-before-destroy handler');
 assert.deepEqual(report.errors,[]);fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({...report,status:'passed',finalPosition:position},null,2));await browser.close();server.close();console.log(JSON.stringify({status:'passed',checks:report.checks}));
})().catch(async error=>{report.status='failed';report.failure=error.stack;try{await page?.screenshot({path:path.join(out,'failure.png')});}catch{}fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));console.error(error);server.close();process.exit(1);});
