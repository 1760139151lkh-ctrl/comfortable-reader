import {addPane,placeNewPane,insertBetween,leaves,defaultLayout,DIVIDER,fitsLayout,measureTree,minimumSize,movePane,normalizeWorkspace,removePane,resizeSplit,type LayoutNode,type Placement,type Rect,type WorkspaceState} from './workspace-layout';
type Hooks={changed:(state:WorkspaceState)=>void;activate:(pane:number)=>void;visible:(panes:number[])=>void;title:(pane:number)=>string;close:(pane:number)=>void;toast:(message:string)=>void;libraryBook:(id:string)=>{pane:number;existing:boolean;title:string;author:string}|null;open:(id:string,pane:number,tree:LayoutNode)=>void};
const escape=(s:string)=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
export class ReadingWorkspace{
  state:WorkspaceState={version:1,tree:null,manual:false,focused:null};
  ids:number[]=[];active=0;visiblePanes:number[]=[];compact=false;
  private dividers=document.createElement('div');private preview=document.createElement('div');private message=document.createElement('div');
  private panel:HTMLElement;private toggle:HTMLButtonElement;private resizeObserver:ResizeObserver;
  private dragging:null|{kind:'book';pane:number;pointer:number;startX:number;startY:number;started:boolean;candidate:LayoutNode|null;target:number|null;placement:Placement|null;bookId:string|null;label:string;author:string}|{kind:'split';id:string;pointer:number;before:LayoutNode;rect:Rect;axis:'x'|'y';minimum:number;maximum:number}=null;
  private ghost=document.createElement('div');private dropSignature='';private lastPoint={x:0,y:0};suppressClickUntil=0;
  get busy(){return Boolean(this.dragging);}
  private frame=0;private notification='';private geometry=measureTree(null,{x:0,y:0,width:0,height:0});
  constructor(private root:HTMLElement,private app:HTMLElement,private hooks:Hooks){
    this.panel=app.querySelector('.workspace-panel')!;this.toggle=app.querySelector('.workspace-toggle')!;
    this.dividers.className='workspace-dividers';this.preview.className='workspace-drop-preview';this.preview.hidden=true;this.message.className='workspace-live';this.message.setAttribute('role','status');this.message.setAttribute('aria-live','polite');root.append(this.dividers,this.preview,this.message);this.ghost.className='book-pickup';this.ghost.hidden=true;app.append(this.ghost);
    app.addEventListener('pointerdown',e=>{const handle=(e.target as Element).closest<HTMLElement>('.book-main');if(!handle||e.button)return;const id=handle.closest<HTMLElement>('[data-book-id]')?.dataset.bookId,book=id?hooks.libraryBook(id):null;if(!id||!book)return;this.dragging={kind:'book',pane:book.pane,pointer:e.pointerId,startX:e.clientX,startY:e.clientY,started:false,candidate:null,target:null,placement:null,bookId:book.existing?null:id,label:book.title,author:book.author};handle.setPointerCapture(e.pointerId);});
    root.addEventListener('pointerdown',e=>this.pointerDown(e));
    window.addEventListener('pointermove',e=>this.pointerMove(e));window.addEventListener('pointerup',e=>this.pointerEnd(e));window.addEventListener('pointercancel',()=>this.cancel());window.addEventListener('blur',()=>this.cancel());
    window.addEventListener('keydown',e=>{if(e.key==='Escape'&&this.dragging){e.preventDefault();e.stopImmediatePropagation();this.cancel();}},true);
    this.dividers.addEventListener('keydown',e=>this.dividerKey(e));
    this.toggle.addEventListener('click',()=>this.showPanel(true));
    app.querySelector('.library-all')?.addEventListener('click',()=>this.showPanel(false));
    app.querySelector('.search-box input')?.addEventListener('input',()=>{if(this.panel.classList.contains('visible'))this.renderPanel();});
    this.panel.addEventListener('pointerdown',e=>{
      const handle=(e.target as Element).closest<HTMLElement>('[data-workspace-select]');if(!handle||e.button)return;
      const pane=Number(handle.dataset.workspaceSelect);this.dragging={kind:'book',pane,pointer:e.pointerId,startX:e.clientX,startY:e.clientY,started:false,candidate:null,target:null,placement:null,bookId:null,label:hooks.title(pane),author:''};handle.setPointerCapture(e.pointerId);
    });
    app.querySelector('.workspace-focus')?.addEventListener('click',()=>this.focus(this.active));
    this.panel.addEventListener('click',e=>{const button=(e.target as Element).closest<HTMLButtonElement>('button');if(!button||performance.now()<this.suppressClickUntil)return;if(button.dataset.workspaceSelect!==undefined){hooks.activate(Number(button.dataset.workspaceSelect));const drawer=this.panel.closest<HTMLElement>('.library-drawer')!;if(drawer.dataset.docked==='float')drawer.querySelector<HTMLButtonElement>('.drawer-close')?.click();}if(button.dataset.workspaceClose!==undefined)hooks.close(Number(button.dataset.workspaceClose));if(button.dataset.place){const target=Number((this.panel.querySelector('.workspace-target') as HTMLSelectElement).value);this.move(this.active,target,button.dataset.place as Placement);}});
    this.resizeObserver=new ResizeObserver(()=>{if(!this.frame)this.frame=requestAnimationFrame(()=>{this.frame=0;this.render();});});this.resizeObserver.observe(root);
  }
  private rect():Rect{return {x:0,y:0,width:this.root.clientWidth,height:this.root.clientHeight};}
  sync(raw:unknown,ids:number[],active:number){
    const box=this.rect();this.ids=ids;this.active=active;this.state=normalizeWorkspace(raw,ids,box.width,box.height);this.render();
  }
  setActive(active:number){this.active=active;if(this.state.focused!==null&&this.state.focused!==active){this.state.focused=active;this.hooks.changed(structuredClone(this.state));}this.render();}
  added(pane:number,target=this.active,placement?:LayoutNode){const box=this.rect();this.ids=[...new Set([...this.ids,pane])];this.state.tree=placement??(this.state.manual?addPane(this.state.tree,pane,target===pane?this.previousTarget(pane):target,box):defaultLayout(this.ids,box.width,box.height));if(placement)this.state.manual=true;this.active=pane;this.state.focused=null;this.commit();}
  private previousTarget(pane:number){return this.visiblePanes.find(p=>p!==pane)??this.ids.find(p=>p!==pane)??pane;}
  removed(pane:number){this.ids=this.ids.filter(p=>p!==pane);this.state.tree=removePane(this.state.tree,pane);if(this.state.focused===pane)this.state.focused=null;if(this.active===pane)this.active=this.ids[0]??0;this.commit();}
  focus(pane:number){if(this.ids.length<2)return;this.active=pane;this.state.focused=this.state.focused===pane?null:pane;this.showPanel(false);this.commit();this.hooks.activate(pane);}
  unfocus(){if(this.state.focused!==null){this.state.focused=null;this.commit();return true;}return false;}
  organize(){this.state.tree=defaultLayout(this.ids,this.root.clientWidth,this.root.clientHeight);this.state.manual=false;this.state.focused=null;this.showPanel(false);this.commit();}
  private commit(){this.hooks.changed(structuredClone(this.state));this.render();}
  showPanel(visible:boolean,returnFocus=true){
    const was=this.panel.classList.contains('visible'),drawer=this.panel.closest<HTMLElement>('.library-drawer')!;
    this.panel.classList.toggle('visible',visible);this.panel.inert=!visible;this.panel.setAttribute('aria-hidden',String(!visible));drawer.dataset.shelfView=visible?'open':'all';
    this.toggle.setAttribute('aria-pressed',String(visible));this.toggle.setAttribute('aria-expanded',String(visible));drawer.querySelector('.library-all')?.setAttribute('aria-pressed',String(!visible));
    const list=drawer.querySelector<HTMLElement>('.library-list')!;list.hidden=visible;list.inert=visible;list.setAttribute('aria-hidden',String(visible));
    drawer.querySelector<HTMLInputElement>('.search-box input')!.placeholder=visible?'查找正在读的书':'搜索书名或作者';
    if(visible)this.renderPanel();else if(was&&returnFocus&&this.panel.contains(document.activeElement))drawer.querySelector<HTMLButtonElement>('.library-all')?.focus({preventScroll:true});
  }
  private renderPanel(){
    const list=this.panel.querySelector<HTMLElement>('.workspace-books')!,scroll=list.scrollTop,query=this.app.querySelector<HTMLInputElement>('.search-box input')!.value.trim().toLocaleLowerCase();
    const focused=list.contains(document.activeElement)?document.activeElement as HTMLElement:null;
    const focusSelector=focused?.dataset.workspaceClose!==undefined?`[data-workspace-close="${focused.dataset.workspaceClose}"]`:focused?.dataset.workspaceSelect!==undefined?`[data-workspace-select="${focused.dataset.workspaceSelect}"]`:null;
    list.innerHTML=this.ids.filter(p=>this.hooks.title(p).toLocaleLowerCase().includes(query)).map(p=>`<div class="workspace-book-row"><button data-workspace-select="${p}" aria-current="${p===this.active}" title="${escape(this.hooks.title(p))}"><span>${p===this.active?'•':''}</span><span class="workspace-book-title">${escape(this.hooks.title(p))}</span></button><button data-workspace-close="${p}" aria-label="关闭 ${escape(this.hooks.title(p))}" title="关闭阅读区域，书籍保留">×</button></div>`).join('')||'<p class="navigation-empty">没有匹配的已打开书籍</p>';list.scrollTop=scroll;
    const target=this.panel.querySelector<HTMLSelectElement>('.workspace-target')!,value=target.value;target.innerHTML=this.ids.filter(p=>p!==this.active).map(p=>`<option value="${p}">${escape(this.hooks.title(p))}</option>`).join('');if([...target.options].some(o=>o.value===value))target.value=value;
    this.panel.querySelector<HTMLElement>('.workspace-move-controls')!.hidden=this.ids.length<2;
    this.panel.querySelector<HTMLElement>('.workspace-current-title')!.textContent=this.hooks.title(this.active);
    if(focusSelector)(list.querySelector<HTMLElement>(focusSelector)??list.querySelector<HTMLElement>('[data-workspace-select]'))?.focus({preventScroll:true});
  }
  render(){
    const box=this.rect();if(box.width<20||box.height<20)return;
    this.compact=this.ids.length>1&&this.state.focused===null&&!fitsLayout(this.state.tree,box);
    const displayTree=this.ids.length&&(this.state.focused!==null||this.compact)?{kind:'book' as const,pane:this.ids.includes(this.active)?this.active:this.ids[0]}:this.state.tree;
    this.geometry=measureTree(displayTree,box);this.visiblePanes=[...this.geometry.books.keys()];
    this.root.dataset.visibleBooks=String(this.visiblePanes.length);this.root.dataset.compact=String(this.compact);
    for(const pane of this.root.querySelectorAll<HTMLElement>('.reader-pane')){
      const id=Number(pane.dataset.paneIndex),r=this.geometry.books.get(id);const show=Boolean(r)||!this.ids.length&&id===0;
      pane.classList.toggle('hidden-pane',!show);pane.inert=!show;pane.setAttribute('aria-hidden',String(!show));
      if(show){const pos=r??box;Object.assign(pane.style,{left:`${pos.x}px`,top:`${pos.y}px`,width:`${pos.width}px`,height:`${pos.height}px`});}
      pane.querySelector('.pane-focus')?.setAttribute('aria-label',this.state.focused===id?'返回对照布局':'临时专注这本书');
    }
    const existing=new Map([...this.dividers.children].map(el=>[(el as HTMLElement).dataset.splitId!,el as HTMLElement]));
    for(const [id,{node,divider}] of this.geometry.splits){
      let el=existing.get(id);if(!el){el=document.createElement('div');el.className='workspace-divider';el.dataset.splitId=id;el.tabIndex=0;el.setAttribute('role','separator');el.setAttribute('aria-label','调整阅读区域比例（方向键微调，Home 均分）');this.dividers.append(el);}existing.delete(id);
      Object.assign(el.style,{left:`${divider.x}px`,top:`${divider.y}px`,width:`${divider.width}px`,height:`${divider.height}px`});el.dataset.axis=node.axis;el.setAttribute('aria-orientation',node.axis==='x'?'vertical':'horizontal');el.setAttribute('aria-valuenow',String(Math.round(node.ratio*100)));el.setAttribute('aria-valuemin','8');el.setAttribute('aria-valuemax','92');
    }
    existing.forEach(el=>el.remove());
    this.toggle.querySelector('span')!.textContent=`正在读 · ${this.ids.length}`;
    this.toggle.hidden=this.ids.length===0;
    const focus=this.app.querySelector<HTMLButtonElement>('.workspace-focus')!;focus.hidden=this.ids.length<2;focus.textContent=this.state.focused===null?'专注一本':'返回对照';focus.setAttribute('aria-pressed',String(this.state.focused!==null));
    this.app.dataset.focusedBook=String(this.state.focused!==null);
    const status=this.app.querySelector<HTMLElement>('.workspace-space-status')!;status.hidden=!this.compact;status.textContent='窗口变宽后恢复并排';status.title='其他书保持打开；点书名即可切换。';
    if(this.panel.classList.contains('visible')&&!this.busy)this.renderPanel();
    const notify=this.visiblePanes.join(',');if(notify!==this.notification){this.notification=notify;this.hooks.visible(this.visiblePanes);}
  }
  private position(element:HTMLElement,r:Rect){Object.assign(element.style,{left:`${r.x}px`,top:`${r.y}px`,width:`${r.width}px`,height:`${r.height}px`});}
  private pointerDown(e:PointerEvent){
    if(e.button!==0)return;const target=e.target as Element,divider=target.closest<HTMLElement>('.workspace-divider'),handle=target.closest<HTMLElement>('.pane-drag-handle,.pane-book-meta');
    if(divider){const item=this.geometry.splits.get(divider.dataset.splitId!);if(!item||!this.state.tree)return;const extent=(item.node.axis==='x'?item.rect.width:item.rect.height)-DIVIDER,a=minimumSize(item.node.first),b=minimumSize(item.node.second);this.dragging={kind:'split',id:item.node.id,pointer:e.pointerId,before:structuredClone(this.state.tree),rect:item.rect,axis:item.node.axis,minimum:(item.node.axis==='x'?a.width:a.height)/extent,maximum:1-(item.node.axis==='x'?b.width:b.height)/extent};}
    else if(handle&&this.ids.length>1&&this.state.focused===null&&!this.compact){const pane=Number(handle.closest<HTMLElement>('.reader-pane')!.dataset.paneIndex);this.dragging={kind:'book',pane,pointer:e.pointerId,startX:e.clientX,startY:e.clientY,started:false,candidate:null,target:null,placement:null,bookId:null,label:this.hooks.title(pane),author:''};}
    else return;
    e.preventDefault();(e.target as HTMLElement).setPointerCapture(e.pointerId);this.root.classList.add('layout-interacting');this.showPanel(false,false);
  }
  private pointerMove(e:PointerEvent){
    const d=this.dragging;if(!d||e.pointerId!==d.pointer)return;
    const root=this.root.getBoundingClientRect(),x=e.clientX-root.left,y=e.clientY-root.top;
    if(d.kind==='split'){
      const extent=(d.axis==='x'?d.rect.width:d.rect.height)-DIVIDER,offset=(d.axis==='x'?x-d.rect.x:y-d.rect.y)-DIVIDER/2;
      this.state.tree=resizeSplit(this.state.tree!,d.id,Math.max(d.minimum,Math.min(d.maximum,offset/extent)));this.render();return;
    }
    if(!d.started&&Math.hypot(e.clientX-d.startX,e.clientY-d.startY)<7)return;
    if(!d.started){d.started=true;this.root.classList.add('layout-interacting');this.app.classList.add('book-dragging');this.dropSignature='';this.ghost.innerHTML=`<span class="pickup-cover">${escape(d.label.slice(0,1))}</span><div><strong>${escape(d.label)}</strong><small>${escape(d.author||'移动这本书')}</small></div>`;}
    this.ghost.hidden=false;this.ghost.style.left=`${Math.min(innerWidth-250,Math.max(8,e.clientX+18))}px`;this.ghost.style.top=`${Math.min(innerHeight-90,e.clientY+18)}px`;
    if(x<0||y<0||x>root.width||y>root.height){this.preview.hidden=true;d.candidate=null;this.dropSignature='';return;}
    let candidate:LayoutNode|null=null,targetPane:number|null=null,placement:Placement='right',signature='';
    const between=[...this.geometry.splits.values()].find(({node,divider:r})=>node.axis==='x'?Math.abs(x-r.x-r.width/2)<18&&y>=r.y&&y<=r.y+r.height:Math.abs(y-r.y-r.height/2)<18&&x>=r.x&&x<=r.x+r.width);
    if(between&&this.state.tree){
      const result=insertBetween(this.state.tree,d.pane,between.node.id);
      if(leaves(result).includes(d.pane)){candidate=result;signature=`between:${between.node.id}`;}
    }
    if(!candidate){
      const target=[...this.geometry.books].find(([p,r])=>(p!==d.pane||Boolean(d.bookId))&&x>=r.x&&x<=r.x+r.width&&y>=r.y&&y<=r.y+r.height);
      if(!target){if(!this.ids.length&&d.bookId){candidate={kind:'book',pane:d.pane};signature='empty';}else{this.preview.hidden=true;d.candidate=null;this.dropSignature='';return;}}
      else{
        const [pane,r]=target,nx=(x-r.x)/r.width,ny=(y-r.y)/r.height;
        targetPane=pane;
        const edges:[Placement,number][]=[['left',nx],['right',1-nx],['top',ny],['bottom',1-ny]];edges.sort((a,b)=>a[1]-b[1]);
        placement=edges[0][1]<.25?edges[0][0]:d.bookId?'right':'swap';
        signature=`${pane}:${placement}`;
        // Hysteresis near a boundary: a few pixels of hand movement cannot flicker the layout.
        if(d.candidate&&signature!==this.dropSignature&&Math.hypot(x-this.lastPoint.x,y-this.lastPoint.y)<18)return;
        if(signature===this.dropSignature)return;
        candidate=d.bookId?placeNewPane(this.state.tree,d.pane,pane,placement):movePane(this.state.tree!,d.pane,pane,placement);
      }
    }
    if(signature===this.dropSignature)return;
    this.dropSignature=signature;this.lastPoint={x,y};d.candidate=candidate;d.target=targetPane;d.placement=placement;
    const fits=fitsLayout(candidate,this.rect());
    this.preview.hidden=false;this.preview.classList.toggle('compact-preview',!fits);this.position(this.preview,this.rect());
    const projection=measureTree(fits?candidate:{kind:'book',pane:d.pane},this.rect());
    this.preview.innerHTML=[...projection.books].map(([pane,r])=>`<div class="workspace-projected-book ${pane===d.pane?'incoming':''}" style="left:${r.x}px;top:${r.y}px;width:${r.width}px;height:${r.height}px"><span><strong>${escape(pane===d.pane?d.label:this.hooks.title(pane))}</strong><small>${pane===d.pane?(fits?(signature.startsWith('between')?'插入两书之间':d.bookId?'加入阅读':'移动到这里'):'空间较窄，先读这本；其他书保持打开'):'保留阅读位置'}</small></span></div>`).join('')+`<div class="placement-caption">松开放置 · Esc 取消${fits?'':' · 放大窗口可恢复并排'}</div>`;
  }
  private pointerEnd(e:PointerEvent){
    const d=this.dragging;if(!d||e.pointerId!==d.pointer)return;
    this.dragging=null;this.cleanupDrag();
    if(d.kind==='split'){this.state.manual=true;this.commit();return;}
    if(d.started)this.suppressClickUntil=performance.now()+400;
    if(!d.candidate)return;
    if(d.bookId)this.hooks.open(d.bookId,d.pane,d.candidate);
    else{this.state.tree=d.candidate;this.state.manual=true;this.state.focused=null;this.commit();this.hooks.activate(d.pane);}
    this.message.textContent=`已放置 ${d.label}`;
  }
  private cleanupDrag(){this.preview.hidden=true;this.ghost.hidden=true;this.root.classList.remove('layout-interacting');this.app.classList.remove('book-dragging');}
  cancel(){if(this.dragging?.kind==='split')this.state.tree=this.dragging.before;if(this.dragging?.kind==='book'&&this.dragging.started)this.suppressClickUntil=performance.now()+400;this.dragging=null;this.cleanupDrag();this.render();}
  private dividerKey(e:KeyboardEvent){const el=e.target as HTMLElement,item=this.geometry.splits.get(el.dataset.splitId!);if(!item)return;const back=item.node.axis==='x'?'ArrowLeft':'ArrowUp',forward=item.node.axis==='x'?'ArrowRight':'ArrowDown';if(![back,forward,'Home'].includes(e.key))return;e.preventDefault();e.stopPropagation();const ratio=e.key==='Home'?.5:item.node.ratio+(e.key===forward?1:-1)*(e.shiftKey?.1:.02),candidate=resizeSplit(this.state.tree!,item.node.id,ratio);if(fitsLayout(candidate,this.rect())){this.state.tree=candidate;this.state.manual=true;this.commit();}}
  move(source:number,target:number,placement:Placement){if(!this.state.tree||source===target)return;const candidate=movePane(this.state.tree,source,target,placement);if(!fitsLayout(candidate,this.rect())){this.hooks.toast('当前空间放不下这个排列，请先放大窗口或使用重新整理。');return;}this.state.tree=candidate;this.state.manual=true;this.state.focused=null;this.showPanel(false);this.commit();}
}
