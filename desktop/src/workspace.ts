import {addPane,defaultLayout,DIVIDER,fitsLayout,measureTree,minimumSize,movePane,normalizeWorkspace,removePane,resizeSplit,type LayoutNode,type Placement,type Rect,type WorkspaceState} from './workspace-layout';
type Hooks={changed:(state:WorkspaceState)=>void;activate:(pane:number)=>void;visible:(panes:number[])=>void;title:(pane:number)=>string;close:(pane:number)=>void;toast:(message:string)=>void};
const directions:Record<Placement,string>={left:'左侧',right:'右侧',top:'上方',bottom:'下方',swap:'交换位置'};
const escape=(s:string)=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
export class ReadingWorkspace{
  state:WorkspaceState={version:1,tree:null,manual:false,focused:null};
  ids:number[]=[];active=0;visiblePanes:number[]=[];compact=false;
  private dividers=document.createElement('div');private preview=document.createElement('div');private message=document.createElement('div');
  private tabs:HTMLElement;private panel:HTMLElement;private toggle:HTMLButtonElement;private resizeObserver:ResizeObserver;
  private dragging:null|{kind:'book';pane:number;pointer:number;startX:number;startY:number;started:boolean;candidate:LayoutNode|null;target:number|null;placement:Placement|null}|{kind:'split';id:string;pointer:number;before:LayoutNode;rect:Rect;axis:'x'|'y';minimum:number;maximum:number}=null;
  private frame=0;private notification='';private geometry=measureTree(null,{x:0,y:0,width:0,height:0});
  constructor(private root:HTMLElement,private app:HTMLElement,private hooks:Hooks){
    this.tabs=app.querySelector('.workspace-tabs')!;this.panel=app.querySelector('.workspace-panel')!;this.toggle=app.querySelector('.workspace-toggle')!;
    this.dividers.className='workspace-dividers';this.preview.className='workspace-drop-preview';this.preview.hidden=true;this.message.className='workspace-live';this.message.setAttribute('role','status');this.message.setAttribute('aria-live','polite');root.append(this.dividers,this.preview,this.message);
    root.addEventListener('pointerdown',e=>this.pointerDown(e));
    window.addEventListener('pointermove',e=>this.pointerMove(e));window.addEventListener('pointerup',e=>this.pointerEnd(e));window.addEventListener('pointercancel',()=>this.cancel());
    window.addEventListener('keydown',e=>{if(e.key==='Escape'&&this.dragging){e.preventDefault();e.stopImmediatePropagation();this.cancel();}},true);
    this.dividers.addEventListener('keydown',e=>this.dividerKey(e));
    this.toggle.addEventListener('click',()=>this.showPanel(!this.panel.classList.contains('visible')));
    app.querySelector('.workspace-panel-close')?.addEventListener('click',()=>this.showPanel(false));
    app.querySelector('.workspace-organize')?.addEventListener('click',()=>this.organize());
    app.querySelector('.workspace-focus')?.addEventListener('click',()=>this.focus(this.active));
    this.panel.addEventListener('click',e=>{const button=(e.target as Element).closest<HTMLButtonElement>('button');if(!button)return;if(button.dataset.workspaceSelect!==undefined){hooks.activate(Number(button.dataset.workspaceSelect));this.showPanel(false);}if(button.dataset.workspaceClose!==undefined)hooks.close(Number(button.dataset.workspaceClose));if(button.dataset.place){const target=Number((this.panel.querySelector('.workspace-target') as HTMLSelectElement).value);this.move(this.active,target,button.dataset.place as Placement);}});
    this.tabs.addEventListener('click',e=>{const button=(e.target as Element).closest<HTMLElement>('[data-workspace-select]');if(button)hooks.activate(Number(button.dataset.workspaceSelect));});
    document.addEventListener('pointerdown',e=>{if(!(e.target as Element).closest('.workspace-panel,.workspace-toggle'))this.showPanel(false,false);});
    this.resizeObserver=new ResizeObserver(()=>{if(!this.frame)this.frame=requestAnimationFrame(()=>{this.frame=0;this.render();});});this.resizeObserver.observe(root);
  }
  private rect():Rect{return {x:0,y:0,width:this.root.clientWidth,height:this.root.clientHeight};}
  sync(raw:unknown,ids:number[],active:number){
    const box=this.rect();this.ids=ids;this.active=active;this.state=normalizeWorkspace(raw,ids,box.width,box.height);this.render();
  }
  setActive(active:number){this.active=active;if(this.state.focused!==null&&this.state.focused!==active){this.state.focused=active;this.hooks.changed(structuredClone(this.state));}this.render();}
  added(pane:number,target=this.active){const box=this.rect();this.ids=[...new Set([...this.ids,pane])];this.state.tree=this.state.manual?addPane(this.state.tree,pane,target===pane?this.previousTarget(pane):target,box):defaultLayout(this.ids,box.width,box.height);this.active=pane;if(this.state.focused!==null)this.state.focused=pane;this.commit();}
  private previousTarget(pane:number){return this.visiblePanes.find(p=>p!==pane)??this.ids.find(p=>p!==pane)??pane;}
  removed(pane:number){this.ids=this.ids.filter(p=>p!==pane);this.state.tree=removePane(this.state.tree,pane);if(this.state.focused===pane)this.state.focused=null;if(this.active===pane)this.active=this.ids[0]??0;this.commit();}
  focus(pane:number){if(this.ids.length<2)return;this.active=pane;this.state.focused=this.state.focused===pane?null:pane;this.showPanel(false);this.commit();this.hooks.activate(pane);}
  unfocus(){if(this.state.focused!==null){this.state.focused=null;this.commit();return true;}return false;}
  organize(){this.state.tree=defaultLayout(this.ids,this.root.clientWidth,this.root.clientHeight);this.state.manual=false;this.state.focused=null;this.showPanel(false);this.commit();}
  private commit(){this.hooks.changed(structuredClone(this.state));this.render();}
  showPanel(visible:boolean,returnFocus=true){const was=this.panel.classList.contains('visible');this.panel.classList.toggle('visible',visible);this.panel.inert=!visible;this.panel.setAttribute('aria-hidden',String(!visible));this.toggle.setAttribute('aria-expanded',String(visible));if(visible){this.renderPanel();if(!was)this.panel.querySelector<HTMLButtonElement>('.workspace-panel-close')?.focus({preventScroll:true});}else if(was&&returnFocus&&this.panel.contains(document.activeElement))this.toggle.focus({preventScroll:true});}
  private renderPanel(){
    this.panel.querySelector('.workspace-books')!.innerHTML=this.ids.map(p=>`<div class="workspace-book-row"><button data-workspace-select="${p}" aria-current="${p===this.active}"><span>${p===this.active?'•':''}</span>${escape(this.hooks.title(p))}</button><button data-workspace-close="${p}" aria-label="关闭 ${escape(this.hooks.title(p))}" title="关闭阅读区域，书籍保留">×</button></div>`).join('');
    const target=this.panel.querySelector<HTMLSelectElement>('.workspace-target')!,value=target.value;target.innerHTML=this.ids.filter(p=>p!==this.active).map(p=>`<option value="${p}">${escape(this.hooks.title(p))}</option>`).join('');if([...target.options].some(o=>o.value===value))target.value=value;
    this.panel.querySelector<HTMLElement>('.workspace-move-controls')!.hidden=this.ids.length<2;
    this.panel.querySelector<HTMLElement>('.workspace-current-title')!.textContent=this.hooks.title(this.active);
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
    const tabSignature=JSON.stringify([this.ids,this.active,this.compact,this.state.focused]);
    if(this.tabs.dataset.signature!==tabSignature){this.tabs.dataset.signature=tabSignature;this.tabs.innerHTML=(this.compact||this.state.focused!==null)?this.ids.map(p=>`<button data-workspace-select="${p}" aria-pressed="${p===this.active}" title="${escape(this.hooks.title(p))}">${escape(this.hooks.title(p))}</button>`).join(''):'';}
    this.toggle.querySelector('span')!.textContent=`${this.ids.length>1?'对照':'书籍'} · ${this.ids.length}`;
    const focus=this.app.querySelector<HTMLButtonElement>('.workspace-focus')!;focus.hidden=this.ids.length<2;focus.textContent=this.state.focused===null?'专注一本':'返回对照';focus.setAttribute('aria-pressed',String(this.state.focused!==null));
    const status=this.app.querySelector<HTMLElement>('.workspace-space-status')!;status.hidden=!this.compact;status.textContent='空间不足，切换阅读';status.title='其他书保持打开；窗口变宽后自动恢复原布局。也可在对照菜单中重新整理。';
    if(this.panel.classList.contains('visible')&&!this.panel.contains(document.activeElement))this.renderPanel();
    const notify=this.visiblePanes.join(',');if(notify!==this.notification){this.notification=notify;this.hooks.visible(this.visiblePanes);}
  }
  private position(element:HTMLElement,r:Rect){Object.assign(element.style,{left:`${r.x}px`,top:`${r.y}px`,width:`${r.width}px`,height:`${r.height}px`});}
  private pointerDown(e:PointerEvent){
    if(e.button!==0)return;const target=e.target as Element,divider=target.closest<HTMLElement>('.workspace-divider'),handle=target.closest<HTMLElement>('.pane-drag-handle');
    if(divider){const item=this.geometry.splits.get(divider.dataset.splitId!);if(!item||!this.state.tree)return;const extent=(item.node.axis==='x'?item.rect.width:item.rect.height)-DIVIDER,a=minimumSize(item.node.first),b=minimumSize(item.node.second);this.dragging={kind:'split',id:item.node.id,pointer:e.pointerId,before:structuredClone(this.state.tree),rect:item.rect,axis:item.node.axis,minimum:(item.node.axis==='x'?a.width:a.height)/extent,maximum:1-(item.node.axis==='x'?b.width:b.height)/extent};}
    else if(handle&&this.ids.length>1&&this.state.focused===null&&!this.compact){const pane=Number(handle.closest<HTMLElement>('.reader-pane')!.dataset.paneIndex);this.dragging={kind:'book',pane,pointer:e.pointerId,startX:e.clientX,startY:e.clientY,started:false,candidate:null,target:null,placement:null};this.hooks.activate(pane);}
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
    if(!d.started&&Math.hypot(e.clientX-d.startX,e.clientY-d.startY)<6)return;d.started=true;
    const target=[...this.geometry.books].find(([p,r])=>p!==d.pane&&x>=r.x&&x<=r.x+r.width&&y>=r.y&&y<=r.y+r.height);
    if(!target){this.preview.hidden=true;d.candidate=null;return;}
    const [pane,r]=target,nx=(x-r.x)/r.width,ny=(y-r.y)/r.height;
    const edges:[Placement,number][]=[['left',nx],['right',1-nx],['top',ny],['bottom',1-ny]];edges.sort((a,b)=>a[1]-b[1]);const placement=edges[0][1]<.25?edges[0][0]:'swap';
    const candidate=movePane(this.state.tree!,d.pane,pane,placement),fits=fitsLayout(candidate,this.rect());d.candidate=fits?candidate:null;d.target=pane;d.placement=placement;
    const preview=placement==='swap'?r:measureTree(candidate,this.rect()).books.get(d.pane)!;
    this.position(this.preview,preview);this.preview.hidden=false;this.preview.classList.toggle('invalid',!fits);this.preview.textContent=fits?`${directions[placement]} · 松开放置`:'这里太窄，换个位置或放大窗口';
  }
  private pointerEnd(e:PointerEvent){const d=this.dragging;if(!d||e.pointerId!==d.pointer)return;if(d.kind==='split'){this.state.manual=true;this.commit();}else if(d.candidate){this.state.tree=d.candidate;this.state.manual=true;this.commit();this.message.textContent=`已${directions[d.placement!]}放置 ${this.hooks.title(d.pane)}`;}this.dragging=null;this.preview.hidden=true;this.root.classList.remove('layout-interacting');}
  cancel(){if(this.dragging?.kind==='split')this.state.tree=this.dragging.before;this.dragging=null;this.preview.hidden=true;this.root.classList.remove('layout-interacting');this.render();}
  private dividerKey(e:KeyboardEvent){const el=e.target as HTMLElement,item=this.geometry.splits.get(el.dataset.splitId!);if(!item)return;const back=item.node.axis==='x'?'ArrowLeft':'ArrowUp',forward=item.node.axis==='x'?'ArrowRight':'ArrowDown';if(![back,forward,'Home'].includes(e.key))return;e.preventDefault();e.stopPropagation();const ratio=e.key==='Home'?.5:item.node.ratio+(e.key===forward?1:-1)*(e.shiftKey?.1:.02),candidate=resizeSplit(this.state.tree!,item.node.id,ratio);if(fitsLayout(candidate,this.rect())){this.state.tree=candidate;this.state.manual=true;this.commit();}}
  move(source:number,target:number,placement:Placement){if(!this.state.tree||source===target)return;const candidate=movePane(this.state.tree,source,target,placement);if(!fitsLayout(candidate,this.rect())){this.hooks.toast('当前空间放不下这个排列，请先放大窗口或使用重新整理。');return;}this.state.tree=candidate;this.state.manual=true;this.state.focused=null;this.showPanel(false);this.commit();}
}
