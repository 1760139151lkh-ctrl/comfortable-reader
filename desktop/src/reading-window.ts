import {currentMonitor, cursorPosition, PhysicalPosition, PhysicalSize} from '@tauri-apps/api/window';
import {getCurrentWindow, isDesktop} from './platform';

/** OS-window gestures are separate from book layouts and the movable reading tools. */
export class ReadingWindow {
  private edge = document.createElement('div');
  private grip: HTMLElement;
  private revealTimer = 0;
  private hideTimer = 0;
  private resizeTimer = 0;
  private busy = false;
  private drag: null | {id:number; y:number; fraction:number; ready:boolean} = null;
  constructor(private shell: HTMLElement, private changed: () => void, private error: (message:string) => void) {
    this.edge.className = 'window-edge'; this.edge.hidden = true;
    this.edge.innerHTML = `<div class="window-grab" role="button" tabindex="0" aria-label="向下拖回窗口，双击或按 Enter 也可恢复"><span class="window-grab-hint">松开，回到窗口</span></div>
      <div class="window-edge-actions" aria-label="窗口操作">
        ${isDesktop?'<button type="button" data-window-action="minimize" aria-label="最小化窗口" title="最小化">−</button>':''}
        <button type="button" data-window-action="restore" aria-label="退出全屏，回到窗口" title="回到窗口（F11）">▣</button>
        ${isDesktop?'<button type="button" data-window-action="close" aria-label="保存并关闭舒适阅读书库" title="保存并关闭">×</button>':''}
      </div>`;
    shell.append(this.edge); this.grip=this.edge.querySelector('.window-grab')!;
    this.edge.querySelector('.window-edge-actions')!.addEventListener('pointerenter',()=>{window.clearTimeout(this.hideTimer);this.revealTimer=window.setTimeout(()=>this.edge.classList.add('awake'),160);});
    this.edge.addEventListener('pointerleave',()=>{window.clearTimeout(this.revealTimer);this.hideTimer=window.setTimeout(()=>{if(!this.drag&&!this.edge.contains(document.activeElement))this.edge.classList.remove('awake');},500);});
    this.edge.addEventListener('focusin',()=>this.edge.classList.add('awake'));
    this.edge.addEventListener('click',event=>{
      const action=(event.target as Element).closest<HTMLElement>('[data-window-action]')?.dataset.windowAction;
      if(action==='restore')void this.restore().catch(this.fail);
      if(action==='minimize')void getCurrentWindow().minimize().catch(this.fail);
      if(action==='close')void getCurrentWindow().close().catch(this.fail);
    });
    this.grip.addEventListener('dblclick',()=>void this.restore().catch(this.fail));
    this.grip.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();void this.restore().catch(this.fail);}});
    this.grip.addEventListener('pointerdown',event=>{
      if(event.button||this.busy)return;
      this.drag={id:event.pointerId,y:event.clientY,fraction:event.clientX/innerWidth,ready:false};
      this.grip.setPointerCapture(event.pointerId); event.preventDefault();
    });
    this.grip.addEventListener('pointermove',event=>{
      if(this.drag?.id!==event.pointerId)return;
      this.drag.ready=event.clientY-this.drag.y>=18;
      this.edge.classList.toggle('pulling',this.drag.ready);
      this.edge.querySelector('.window-grab-hint')!.textContent=this.drag.ready?'松开，回到窗口':'向下拖回窗口';
    });
    this.grip.addEventListener('pointerup',event=>{
      if(this.drag?.id!==event.pointerId)return;
      const drag=this.drag;this.cancel();
      if(drag.ready)void this.restore(drag.fraction).catch(this.fail);
    });
    this.grip.addEventListener('pointercancel',()=>this.cancel());
    window.addEventListener('blur',()=>this.cancel());
    window.addEventListener('keydown',event=>{if(event.key==='Escape'&&this.drag){event.preventDefault();event.stopImmediatePropagation();this.cancel();}},true);
    const rail=shell.querySelector<HTMLElement>('.workspace-strip')!;
    const fit=()=>{const right=rail.getBoundingClientRect().right-shell.getBoundingClientRect().left;this.grip.style.left=`${Math.min(right+6,Math.max(0,shell.clientWidth-164))}px`;};
    new ResizeObserver(fit).observe(rail);new ResizeObserver(fit).observe(shell);
    document.addEventListener('fullscreenchange',()=>void this.sync().catch(this.fail));
  }
  private fail=(error:unknown)=>this.error(`窗口操作未完成：${String(error)}。可使用 F11 或系统窗口菜单。`);
  private cancel(){const id=this.drag?.id;this.drag=null;if(id!==undefined&&this.grip.hasPointerCapture(id))this.grip.releasePointerCapture(id);this.edge.classList.remove('pulling');this.edge.querySelector('.window-grab-hint')!.textContent='向下拖回窗口';}
  async initialize(){
    await this.sync();
    if(isDesktop)await getCurrentWindow().onResized(()=>{window.clearTimeout(this.resizeTimer);this.resizeTimer=window.setTimeout(()=>void this.sync().catch(this.fail),80);});
  }
  async sync(){
    const full=await getCurrentWindow().isFullscreen();
    this.edge.hidden=!full;this.shell.dataset.windowFullscreen=String(full);
    for(const button of this.shell.querySelectorAll<HTMLElement>('.fullscreen-toggle,.compact-fullscreen')){
      button.setAttribute('aria-label',full?'退出全屏':'进入全屏');button.setAttribute('aria-pressed',String(full));button.title=full?'退出全屏（F11）':'全屏（F11）';
      if(button.classList.contains('compact-fullscreen'))button.textContent=full?'退出全屏':'全屏';
    }
    this.changed();
  }
  async toggle(){
    if(this.busy)return;this.busy=true;
    try{await getCurrentWindow().setFullscreen(!(await getCurrentWindow().isFullscreen()));await this.sync();}
    finally{this.busy=false;}
  }
  async restore(fraction?:number){
    if(this.busy)return;this.busy=true;
    try{
      const win=getCurrentWindow();
      if(!(await win.isFullscreen()))return;
      // Commit on release: cancellation cannot change window or reader geometry.
      const placement=isDesktop?await Promise.all([fraction!==undefined?cursorPosition():Promise.resolve(null),currentMonitor()]):null;
      await win.setFullscreen(false);
      if(isDesktop&&await win.isMaximized())await win.unmaximize();
      if(placement){
        const [cursor,monitor]=placement;
        await new Promise<void>(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve())));
        let [size,innerSize,outer,inner]=await Promise.all([win.outerSize(),win.innerSize(),win.outerPosition(),win.innerPosition()]);
        if(monitor){
          const area=monitor.workArea,frameWidth=Math.max(0,size.width-innerSize.width),frameHeight=Math.max(0,size.height-innerSize.height);
          // A saved full-screen client size is not a useful restored window size.
          if(size.width>area.size.width||size.height>area.size.height||size.width>=area.size.width*.96&&size.height>=area.size.height*.92){
            await win.setSize(new PhysicalSize(Math.max(480*monitor.scaleFactor,Math.round(area.size.width*.8-frameWidth)),Math.max(420*monitor.scaleFactor,Math.round(area.size.height*.82-frameHeight))));
            await new Promise<void>(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve())));
            size=await win.outerSize();
          }
        }
        let x=cursor?cursor.x-size.width*Math.max(.1,Math.min(.9,fraction!)):outer.x,y=cursor?cursor.y-Math.max(12,(inner.y-outer.y)/2):outer.y;
        if(monitor){const area=monitor.workArea;x=Math.max(area.position.x,Math.min(area.position.x+Math.max(0,area.size.width-size.width),x));y=Math.max(area.position.y,Math.min(area.position.y+Math.max(0,area.size.height-size.height),y));}
        await win.setPosition(new PhysicalPosition(Math.round(x),Math.round(y)));
      }
      await this.sync();
    }finally{this.busy=false;}
  }
}
