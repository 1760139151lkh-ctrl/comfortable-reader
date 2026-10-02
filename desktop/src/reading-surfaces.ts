type Dock = 'float' | 'left' | 'right';
type Box = {x: number; y: number; width: number; height: number};
type PanelState = Box & {dock: Dock; open?: boolean};
export type SurfaceState = {version: 2; toolbarEdge: 'top' | 'bottom'; uiScale: number; panels: Record<string, PanelState>};
type Panel = {id: string; element: HTMLElement; preferred: 'left' | 'right'; initial: Box};
const clamp = (n: number, a: number, b: number) => Math.max(a, Math.min(b, n));
const defaults = (): SurfaceState => ({version: 2, toolbarEdge: 'top', uiScale: 1, panels: {}});

/** Panel geometry is separate from books, EPUB anchors, and temporary viewport constraints. */
export class ReadingSurfaces {
  state = defaults();
  private panels = new Map<string, Panel>();
  private preview = document.createElement('div');
  private veil = document.createElement('div');
  private operation: null | {id: string; pointer: number; x: number; y: number; before: PanelState; box: Box; edge: string; dock: Dock | null; started: boolean} = null;
  private railDrag: null | {pointer: number; edge: 'top' | 'bottom'; started: boolean; y: number} = null;
  private ready = false;
  private frame = 0;
  private chromeTimer = 0;
  private revealTimer = 0;
  private overRail = false;
  private keyboardFocus = false;
  constructor(private shell: HTMLElement, private grid: HTMLElement, private save: (state: SurfaceState) => void, private quiet: () => void = () => {}) {
    this.preview.className = 'surface-dock-preview'; this.preview.hidden = true;
    this.veil.className = 'surface-gesture-veil'; this.veil.hidden = true;
    shell.append(this.preview, this.veil);
    window.addEventListener('pointermove', event => this.move(event));
    window.addEventListener('pointerup', event => this.end(event));
    window.addEventListener('pointercancel', () => this.cancel());
    window.addEventListener('blur', () => this.cancel());
    window.addEventListener('keydown', event => {
      if (event.key === 'Escape' && (this.operation || this.railDrag)) { event.preventDefault(); event.stopImmediatePropagation(); this.cancel(); }
    }, true);
    new ResizeObserver(() => this.schedule()).observe(shell);
    const rail = shell.querySelector<HTMLElement>('.workspace-strip')!;
    rail.addEventListener('pointerenter', () => {
      this.overRail = true; window.clearTimeout(this.chromeTimer);
      this.revealTimer = window.setTimeout(() => this.wakeChrome(), 180);
    });
    rail.addEventListener('pointerleave', () => {
      this.overRail = false; window.clearTimeout(this.revealTimer); this.restChrome();
    });
    document.addEventListener('keydown', event => { if (event.key === 'Tab') this.keyboardFocus = true; }, true);
    document.addEventListener('pointerdown', () => { this.keyboardFocus = false; }, true);
    rail.addEventListener('focusin', () => this.wakeChrome());
    rail.addEventListener('focusout', () => this.restChrome());
    shell.addEventListener('pointerdown', event => {
      if ((event.target as Element).closest('.workspace-strip,[data-surface]')) this.wakeChrome();
      else this.restChrome();
    });
    const grip = shell.querySelector<HTMLElement>('.rail-grip')!;
    grip.addEventListener('pointerdown', event => {
      if (event.button) return;
      this.railDrag = {pointer: event.pointerId, edge: this.state.toolbarEdge, started: false, y: event.clientY};
      grip.setPointerCapture(event.pointerId); event.preventDefault();
    });
    grip.addEventListener('keydown', event => {
      if (!['ArrowUp', 'ArrowDown', 'Home'].includes(event.key)) return;
      event.preventDefault(); this.state.toolbarEdge = event.key === 'ArrowDown' ? 'bottom' : 'top'; this.commit();
    });
  }
  wakeChrome() {
    window.clearTimeout(this.chromeTimer); this.shell.classList.remove('chrome-quiet'); this.restChrome();
  }
  restChrome() {
    window.clearTimeout(this.chromeTimer);
    this.chromeTimer = window.setTimeout(() => {
      const rail = this.shell.querySelector('.workspace-strip')!;
      const protectedInteraction = this.overRail || this.keyboardFocus && rail.contains(document.activeElement)
        || this.shell.querySelector('[data-surface].visible,dialog[open],.layout-interacting,.surface-interacting,.window-edge.pulling');
      if (protectedInteraction || this.operation || this.railDrag) { this.restChrome(); return; }
      this.quiet(); this.shell.classList.add('chrome-quiet');
    }, 1800);
  }
  register(id: string, element: HTMLElement, preferred: 'left' | 'right', initial: Box) {
    this.panels.set(id, {id, element, preferred, initial}); element.dataset.surface = id;
    element.addEventListener('pointerdown', () => this.raise(id), true);
    const header = element.querySelector<HTMLElement>('header,.drawer-header')!;
    header.classList.add('surface-header');
    const grip = document.createElement('button'); grip.type = 'button'; grip.className = 'surface-grip'; grip.textContent = '⠿';
    grip.title = '拖动面板 · 方向键移动，Home 恢复位置'; grip.setAttribute('aria-label', '移动面板'); header.prepend(grip);
    const dock = document.createElement('button'); dock.type = 'button'; dock.className = 'surface-dock'; dock.textContent = '▥';
    dock.setAttribute('aria-label', preferred === 'left' ? '停靠到左侧' : '停靠到右侧'); dock.title = dock.getAttribute('aria-label')!;
    header.insertBefore(dock, header.lastElementChild);
    const body = document.createElement('div'); body.className = 'surface-body';
    for (const child of [...element.children]) if (child !== header) body.append(child);
    element.append(body);
    dock.addEventListener('click', () => {
      const state = this.panelState(id); state.dock = state.dock === 'float' ? preferred : 'float'; this.commit();
    });
    header.addEventListener('pointerdown', event => {
      if (event.button || ((event.target as Element).closest('button,input,select,a') && !(event.target as Element).closest('.surface-grip'))) return;
      this.begin(id, event, 'move');
    });
    grip.addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home'].includes(event.key)) return;
      event.preventDefault(); event.stopPropagation(); const value = this.panelState(id), amount = event.shiftKey ? 50 : 20;
      if (event.key === 'Home') this.state.panels[id] = {...initial, dock: 'float'};
      else { value.dock = 'float'; value.x += event.key === 'ArrowLeft' ? -amount : event.key === 'ArrowRight' ? amount : 0; value.y += event.key === 'ArrowUp' ? -amount : event.key === 'ArrowDown' ? amount : 0; }
      this.commit();
    });
    for (const edge of ['n','s','e','w','ne','nw','se','sw']) {
      const handle = document.createElement('div'); handle.className = 'surface-resize'; handle.dataset.edge = edge;
      if (edge === 'se') { handle.tabIndex = 0; handle.setAttribute('role','button'); handle.setAttribute('aria-label','调整面板大小（方向键）'); }
      handle.addEventListener('pointerdown', event => this.begin(id, event, edge));
      handle.addEventListener('keydown', event => {
        if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) return;
        event.preventDefault(); event.stopPropagation(); const state = this.panelState(id);
        state.width = Math.max(280, state.width + (event.key === 'ArrowRight' ? 20 : event.key === 'ArrowLeft' ? -20 : 0));
        state.height = Math.max(220, state.height + (event.key === 'ArrowDown' ? 20 : event.key === 'ArrowUp' ? -20 : 0)); this.commit();
      });
      element.append(handle);
    }
    new MutationObserver(() => {
      if (!this.ready) return;
      const state = this.panelState(id), visible = element.classList.contains('visible');
      if (state.open !== visible) { state.open = visible; if (visible) this.raise(id); this.save(structuredClone(this.state)); }
      this.schedule();
    }).observe(element, {attributes: true, attributeFilter: ['class','data-shelf-view']});
  }
  load(raw: unknown) {
    const saved = raw as (Omit<SurfaceState,'version'> & {version:number}) | null; this.state = defaults();
    if (saved?.version === 1 || saved?.version === 2) {
      this.state.toolbarEdge = saved.toolbarEdge === 'bottom' ? 'bottom' : 'top';
      this.state.uiScale = [1,1.15,1.3].includes(saved.uiScale) ? saved.uiScale : 1;
      for (const [id, value] of Object.entries(saved.panels ?? {})) {
        if (!this.panels.has(id) || !value || ![value.x,value.y,value.width,value.height].every(Number.isFinite)) continue;
        this.state.panels[id] = {...value, width: clamp(value.width,280,1800), height: clamp(value.height,220,1800), dock: ['left','right'].includes(value.dock) ? value.dock : 'float'};
        // Only replace untouched former defaults; manually arranged panels keep their geometry.
        if (id === 'navigation' && value.width === 344 && value.height === 620 && value.x === 12 && value.y === 54)
          Object.assign(this.state.panels[id], {width:304,height:420});
        if (id === 'library' && saved.version === 1)
          Object.assign(this.state.panels[id], {x:12,y:54,width:280,height:460,dock:'float',open:false});
      }
    }
    for (const [id, panel] of this.panels) {
      const state = this.panelState(id);
      if (state.open && state.dock !== 'float') { panel.element.classList.add('visible'); panel.element.inert = false; panel.element.setAttribute('aria-hidden','false'); }
    }
    this.ready = true; this.render(); this.restChrome();
  }
  isDocked(id: string) { return this.ready && this.panels.has(id) && this.panelState(id).dock !== 'float'; }
  private raise(id: string) { for (const panel of this.panels.values()) panel.element.style.zIndex = panel.id === id ? '170' : '160'; }
  private panelState(id: string): PanelState { return this.state.panels[id] ??= {...this.panels.get(id)!.initial, dock: 'float'}; }
  setScale(scale: number) { this.state.uiScale = [1,1.15,1.3].includes(scale) ? scale : 1; this.commit(); }
  reset() { this.cancel(); this.state.panels = {}; this.state.toolbarEdge = 'top'; this.commit(); }
  private commit() { this.render(); this.save(structuredClone(this.state)); }
  private schedule() { if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.render(); }); }
  private bounds(): Box {
    const rail = Math.ceil(44 * this.state.uiScale);
    const windowEdge=this.state.toolbarEdge==='bottom'&&this.shell.dataset.windowFullscreen==='true'?32:0;
    return {x: 8, y: this.state.toolbarEdge === 'top' ? rail + 6 : windowEdge+6, width: Math.max(100,this.shell.clientWidth-16), height: Math.max(100,this.shell.clientHeight-rail-12-windowEdge)};
  }
  private widthLimit(id?:string) { return id==='library'?Math.min(360,Math.max(280,this.shell.clientWidth*.30)):this.bounds().width; }
  private fit(box: Box,id?:string): Box {
    const b = this.bounds(), width = Math.min(b.width,this.widthLimit(id),Math.max(280,box.width)), height = Math.min(b.height,Math.max(220,box.height));
    return {width,height,x:clamp(box.x,b.x,b.x+b.width-width),y:clamp(box.y,b.y,b.y+b.height-height)};
  }
  private position(element: HTMLElement, box: Box) {
    Object.assign(element.style, {left: `${box.x}px`, top: `${box.y}px`, width: `${box.width}px`, height: `${box.height}px`, right: 'auto', bottom: 'auto'});
  }
  render() {
    const b = this.bounds(); this.shell.dataset.toolbarEdge = this.state.toolbarEdge;
    this.shell.style.setProperty('--ui-scale',String(this.state.uiScale));
    this.shell.style.setProperty('--rail-height',`${Math.ceil(44*this.state.uiScale)}px`);
    const docked = (side: Dock) => [...this.panels.values()].filter(p => this.panelState(p.id).dock === side && p.element.classList.contains('visible'));
    const left = docked('left'), right = docked('right');
    const occupiedSides = Number(!!left.length) + Number(!!right.length);
    const canDock = b.width >= Math.max(680, occupiedSides * 280 + 376)
      && Math.max(left.length,right.length) * 220 + Math.max(0,Math.max(left.length,right.length)-1)*8 <= b.height;
    const maxWidth = Math.max(280,(b.width-360-16)/Math.max(1,Number(!!left.length)+Number(!!right.length)));
    const sideWidth = (panels: Panel[]) => panels.length && canDock ? Math.min(maxWidth,Math.max(...panels.map(p=>Math.min(this.widthLimit(p.id),this.panelState(p.id).width)))) : 0;
    const lw = sideWidth(left), rw = sideWidth(right);
    this.grid.style.left = `${lw ? lw+16 : 10}px`; this.grid.style.right = `${rw ? rw+16 : 10}px`;
    for (const [id, panel] of this.panels) {
      const state = this.panelState(id), list = state.dock === 'left' ? left : right, width = state.dock === 'left' ? lw : rw;
      const dock = state.dock !== 'float' && width && list.includes(panel);
      const height = (b.height-8*(list.length-1))/Math.max(1,list.length);
      const box = dock ? {x:state.dock === 'left' ? b.x : b.x+b.width-width,y:b.y+list.indexOf(panel)*(height+8),width,height} : this.fit(state,id);
      if(id==='library'&&!dock&&panel.element.dataset.shelfView==='open'&&!this.operation){
        const outerHeight=(el:Element|null)=>{if(!(el instanceof HTMLElement)||!el.getClientRects().length)return 0;const css=getComputedStyle(el);return el.getBoundingClientRect().height+(parseFloat(css.marginTop)||0)+(parseFloat(css.marginBottom)||0);};
        const controls=['.surface-header','.search-box','.shelf-workspace','.library-options','.workspace-move-controls'].reduce((sum,selector)=>sum+outerHeight(panel.element.querySelector(selector)),0);
        const rows=[...panel.element.querySelector('.workspace-books')!.children].reduce((sum,el)=>sum+outerHeight(el),0);
        box.height=Math.min(box.height,Math.max(220,controls+rows+18));
      }
      if (!this.operation || this.operation.id !== id || this.operation.edge !== 'move') this.position(panel.element,box);
      panel.element.dataset.docked = dock ? state.dock : 'float';
      const button = panel.element.querySelector<HTMLButtonElement>('.surface-dock')!;
      button.textContent = state.dock === 'float' ? '▥' : '↗';
      button.setAttribute('aria-label',state.dock === 'float' ? `停靠到${panel.preferred === 'left' ? '左' : '右'}侧` : '浮动面板');
      button.title = state.dock !== 'float' && !canDock ? '窗口较窄，暂时浮动；变宽后恢复停靠' : button.getAttribute('aria-label')!;
    }
  }
  private begin(id: string, event: PointerEvent, edge: string) {
    if (event.button) return;
    const panel = this.panels.get(id)!, r = panel.element.getBoundingClientRect(), origin = this.shell.getBoundingClientRect();
    this.operation = {id,pointer:event.pointerId,x:event.clientX,y:event.clientY,before:structuredClone(this.panelState(id)),box:{x:r.x-origin.x,y:r.y-origin.y,width:r.width,height:r.height},edge,dock:null,started:false};
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId); event.preventDefault(); event.stopPropagation();
  }
  private move(event: PointerEvent) {
    if (this.railDrag?.pointer === event.pointerId) {
      const d = this.railDrag; if (Math.abs(event.clientY-d.y)<8 && !d.started) return;
      d.started = true; this.veil.hidden = false;
      d.edge = event.clientY < this.shell.clientHeight/2 ? 'top' : 'bottom';
      this.preview.hidden = false; this.preview.textContent = `松开，工具停在${d.edge === 'top'?'顶部':'底部'} · 正文留在工具区之外`;
      this.position(this.preview,{x:8,y:d.edge === 'top'?0:this.shell.clientHeight-44*this.state.uiScale,width:this.shell.clientWidth-16,height:44*this.state.uiScale});return;
    }
    const d = this.operation; if (!d || d.pointer !== event.pointerId) return;
    const dx = event.clientX-d.x, dy = event.clientY-d.y;
    if (!d.started && Math.hypot(dx,dy)<5) return;
    d.started = true; this.veil.hidden = false; this.shell.classList.add('surface-interacting');
    const panel = this.panels.get(d.id)!, state = this.panelState(d.id);
    if (d.edge === 'move') {
      this.position(panel.element,this.fit({...d.box,x:d.box.x+dx,y:d.box.y+dy},d.id));
      d.dock = this.shell.clientWidth >= 696 && event.clientX < 32 ? 'left' : this.shell.clientWidth >= 696 && event.clientX > this.shell.clientWidth-32 ? 'right' : null;
      this.preview.hidden = !d.dock;
      if (d.dock) { const b=this.bounds(), width=Math.min(d.box.width,b.width-380); this.position(this.preview,{...b,x:d.dock==='left'?b.x:b.x+b.width-width,width});this.preview.textContent='松开后停靠 · 为正文留出空间'; }
    } else {
      const box = {...d.box};
      if (d.edge.includes('e')) box.width += dx;
      if (d.edge.includes('s')) box.height += dy;
      if (d.edge.includes('w')) { box.x += dx; box.width -= dx; }
      if (d.edge.includes('n')) { box.y += dy; box.height -= dy; }
      Object.assign(state,this.fit(box,d.id)); this.render();
    }
  }
  private end(event: PointerEvent) {
    if (this.railDrag?.pointer === event.pointerId) { if (this.railDrag.started) this.state.toolbarEdge=this.railDrag.edge; this.railDrag=null; this.cleanup(); this.commit(); return; }
    const d = this.operation; if (!d || d.pointer !== event.pointerId) return;
    if (d.started && d.edge === 'move') Object.assign(this.panelState(d.id),this.fit({...d.box,x:d.box.x+event.clientX-d.x,y:d.box.y+event.clientY-d.y},d.id),{dock:d.dock??'float'});
    this.operation = null; this.cleanup(); this.commit();
  }
  private cleanup() { this.preview.hidden=true; this.veil.hidden=true; this.shell.classList.remove('surface-interacting'); }
  cancel() { if(this.operation)this.state.panels[this.operation.id]=this.operation.before;this.operation=null;this.railDrag=null;this.cleanup();this.render(); }
}
