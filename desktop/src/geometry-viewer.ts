import {parsePly, geometryChanges, type GeometryData} from './geometry-data';

export type GeometryAsset = {id: string; title: string; sha256: string};
type View = {yaw: number; pitch: number; distance: number; projection: string; representation: string; vertex: number; highlight: boolean; framing: string};
const clamp = (n: number, low: number, high: number) => Math.max(low, Math.min(high, Number.isFinite(n) ? n : low));
const vertexShader = `#version 300 es
in vec3 aPosition; in vec3 aColor; in float aChanged;
uniform vec3 uCenter; uniform float uRadius; uniform vec4 uCamera;
uniform float uAspect; uniform float uPerspective; uniform float uHighlight; uniform float uPointSize; uniform float uSelection;
out vec3 vColor; out vec3 vPoint;
void main(){
 vec3 q=(aPosition-uCenter)/uRadius; q.yz=-q.yz;
 float cy=cos(uCamera.x),sy=sin(uCamera.x),cp=cos(uCamera.y),sp=sin(uCamera.y);
 vec3 t=vec3(cy*q.x+sy*q.z,q.y,-sy*q.x+cy*q.z);
 vec3 p=vec3(t.x,cp*t.y-sp*t.z,sp*t.y+cp*t.z);
 float z=p.z-uCamera.z; float near=0.02; float far=30.0; float f=1.880726;
 if(uPerspective>0.5) gl_Position=vec4(p.x*f/uAspect,p.y*f,(far+near)/(near-far)*z+2.0*far*near/(near-far),-z);
 else gl_Position=vec4(p.x*f/(uAspect*uCamera.z),p.y*f/uCamera.z,(-z-near)/(far-near)*2.0-1.0,1.0);
 gl_PointSize=uPointSize;
 vColor=mix(mix(aColor,vec3(1.0,0.65,0.20),aChanged*uHighlight),vec3(0.3,0.95,1.0),uSelection);vPoint=p;
}`;
const fragmentShader = `#version 300 es
precision highp float; in vec3 vColor; in vec3 vPoint;
uniform float uSurface; out vec4 color;
void main(){
 float light=1.0;
 if(uSurface>0.5){vec3 n=cross(dFdx(vPoint),dFdy(vPoint));float l=length(n);if(l>0.000001)light=0.72+0.28*abs(dot(n/l,normalize(vec3(0.3,0.6,1.0))));}
 color=vec4(vColor*light,1.0);
}`;

/** No animation loop: paint only on input/resize, and release all GPU objects on exit. */
export async function mountGeometry(parent: HTMLElement, assets: GeometryAsset[], initialId: string,
  read: (id: string) => Promise<ArrayBuffer>, saved: Partial<View> | undefined,
  onChange: (view: View) => void, onError: (e: unknown) => void, onAsset: (id: string) => void): Promise<() => void> {
  parent.classList.add('study-geometry');
  parent.innerHTML = `<details class="geometry-guide" ${innerWidth>1000?'open':''}><summary>如何观察这份三维原件</summary><p class="study-callout">转到侧面，观察有深度的表面与缺失的背面。这里绘制的是原件中的真实顶点；文件有三角面时也可查看表面。转动只改变观察视角，不改原文件，也不重新训练。</p></details>
    <div class="geometry-toolbar"><label>原件<select class="geometry-source" aria-label="三维原件"></select></label><label>表示<select class="geometry-representation"><option value="surface">三角面</option><option value="points">点云</option><option value="edges">三角边</option></select></label><label>投影<select class="geometry-projection"><option value="perspective">透视 · 近大远小</option><option value="orthographic">正交 · 保留平行</option></select></label><label>取景<select class="geometry-framing"><option value="surface">主要表面</option><option value="all">全部范围</option></select></label><button class="geometry-reset">重置视角</button></div>
    <div class="geometry-viewport"><canvas class="geometry-canvas" tabindex="0" role="img" aria-label="三维几何视图；拖动转动，方向键旋转，加减键缩放。下方可读取实际坐标。"></canvas><span class="geometry-hud">正在读取几何原件…</span></div>
    <div class="geometry-controls"><label>左右转动<input class="geometry-yaw" type="range" min="-180" max="180" step="1"/></label><label>上下转动<input class="geometry-pitch" type="range" min="-89" max="89" step="1"/></label><label>观察距离<input class="geometry-distance" type="range" min="2.1" max="8" step="0.05"/></label><button class="geometry-front">正面</button><button class="geometry-side">侧面</button></div>
    <p class="geometry-caption">拖动或使用方向键转动；滚轮或加减键缩放。青点是所选顶点，也可输入序号。主要表面取景依据第一份原件各轴中间 90% 的坐标范围，只改变相机，不筛掉顶点；选“全部范围”可查看远端点。</p>
    <div class="geometry-summary" role="status"></div><label class="geometry-highlight"><input type="checkbox"/>突出同序顶点的坐标改动</label>
    <div class="geometry-inspector"><label>顶点序号（从 1 开始）<input class="geometry-vertex" type="number" min="1" value="1" step="1"/></label><output class="geometry-coordinate"></output></div>
    <details class="study-details"><summary>坐标、来源与观察边界</summary><p class="geometry-provenance"></p><p>观察相机是这个查看器提供的虚拟相机，不等于拍摄照片时的标定相机。视图统一按第一份原件的范围缩放；坐标读数保留原单位。同序比较要求两份网格的顶点数与面索引完全一致，它不证明顶点的语义身份。未被观测的背面不会自动补全。</p></details>`;
  const get = <T extends HTMLElement>(s: string) => parent.querySelector<T>(s)!;
  const stage=document.createElement('div');stage.className='geometry-stage-layout';
  const viewport=get<HTMLElement>('.geometry-viewport');viewport.before(stage);stage.append(viewport);
  const workbench=document.createElement('aside');workbench.className='geometry-workbench';workbench.setAttribute('aria-label','视角与坐标');
  for(const selector of ['.geometry-controls','.geometry-summary','.geometry-highlight','.geometry-inspector'])workbench.append(get(selector));stage.append(workbench);
  const caption=get<HTMLElement>('.geometry-caption');const details=get<HTMLElement>('.study-details');details.prepend(caption);
  const focus=document.createElement('button');focus.type='button';focus.className='geometry-focus';focus.textContent='专注查看';get('.geometry-toolbar').append(focus);
  focus.addEventListener('click',()=>{const active=parent.classList.toggle('geometry-focused');focus.textContent=active?'显示坐标与控件':'专注查看';});
  const canvas = get<HTMLCanvasElement>('.geometry-canvas');
  const gl = canvas.getContext('webgl2', {alpha: false, antialias: true, powerPreference: 'low-power', preserveDrawingBuffer: false});
  if (!gl) throw new Error('当前图形环境不支持 WebGL 2。可继续查看本章参考图和原件信息。');
  const shader = (type: number, text: string) => {const s = gl.createShader(type)!; gl.shaderSource(s,text);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)){gl.deleteShader(s);throw new Error('三维绘图程序初始化失败。');}return s;};
  const vs=shader(gl.VERTEX_SHADER,vertexShader),fs=shader(gl.FRAGMENT_SHADER,fragmentShader),program=gl.createProgram()!;
  gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);gl.deleteShader(vs);gl.deleteShader(fs);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS)){gl.deleteProgram(program);throw new Error('三维绘图程序连接失败。');}
  const buffers: WebGLBuffer[] = [], controller=new AbortController(), options={signal:controller.signal};
  const cache = new Map<string,GeometryData>();
  let data: GeometryData, base: GeometryData, flags: Float32Array, disposed=false, loading=0, raf=0, paints=0;
  let center=[0,0,0],radius=1,edgeCount=0,comparable=false,focusCenter=[0,0,0],focusRadius=1;
  const state:View={yaw:clamp(saved?.yaw??0,-180,180),pitch:clamp(saved?.pitch??0,-89,89),distance:clamp(saved?.distance??2.1,2.1,8),projection:saved?.projection==='orthographic'?'orthographic':'perspective',representation:['points','edges'].includes(saved?.representation??'')?saved!.representation!:'surface',vertex:Math.max(1,Math.round(saved?.vertex??1)),highlight:saved?.highlight===true,framing:saved?.framing==='all'?'all':'surface'};
  const source=get<HTMLSelectElement>('.geometry-source'),representation=get<HTMLSelectElement>('.geometry-representation'),projection=get<HTMLSelectElement>('.geometry-projection');
  for(const a of assets){const option=document.createElement('option');option.value=a.id;option.textContent=a.title;source.append(option);}source.value=initialId;
  function bindAttribute(name: string, array: Float32Array, components: number) {const buffer=gl!.createBuffer()!;buffers.push(buffer);gl!.bindBuffer(gl!.ARRAY_BUFFER,buffer);gl!.bufferData(gl!.ARRAY_BUFFER,array,gl!.STATIC_DRAW);const location=gl!.getAttribLocation(program,name);gl!.enableVertexAttribArray(location);gl!.vertexAttribPointer(location,components,gl!.FLOAT,false,0,0);}
  const faceBuffer=gl.createBuffer()!,edgeBuffer=gl.createBuffer()!;
  function synchronize() {
    get<HTMLInputElement>('.geometry-yaw').value=String(state.yaw);get<HTMLInputElement>('.geometry-pitch').value=String(state.pitch);get<HTMLInputElement>('.geometry-distance').value=String(state.distance);
    representation.value=state.representation;projection.value=state.projection;get<HTMLInputElement>('.geometry-vertex').value=String(state.vertex);
    get<HTMLSelectElement>('.geometry-framing').value=state.framing;
    get<HTMLInputElement>('.geometry-highlight input').checked=state.highlight;
  }
  function coordinate() {
    if(!data)return;const i=state.vertex-1;const units=data.comments.some(s=>/metres|meters/.test(s))?' m':'（原件单位未声明）';
    const xyz=Array.from(data.positions.slice(i*3,i*3+3)).map(v=>v.toFixed(5));
    const delta=comparable?[0,1,2].map(j=>(data.positions[i*3+j]-base.positions[i*3+j]).toFixed(5)):null;
    get<HTMLOutputElement>('.geometry-coordinate').textContent=`X ${xyz[0]} · Y ${xyz[1]} · Z ${xyz[2]}${units}${delta?'　相对第一份网格 Δ = ('+delta.join(', ')+')'+units:''}`;
  }
  function draw() {
    raf=0;if(disposed||!data||gl!.isContextLost())return;
    const rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,2,1920/Math.max(1,rect.width),1200/Math.max(1,rect.height));
    const width=Math.max(1,Math.round(rect.width*ratio)),height=Math.max(1,Math.round(rect.height*ratio));
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
    gl!.viewport(0,0,width,height);gl!.clearColor(.11,.13,.15,1);gl!.clear(gl!.COLOR_BUFFER_BIT|gl!.DEPTH_BUFFER_BIT);gl!.enable(gl!.DEPTH_TEST);gl!.disable(gl!.CULL_FACE);gl!.useProgram(program);
    const u=(name:string)=>gl!.getUniformLocation(program,name);
    gl!.uniform3fv(u('uCenter'),center);gl!.uniform1f(u('uRadius'),radius);gl!.uniform4f(u('uCamera'),state.yaw*Math.PI/180,state.pitch*Math.PI/180,state.distance,0);
    gl!.uniform1f(u('uAspect'),width/height);gl!.uniform1f(u('uPerspective'),state.projection==='perspective'?1:0);gl!.uniform1f(u('uHighlight'),state.highlight?1:0);gl!.uniform1f(u('uPointSize'),2.6*ratio);
    gl!.uniform1f(u('uSurface'),state.representation==='surface'?1:0);gl!.uniform1f(u('uSelection'),0);
    if(state.representation==='points')gl!.drawArrays(gl!.POINTS,0,data.positions.length/3);
    else {gl!.bindBuffer(gl!.ELEMENT_ARRAY_BUFFER,state.representation==='edges'?edgeBuffer:faceBuffer);gl!.drawElements(state.representation==='edges'?gl!.LINES:gl!.TRIANGLES,state.representation==='edges'?edgeCount:data.faces.length,gl!.UNSIGNED_INT,0);}
    gl!.uniform1f(u('uSurface'),0);gl!.uniform1f(u('uSelection'),1);gl!.uniform1f(u('uPointSize'),8*ratio);gl!.depthFunc(gl!.LEQUAL);gl!.drawArrays(gl!.POINTS,state.vertex-1,1);gl!.depthFunc(gl!.LESS);
    canvas.dataset.renderCount=String(++paints);canvas.dataset.vertices=String(data.positions.length/3);canvas.dataset.faces=String(data.faces.length/3);
    get<HTMLElement>('.geometry-hud').textContent=`${state.projection==='perspective'?'透视':'正交'} · ${Math.round(state.yaw)}° / ${Math.round(state.pitch)}° · ${state.representation==='points'?'点云':state.representation==='edges'?'三角边':'三角面'}`;
  }
  function schedule(persist=true) {if(disposed)return;if(base){center=state.framing==='all'?base.min.map((v,i)=>(v+base.max[i])/2):focusCenter;radius=state.framing==='all'?Math.max(...base.max.map((v,i)=>(v-base.min[i])/2),1e-6):focusRadius;}synchronize();coordinate();if(!raf)raf=requestAnimationFrame(draw);if(persist)onChange({...state});}
  async function load(id: string) {
    const generation=++loading;source.disabled=true;
    try {
      if(!cache.has(id))cache.set(id,parsePly(await read(id)));
      if(disposed||generation!==loading)return;
      data=cache.get(id)!;
      if(!base){base=data;const count=base.positions.length/3;const axes=[0,1,2].map(j=>Array.from({length:count},(_,i)=>base.positions[i*3+j]).sort((a,b)=>a-b));focusCenter=axes.map(a=>a[Math.floor(count*.5)]);focusRadius=Math.max(...axes.map(a=>(a[Math.min(count-1,Math.floor(count*.95))]-a[Math.floor(count*.05)])/2),1e-6);}
      const changes=geometryChanges(base,data);comparable=changes!==null;flags=changes?.flags??new Float32Array(data.positions.length/3);
      buffers.splice(0).forEach(b=>gl!.deleteBuffer(b));gl!.useProgram(program);
      bindAttribute('aPosition',data.positions,3);bindAttribute('aColor',data.colors,3);bindAttribute('aChanged',flags,1);
      gl!.bindBuffer(gl!.ELEMENT_ARRAY_BUFFER,faceBuffer);gl!.bufferData(gl!.ELEMENT_ARRAY_BUFFER,data.faces,gl!.STATIC_DRAW);
      const edges=new Uint32Array(data.faces.length*2);for(let i=0;i<data.faces.length;i+=3){const [a,b,c]=data.faces.slice(i,i+3);edges.set([a,b,b,c,c,a],i*2);}edgeCount=edges.length;
      gl!.bindBuffer(gl!.ELEMENT_ARRAY_BUFFER,edgeBuffer);gl!.bufferData(gl!.ELEMENT_ARRAY_BUFFER,edges,gl!.STATIC_DRAW);
      for(const option of Array.from(representation.options))option.disabled=option.value!=='points'&&!data.faces.length;
      if(!data.faces.length)state.representation='points';state.vertex=clamp(state.vertex,1,data.positions.length/3);
      get<HTMLInputElement>('.geometry-vertex').max=String(data.positions.length/3);
      get<HTMLElement>('.geometry-summary').textContent=`${data.positions.length/3} 个顶点 · ${data.faces.length/3} 个三角面${changes&&changes.count?` · ${changes.count} 个同序顶点改变，最大位移 ${changes.maximum.toFixed(5)} 原件单位`:''}`;
      get<HTMLElement>('.geometry-highlight').hidden=!changes?.count;
      get<HTMLElement>('.geometry-provenance').textContent=data.comments.join('；')||'原件没有附带坐标或来源说明，请结合本章来源卡核对。';
      source.value=id;canvas.dataset.asset=id;onAsset(id);schedule(false);
    } catch(e){if(!disposed&&generation===loading){get<HTMLElement>('.geometry-hud').textContent='这份几何原件暂不可用';onError(e);}}finally{if(!disposed&&generation===loading)source.disabled=false;}
  }
  source.addEventListener('change',()=>{void load(source.value);},options);
  representation.addEventListener('change',()=>{state.representation=representation.value;schedule();},options);
  projection.addEventListener('change',()=>{state.projection=projection.value;schedule();},options);
  get<HTMLSelectElement>('.geometry-framing').addEventListener('change',e=>{state.framing=(e.target as HTMLSelectElement).value;state.distance=state.framing==='all'?3.5:2.1;schedule();},options);
  for(const [selector,key] of [['.geometry-yaw','yaw'],['.geometry-pitch','pitch'],['.geometry-distance','distance']] as const)get<HTMLInputElement>(selector).addEventListener('input',e=>{state[key]=Number((e.target as HTMLInputElement).value);schedule();},options);
  get('.geometry-reset').addEventListener('click',()=>{state.yaw=0;state.pitch=0;state.framing='surface';state.distance=2.1;schedule();},options);
  get('.geometry-front').addEventListener('click',()=>{state.yaw=0;state.pitch=0;schedule();},options);
  get('.geometry-side').addEventListener('click',()=>{state.yaw=90;state.pitch=0;schedule();},options);
  get<HTMLInputElement>('.geometry-vertex').addEventListener('change',e=>{if(data){state.vertex=Math.round(clamp(Number((e.target as HTMLInputElement).value),1,data.positions.length/3));schedule();}},options);
  get<HTMLInputElement>('.geometry-highlight input').addEventListener('change',e=>{state.highlight=(e.target as HTMLInputElement).checked;schedule();},options);
  canvas.addEventListener('wheel',e=>{e.preventDefault();state.distance=clamp(state.distance+e.deltaY*.003,2.1,8);schedule();},{...options,passive:false});
  canvas.addEventListener('keydown',e=>{let handled=true;switch(e.key){case'ArrowLeft':state.yaw-=5;break;case'ArrowRight':state.yaw+=5;break;case'ArrowUp':state.pitch-=5;break;case'ArrowDown':state.pitch+=5;break;case'+':case'=':state.distance-=.2;break;case'-':state.distance+=.2;break;default:handled=false;}if(handled){e.preventDefault();state.yaw=((state.yaw+540)%360)-180;state.pitch=clamp(state.pitch,-89,89);state.distance=clamp(state.distance,2.1,8);schedule();}},options);
  let drag: {x:number;y:number;distance:number}|null=null;
  canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;canvas.focus();canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,distance:0};},options);
  canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.distance+=Math.hypot(dx,dy);drag.x=e.clientX;drag.y=e.clientY;state.yaw=((state.yaw+dx*.35+540)%360)-180;state.pitch=clamp(state.pitch+dy*.35,-89,89);schedule();},options);
  canvas.addEventListener('pointerup',e=>{const click=drag&&drag.distance<4;drag=null;if(click)pick(e.clientX,e.clientY);},options);
  canvas.addEventListener('pointercancel',()=>{drag=null;},options);
  function pick(x:number,y:number){
    if(!data)return;const rect=canvas.getBoundingClientRect(),cy=Math.cos(state.yaw*Math.PI/180),sy=Math.sin(state.yaw*Math.PI/180),cp=Math.cos(state.pitch*Math.PI/180),sp=Math.sin(state.pitch*Math.PI/180);
    let chosen=-1,best=14*14;
    for(let i=0;i<data.positions.length;i+=3){const qx=(data.positions[i]-center[0])/radius,qy=-(data.positions[i+1]-center[1])/radius,qz=-(data.positions[i+2]-center[2])/radius;
      const px=cy*qx+sy*qz,tz=-sy*qx+cy*qz,py=cp*qy-sp*tz,pz=sp*qy+cp*tz;
      const divisor=state.projection==='perspective'?state.distance-pz:state.distance;if(divisor<=.02)continue;
      const sx=rect.left+rect.width/2+px*1.880726/divisor*rect.height/2,syScreen=rect.top+rect.height/2-py*1.880726/divisor*rect.height/2;
      const d=(sx-x)**2+(syScreen-y)**2;if(d<best){best=d;chosen=i/3;}
    }
    if(chosen>=0){state.vertex=chosen+1;schedule();}
  }
  const observer=new ResizeObserver(()=>schedule(false));observer.observe(canvas);
  const cleanup=()=>{disposed=true;loading++;controller.abort();observer.disconnect();if(raf)cancelAnimationFrame(raf);buffers.forEach(b=>gl.deleteBuffer(b));gl.deleteBuffer(faceBuffer);gl.deleteBuffer(edgeBuffer);gl.deleteProgram(program);cache.clear();gl.getExtension('WEBGL_lose_context')?.loseContext();};
  canvas.addEventListener('webglcontextlost',e=>{if(!disposed){e.preventDefault();get<HTMLElement>('.geometry-hud').textContent='绘图环境已中断；返回本章后可重新打开。';}},options);
  // Begin with the same registered original so switching never hides a change by recentering.
  await load(assets[0].id);if(initialId!==assets[0].id)await load(initialId);
  return cleanup;
}
