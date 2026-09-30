/** Persisted spatial relationships, independent of EPUB geometry and window size. */
export type LayoutNode = {kind:'book'; pane:number} | {kind:'split'; id:string; axis:'x'|'y'; ratio:number; first:LayoutNode; second:LayoutNode};
export type WorkspaceState = {version:1; tree:LayoutNode|null; manual:boolean; focused:number|null};
export type Rect = {x:number;y:number;width:number;height:number};
export type Placement = 'left'|'right'|'top'|'bottom'|'swap';
export const MAX_OPEN_BOOKS=12;
export const DIVIDER=12;
export const MIN_BOOK_WIDTH=350;
export const MIN_BOOK_HEIGHT=260;
const leaf=(pane:number):LayoutNode=>({kind:'book',pane});
let serial=0;
const split=(axis:'x'|'y',first:LayoutNode,second:LayoutNode,ratio=.5):LayoutNode=>({kind:'split',id:`split-${Date.now().toString(36)}-${serial++}`,axis,ratio,first,second});
export function leaves(node:LayoutNode|null):number[]{return !node?[]:node.kind==='book'?[node.pane]:[...leaves(node.first),...leaves(node.second)];}
export function removePane(node:LayoutNode|null,pane:number):LayoutNode|null{
  if(!node||node.kind==='book')return node?.pane===pane?null:node;
  const first=removePane(node.first,pane),second=removePane(node.second,pane);
  return first&&second?{...node,first,second}:first??second;
}
export function replacePane(node:LayoutNode,pane:number,replacement:LayoutNode):LayoutNode{
  if(node.kind==='book')return node.pane===pane?replacement:node;
  return {...node,first:replacePane(node.first,pane,replacement),second:replacePane(node.second,pane,replacement)};
}
export function defaultLayout(ids:number[],width:number,height:number):LayoutNode|null{
  if(!ids.length)return null;
  if(ids.length===1)return leaf(ids[0]);
  if(ids.length===3 && width>=MIN_BOOK_WIDTH*2+DIVIDER && height>=MIN_BOOK_HEIGHT*2+DIVIDER)return split('x',leaf(ids[0]),split('y',leaf(ids[1]),leaf(ids[2])),Math.min(.58,1-MIN_BOOK_WIDTH/(width-DIVIDER)));
  const axis=width/Math.max(1,height)>1.28?'x':'y';
  const firstCount=Math.ceil(ids.length/2),ratio=firstCount/ids.length;
  return split(axis,defaultLayout(ids.slice(0,firstCount),axis==='x'?width*ratio:width,axis==='y'?height*ratio:height)!,defaultLayout(ids.slice(firstCount),axis==='x'?width*(1-ratio):width,axis==='y'?height*(1-ratio):height)!,ratio);
}
export function normalizeWorkspace(raw:unknown,ids:number[],width:number,height:number):WorkspaceState{
  const value=raw as Partial<WorkspaceState>|null,seen=new Set<number>(),splitIds=new Set<string>();
  function read(n:any,depth=0):LayoutNode|null{
    if(!n||depth>MAX_OPEN_BOOKS)return null;
    if(n.kind==='book'){
      if(!Number.isInteger(n.pane)||!ids.includes(n.pane)||seen.has(n.pane))return null;
      seen.add(n.pane);return leaf(n.pane);
    }
    if(n.kind!=='split'||!['x','y'].includes(n.axis))return null;
    const a=read(n.first,depth+1),b=read(n.second,depth+1);
    if(!a||!b)return a??b;
    const next=split(n.axis,a,b,Number.isFinite(n.ratio)?Math.max(.08,Math.min(.92,n.ratio)):.5);
    if(next.kind==='split'&&typeof n.id==='string'&&n.id.length<100&&!splitIds.has(n.id)){next.id=n.id;splitIds.add(n.id);}
    return next;
  }
  let tree=value?.version===1?read(value.tree):null;
  if(!tree)tree=defaultLayout(ids,width,height);
  for(const pane of ids.filter(p=>!leaves(tree).includes(p)))tree=tree?split(width>=height?'x':'y',tree,leaf(pane),.65):leaf(pane);
  return {version:1,tree,manual:Boolean(value?.manual),focused:ids.includes(value?.focused as number)?value!.focused!:null};
}
export function measureTree(node:LayoutNode|null,rect:Rect,books=new Map<number,Rect>(),splits=new Map<string,{node:Extract<LayoutNode,{kind:'split'}>;rect:Rect;divider:Rect}>()){
  if(!node)return {books,splits};
  if(node.kind==='book'){books.set(node.pane,rect);return {books,splits};}
  const horizontal=node.axis==='x',size=Math.max(0,(horizontal?rect.width:rect.height)-DIVIDER),firstSize=size*node.ratio;
  const a={...rect,[horizontal?'width':'height']:firstSize};
  const b={...rect,[horizontal?'x':'y']:(horizontal?rect.x:rect.y)+firstSize+DIVIDER,[horizontal?'width':'height']:size-firstSize};
  const divider={...rect,[horizontal?'x':'y']:(horizontal?rect.x:rect.y)+firstSize,[horizontal?'width':'height']:DIVIDER};
  splits.set(node.id,{node,rect,divider});measureTree(node.first,a,books,splits);measureTree(node.second,b,books,splits);return {books,splits};
}
export function minimumSize(node:LayoutNode):{width:number;height:number}{
  if(node.kind==='book')return {width:MIN_BOOK_WIDTH,height:MIN_BOOK_HEIGHT};
  const a=minimumSize(node.first),b=minimumSize(node.second);
  return node.axis==='x'?{width:a.width+b.width+DIVIDER,height:Math.max(a.height,b.height)}:{width:Math.max(a.width,b.width),height:a.height+b.height+DIVIDER};
}
export function fitsLayout(node:LayoutNode|null,rect:Rect):boolean{
  return !node||[...measureTree(node,rect).books.values()].every(r=>r.width>=MIN_BOOK_WIDTH-1&&r.height>=MIN_BOOK_HEIGHT-1);
}
export function resizeSplit(node:LayoutNode,id:string,ratio:number):LayoutNode{
  if(node.kind==='book')return node;
  return node.id===id?{...node,ratio}:{...node,first:resizeSplit(node.first,id,ratio),second:resizeSplit(node.second,id,ratio)};
}
export function movePane(tree:LayoutNode,source:number,target:number,placement:Placement):LayoutNode{
  if(source===target||!leaves(tree).includes(source)||!leaves(tree).includes(target))return tree;
  if(placement==='swap'){
    const swap=(n:LayoutNode):LayoutNode=>n.kind==='book'?leaf(n.pane===source?target:n.pane===target?source:n.pane):{...n,first:swap(n.first),second:swap(n.second)};
    return swap(tree);
  }
  const without=removePane(tree,source)!;
  const before=placement==='left'||placement==='top',axis=placement==='left'||placement==='right'?'x':'y';
  return replacePane(without,target,split(axis,before?leaf(source):leaf(target),before?leaf(target):leaf(source)));
}
export function addPane(tree:LayoutNode|null,pane:number,target:number,rect:Rect):LayoutNode{
  if(!tree)return leaf(pane);
  const targetRect=measureTree(tree,rect).books.get(target);
  const actual=targetRect?target:leaves(tree)[0],box=targetRect??rect;
  return replacePane(tree,actual,split(box.width/Math.max(box.height,1)>1.1?'x':'y',leaf(actual),leaf(pane),.58));
}
