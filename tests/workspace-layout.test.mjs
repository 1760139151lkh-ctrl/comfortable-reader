import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from '../desktop/node_modules/typescript/lib/typescript.js';
const source=await readFile(new URL('../desktop/src/workspace-layout.ts',import.meta.url),'utf8');
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ES2022,target:ts.ScriptTarget.ES2022}}).outputText;
const {defaultLayout,normalizeWorkspace,measureTree,leaves,movePane,removePane,addPane,resizeSplit,fitsLayout}=await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
const box={x:0,y:0,width:1480,height:850};
function invariant(tree,ids){
  assert.deepEqual([...leaves(tree)].sort((a,b)=>a-b),[...ids].sort((a,b)=>a-b));
  const rects=[...measureTree(tree,box).books.values()];
  for(const r of rects){assert.ok(r.width>0&&r.height>0);assert.ok(r.x>=0&&r.y>=0&&r.x+r.width<=box.width+.001&&r.y+r.height<=box.height+.001);}
  for(let i=0;i<rects.length;i++)for(let j=i+1;j<rects.length;j++){const a=rects[i],b=rects[j];assert.ok(a.x+a.width<=b.x+.001||b.x+b.width<=a.x+.001||a.y+a.height<=b.y+.001||b.y+b.height<=a.y+.001);}
}
test('one book owns the space; defaults give a primary book and references',()=>{
  assert.deepEqual([...measureTree(defaultLayout([0],1480,850),box).books.values()],[box]);
  const three=defaultLayout([0,1,2],1480,850);invariant(three,[0,1,2]);const m=measureTree(three,box).books;
  assert.ok(m.get(0).width>m.get(1).width);assert.equal(m.get(1).x,m.get(2).x);assert.ok(fitsLayout(three,box));
});
test('manual movement, ratios and sparse slot identities survive serialization and narrow windows',()=>{
  let tree=defaultLayout([0,3,8],1480,850);tree=movePane(tree,8,0,'swap');tree=resizeSplit(tree,tree.id,.62);
  const raw={version:1,tree,manual:true,focused:8};const saved=JSON.parse(JSON.stringify(raw));
  assert.deepEqual(normalizeWorkspace(saved,[0,3,8],400,700),raw);
  assert.equal(fitsLayout(tree,{...box,width:400}),false);assert.deepEqual(saved,raw);
  assert.deepEqual(normalizeWorkspace(saved,[0,3,8],2000,1000),raw);invariant(tree,[0,3,8]);
});
test('adding beside a book keeps the opposite branch and all existing identities',()=>{
  const tree=defaultLayout([1,4],1480,850),opposite=structuredClone(tree.second);const next=addPane(tree,7,1,box);
  assert.deepEqual(next.second,opposite);assert.equal(next.ratio,tree.ratio);invariant(next,[1,4,7]);assert.deepEqual(removePane(next,7),tree);
});
test('moves and closes never duplicate or lose the remaining books',()=>{
  for(const direction of ['left','right','top','bottom','swap'])for(let s=0;s<4;s++)for(let t=0;t<4;t++){
    const tree=movePane(defaultLayout([0,1,2,3],1480,850),s,t,direction);invariant(tree,[0,1,2,3]);invariant(removePane(tree,t),[0,1,2,3].filter(n=>n!==t));
  }
});
test('damaged and legacy layouts recover open books without trusting unknown leaves',()=>{
  const tree={kind:'split',id:'same',axis:'x',ratio:NaN,first:{kind:'book',pane:1},second:{kind:'split',id:'same',axis:'y',ratio:9,first:{kind:'book',pane:1},second:{kind:'book',pane:19}}};
  for(const raw of [null,{}, {version:1,tree,manual:true,focused:19}]){
    const state=normalizeWorkspace(raw,[0,1,2],1480,850);invariant(state.tree,[0,1,2]);assert.equal(state.focused,null);
  }
});
