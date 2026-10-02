import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from '../desktop/node_modules/typescript/lib/typescript.js';
const source=await readFile(new URL('../desktop/src/reading-input.ts',import.meta.url),'utf8');
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ES2022,target:ts.ScriptTarget.ES2022}}).outputText;
const {WheelPager}=await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
test('one touchpad swipe plus a decaying inertial tail turns once',()=>{
  const pager=new WheelPager();let now=0;
  const turns=[4,8,15,24,28,21,16,12,8,6,4,3,2,1].map(d=>pager.feed(d,0,now+=16)).filter(Boolean);
  assert.deepEqual(turns,[1]);
});
test('a fast large-delta swipe does not mistake its inertial 120/100 pixels for new notches',()=>{
  const pager=new WheelPager();let now=0;
  assert.deepEqual([240,180,120,100,90,60,45,30,20,12,6].map(d=>pager.feed(d,0,now+=25)).filter(Boolean),[1]);
});
test('separate wheel detents remain responsive and duplicate burst events do not skip pages',()=>{
  const pager=new WheelPager();
  assert.deepEqual([0,8,20,100,108,220].map(t=>pager.feed(120,0,t)),[1,0,0,1,0,1]);
});
test('reversing direction is deliberate and a fresh swipe works after the tail',()=>{
  const pager=new WheelPager();
  assert.equal(pager.feed(60,0,0),1);assert.equal(pager.feed(-60,0,50),-1);
  assert.equal(pager.feed(-60,0,80),0);assert.equal(pager.feed(-60,0,300),-1);
});
test('line-mode mouse input and page-mode input produce one intent; tiny/invalid input does not',()=>{
  const pager=new WheelPager();
  assert.equal(pager.feed(.1,0,0),0);assert.equal(pager.feed(NaN,0,10),0);
  assert.equal(pager.feed(3,1,30),1);assert.equal(pager.feed(1,2,400),1);
});
