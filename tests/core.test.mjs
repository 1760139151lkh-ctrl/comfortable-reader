import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {join,dirname,resolve} from 'node:path';
import {validateCatalog,validateBook,validateChapter,selection,planSelection,validPath,resourceUnits} from '../site/core.mjs';
import {leastSquares,parsePairs} from '../site/engine.mjs';

const root=dirname(dirname(fileURLToPath(import.meta.url)));
const built=resolve(root,process.env.SITE_DIST??'site-dist');
const json=async relative=>JSON.parse(await readFile(join(built,relative),'utf8'));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');

test('real three-book catalog has independent chapters and shared capability',async()=>{
  const catalog=validateCatalog(await json('catalog.json'));
  assert.equal(catalog.books.length,3);
  const manifests=await Promise.all(catalog.books.map(async row=>{
    const bytes=await readFile(join(built,row.manifest.path));assert.equal(sha(bytes),row.manifest.sha256);return validateBook(JSON.parse(bytes));
  }));
  assert.notEqual(manifests[0].id,manifests[1].id);
  assert.equal(manifests[0].chapters.length,2);assert.equal(manifests[1].chapters.length,1);assert.equal(manifests[2].chapters.length,2);
  assert.equal(manifests[2].activities.length,0);
  assert.equal(manifests[0].activities[0].capability,manifests[1].activities[0].capability);
  const codes=manifests.map(b=>b.resources.find(r=>r.id==='python-fit'));
  assert.equal(codes[0].sha256,codes[1].sha256);
  for(const [index,book] of manifests.entries())for(const chapter of book.chapters){
    const body=await readFile(join(built,'books',catalog.books[index].slug,chapter.body.path));
    assert.equal(sha(body),chapter.body.sha256);validateChapter(JSON.parse(body),book,chapter.id);
  }
});
test('a chapter and one activity do not pull the next chapter or sibling experiment',async()=>{
  const book=validateBook(await json('books/measurement-lab/manifest.json'));
  const selectedContent=items=>items.filter(x=>x.kind!=='metadata').map(x=>x.id);
  assert.deepEqual(selectedContent(selection(book,'chapter','c01')),['chapter:c01']);
  assert.deepEqual(selectedContent(selection(book,'activity','fit-three')),['chapter:c01','three-points']);
  const withCode=selection(book,'activity','fit-three',['python-fit']);assert.deepEqual(selectedContent(withCode),['chapter:c01','three-points','python-fit']);
  assert.ok(resourceUnits(withCode.find(x=>x.id==='python-fit')).length>1);
  assert.ok(withCode.filter(x=>x.kind==='metadata').every(x=>!x.path.includes('chapter-c02')));
  const stored=new Map(await Promise.all(resourceUnits(book.resources.find(r=>r.id==='python-fit')).map(async unit=>{
    const bytes=await readFile(join(built,'books/measurement-lab',unit.path));
    return [unit.sha256,bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)];
  })));
  const fake={getChunk:async x=>stored.get(x),removeChunk:async x=>stored.delete(x)};
  const plan=await planSelection(fake,book,'activity','fit-three',['python-fit']);
  assert.equal(plan.items.find(x=>x.id==='python-fit').cached,true);
  assert.equal(plan.addedBytes,book.chapters[0].readingDocument.bytes+book.resources.find(x=>x.id==='three-points').bytes+book.reader.support.reduce((n,x)=>n+x.bytes,0));
  assert.ok(!plan.items.some(x=>x.id==='outlier-points'||x.id==='chapter:c02'));
});
test('a damaged selected cache entry is counted missing on the first plan without touching other chapters',async()=>{
  const bytes=values=>Uint8Array.from(values).buffer;
  const body=bytes([1,2,3,4]),resource=bytes([5,6,7]),other=bytes([8,9]);
  const bodyHash=sha(Buffer.from(body)),resourceHash=sha(Buffer.from(resource)),otherHash=sha(Buffer.from(other));
  const book={schemaVersion:1,id:'urn:uuid:00000000-0000-4000-8000-000000000001',revision:'0.1.0',slug:'cache-test',chapters:[
    {id:'c01',title:'One',body:{path:'one.json',bytes:4,sha256:bodyHash},essential:['shared']},
    {id:'c02',title:'Two',body:{path:'two.json',bytes:2,sha256:otherHash}},
  ],resources:[{id:'shared',kind:'text',logicalPath:'shared.txt',path:'shared.txt',bytes:3,sha256:resourceHash}],activities:[],sources:[]};
  const stored=new Map([[bodyHash,bytes([1,2,3,0])],[resourceHash,bytes([5])],[otherHash,other]]);
  const touched=[];const store={
    async hasChunk(){throw new Error('仅有键不能证明内容已保存');},
    async getChunk(key){touched.push(key);return stored.get(key);},
    async removeChunkIfUnchanged(key,observed){if(stored.get(key)!==observed)return false;stored.delete(key);return true;},
  };
  const plan=await planSelection(store,book,'chapter','c01');
  assert.deepEqual(plan.items.map(x=>x.cached),[false,false]);
  assert.equal(plan.addedBytes,7);assert.equal(plan.alreadyPresentBytes,0);
  assert.deepEqual(touched,[bodyHash,resourceHash]);
  assert.equal(stored.has(bodyHash),false);assert.equal(stored.has(resourceHash),false);
  assert.strictEqual(stored.get(otherHash),other);
});
test('trusted generic arithmetic uses actual classroom rows and rejects bad inputs',async()=>{
  const text=await readFile(join(root,'examples/books/measurement-lab/data/three-points.csv'),'utf8');const rows=parsePairs(text),result=leastSquares(rows,{rate:.12,steps:1});
  assert.equal(result.trace[0].loss_before,35/3);assert.ok(Math.abs(result.weights.w-1.04)<1e-12);assert.ok(Math.abs(result.weights.b-.72)<1e-12);
  const second=parsePairs(await readFile(join(root,'examples/books/cooling-lab/data/cooling.csv'),'utf8'));
  assert.ok(leastSquares(second,{rate:.12,steps:100}).weights.w<0);
  assert.throws(()=>parsePairs('x,y\n0,NaN\n1,2'));assert.throws(()=>leastSquares(rows,{rate:-1,steps:30}));
});
test('untrusted declarations cannot add commands, foreign paths or unknown components',async()=>{
  const book=await json('books/measurement-lab/manifest.json');
  const unsafe=structuredClone(book);unsafe.activities[0].command='powershell';assert.throws(()=>validateBook(unsafe));
  const evil=structuredClone(book);evil.resources[0].path='../private.txt';assert.throws(()=>validateBook(evil));
  assert.equal(validPath('../private.txt'),false);assert.equal(validPath('C:\\Users\\someone\\file'),false);assert.equal(validPath('data/point.csv'),true);
  const chapter=await json('books/measurement-lab/'+book.chapters[0].body.path);
  chapter.nodes.push({type:'script',text:'alert(1)'});assert.throws(()=>validateChapter(chapter,book,'c01'));
  const nested=await json('books/measurement-lab/'+book.chapters[0].body.path);
  const table=nested.nodes.find(n=>n.type==='table');
  assert.ok(table);table.rows[0][0].push({type:'script',text:'alert(1)'});
  assert.throws(()=>validateChapter(nested,book,'c01'));
});
