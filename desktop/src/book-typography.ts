// Compatibility for our own Markdown-derived books. The embedded original
// supplies the formatting intent; the EPUB text and its CFI offsets stay exact.
export function sourceStrongPatterns(markdown:string):Set<string>{
  const result=new Set<string>();let fenced=false;
  for(const line of markdown.split(/\r?\n/)){
    if(/^\s*(```|~~~)/.test(line)){fenced=!fenced;continue;}if(fenced)continue;
    const withoutCode=line.replace(/`[^`]*`/g,'');
    for(const match of withoutCode.matchAll(/(?<!\\)\*\*([^*\n]{1,500})\*\*/g))result.add(match[0]);
  }
  return result;
}
export function restoreSourceStrong(document:Document,allowed:Set<string>):number{
  if(!allowed.size)return 0;
  const root=document.body??document.querySelector('body');if(!root)return 0;
  const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT),nodes:Text[]=[];let node:Node|null;
  while(node=walker.nextNode())if(!(node.parentElement?.closest('pre,code,math,script,style,.reader-annotation')))nodes.push(node as Text);
  let changed=0;
  for(const text of nodes){const value=text.data,matches=[...value.matchAll(/\*\*([^*\n]{1,500})\*\*/g)].filter(m=>allowed.has(m[0]));if(!matches.length)continue;
    const fragment=document.createDocumentFragment();let end=0;
    for(const match of matches){fragment.append(document.createTextNode(value.slice(end,match.index)));
      const before=document.createElement('span'),strong=document.createElement('strong'),after=document.createElement('span');
      for(const el of [before,strong,after])el.className='reader-annotation';
      before.textContent=after.textContent='**';before.hidden=after.hidden=true;strong.textContent=match[1];strong.dataset.sourceFormatting='strong';
      fragment.append(before,strong,after);end=match.index!+match[0].length;changed++;
    }
    fragment.append(document.createTextNode(value.slice(end)));text.replaceWith(fragment);
  }
  return changed;
}
export function protectHeadingWords(document:Document):void{
  const Segmenter=(Intl as any).Segmenter;if(!Segmenter)return;
  const segmenter=new Segmenter('zh-CN',{granularity:'word'});
  for(const heading of document.querySelectorAll<HTMLElement>('h1')){
    if(heading.dataset.readerWords==='true')continue;
    const walker=document.createTreeWalker(heading,NodeFilter.SHOW_TEXT),nodes:Text[]=[];let next:Node|null;
    while(next=walker.nextNode())if(!next.parentElement?.closest('.reader-annotation,math'))nodes.push(next as Text);
    for(const text of nodes){
      const parts=Array.from(segmenter.segment(text.data)) as Array<{segment:string,isWordLike:boolean,index:number}>;
      const first=parts.findIndex(p=>p.isWordLike&&/\p{Script=Han}/u.test(p.segment));if(first<0)continue;
      const prefixEnd=parts[first].index+parts[first].segment.length,fragment=document.createDocumentFragment();
      for(const part of parts.slice(first+1)){
        if(part.isWordLike&&/\p{Script=Han}/u.test(part.segment)&&Array.from(part.segment).length<=8){
          const word=document.createElementNS('http://www.w3.org/1999/xhtml','span');word.className='reader-annotation reader-heading-word';word.style.whiteSpace='nowrap';word.textContent=part.segment;fragment.append(word);
        }else fragment.append(document.createTextNode(part.segment));
      }
      text.data=text.data.slice(0,prefixEnd);text.after(fragment);
    }
    heading.dataset.readerWords='true';
  }
}
type TrackingStyle={element:HTMLElement,value:string,priority:string};
const originalTracking=new WeakMap<HTMLElement,TrackingStyle[]>();
export function fitInlineStops(document:Document):void{
  // Chromium may strand a CJK stop after an atomic inline MathML box.
  // Fit with bounded text tracking; math metrics and all text/CFI nodes stay exact.
  for(const parent of document.querySelectorAll<HTMLElement>('[data-reader-stop-tracking]')){
    for(const old of originalTracking.get(parent)??[]){if(old.value)old.element.style.setProperty('letter-spacing',old.value,old.priority);else old.element.style.removeProperty('letter-spacing');}
    delete parent.dataset.readerStopTracking;
  }
  const groups=new Map<HTMLElement,Array<{span:HTMLElement,range:Range}>>();
  for(const span of document.querySelectorAll<HTMLElement>('span.math-inline')){
    const next=span.nextSibling,parent=span.closest<HTMLElement>('p,li,figcaption');
    if(!parent||next?.nodeType!==Node.TEXT_NODE||!next.textContent?.match(/^[。！，；：、]/))continue;
    const range=document.createRange();range.setStart(next,0);range.setEnd(next,1);
    const entries=groups.get(parent)??[];entries.push({span,range});groups.set(parent,entries);
  }
  for(const [parent,entries] of groups){
    const orphan=()=>entries.some(({span,range})=>range.getBoundingClientRect().top>span.getBoundingClientRect().bottom-2);
    if(!orphan())continue;
    const elements=[parent,...Array.from(parent.querySelectorAll<HTMLElement>('span.math-inline'))];
    const original=elements.map(element=>({element,value:element.style.getPropertyValue('letter-spacing'),priority:element.style.getPropertyPriority('letter-spacing')}));
    for(const element of elements.slice(1))element.style.setProperty('letter-spacing','normal','important');
    let amount=0;
    for(let step=1;step<=6;step++){amount=-step*.005;parent.style.setProperty('letter-spacing',`${amount}em`,'important');if(!orphan())break;}
    if(orphan()){for(const old of original){if(old.value)old.element.style.setProperty('letter-spacing',old.value,old.priority);else old.element.style.removeProperty('letter-spacing');}continue;}
    originalTracking.set(parent,original);parent.dataset.readerStopTracking=String(amount);
  }
}
