/** The study view occupies the reader's workspace; it is not a second product window. */
export function connectStudyWorkspace(surface,{root,readingRoot,toolbar}) {
  let opened=false,previousOverflow='',held=null;
  surface.setAttribute('role','region');surface.hidden=true;
  const readingElement=()=>typeof readingRoot==='function'?readingRoot():readingRoot;
  const place=()=>{
    if(!opened)return;
    const area=(held??readingElement())?.getBoundingClientRect();
    if(area)Object.assign(surface.style,{top:`${area.top}px`,left:`${area.left}px`,width:`${area.width}px`,height:`${area.height}px`,right:'auto',bottom:'auto',margin:'0',maxWidth:'none',maxHeight:'none'});
    else surface.style.top=`${Math.max(0,toolbar?.getBoundingClientRect().bottom??0)}px`;
  };
  const observer=new ResizeObserver(place);if(toolbar)observer.observe(toolbar);if(readingElement())observer.observe(readingElement());
  const movement=new MutationObserver(place);movement.observe(root,{attributes:true,attributeFilter:['data-toolbar-edge']});
  return {
    get open(){return opened;},
    show(){
      if(opened)return;opened=true;previousOverflow=document.body.style.overflow;
      held=typeof readingRoot==='function'?readingRoot():readingRoot;
      if(held){held.dataset.studyPreviousInert=String(held.inert);held.inert=true;}
      document.body.style.overflow='hidden';root.dataset.studyOpen='true';surface.hidden=false;surface.setAttribute('open','');place();
    },
    close(){
      if(!opened)return;opened=false;surface.hidden=true;surface.removeAttribute('open');delete root.dataset.studyOpen;
      document.body.style.overflow=previousOverflow;if(held){held.inert=held.dataset.studyPreviousInert==='true';delete held.dataset.studyPreviousInert;held=null;}
    },
    destroy(){this.close();observer.disconnect();movement.disconnect();},
  };
}
