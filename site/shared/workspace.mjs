/** The study view occupies the reader's workspace; it is not a second product window. */
export function connectStudyWorkspace(surface,{root,readingRoot,toolbar}) {
  let opened=false,previousOverflow='',held=null;
  surface.setAttribute('role','region');surface.hidden=true;
  const place=()=>{if(opened)surface.style.top=`${Math.max(0,toolbar?.getBoundingClientRect().bottom??0)}px`;};
  const observer=new ResizeObserver(place);if(toolbar)observer.observe(toolbar);
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
    destroy(){this.close();observer.disconnect();},
  };
}
