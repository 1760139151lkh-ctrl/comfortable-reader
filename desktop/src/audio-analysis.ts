// Deterministic display analysis of the selected decoded channel, off the UI thread.
self.onmessage = (event:MessageEvent<{samples:Float32Array; sampleRate:number}>) => {
  const {samples,sampleRate}=event.data;
  const width=480,height=192,size=512;
  const envelope=new Float32Array(width*2);let sum=0,peak=0;
  for(const value of samples){sum+=value*value;peak=Math.max(peak,Math.abs(value));}
  for(let x=0;x<width;x++){
    const start=Math.floor(x*samples.length/width),end=Math.max(start+1,Math.floor((x+1)*samples.length/width));let lo=1,hi=-1;
    for(let i=start;i<end&&i<samples.length;i++){lo=Math.min(lo,samples[i]);hi=Math.max(hi,samples[i]);}
    envelope[x*2]=lo;envelope[x*2+1]=hi;
  }
  const pixels=new Uint8ClampedArray(width*height*4);
  for(let x=0;x<width;x++){
    const re=new Float64Array(size),im=new Float64Array(size);
    const offset=Math.round(x*Math.max(0,samples.length-size)/(width-1));
    for(let i=0;i<size;i++)re[i]=(samples[offset+i]??0)*(.5-.5*Math.cos(2*Math.PI*i/(size-1)));
    for(let i=1,j=0;i<size;i++){
      let bit=size>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;
      if(i<j){[re[i],re[j]]=[re[j],re[i]];}
    }
    for(let length=2;length<=size;length<<=1){
      const angle=-2*Math.PI/length,wr=Math.cos(angle),wi=Math.sin(angle);
      for(let start=0;start<size;start+=length){let ur=1,ui=0;
        for(let j=0;j<length/2;j++){
          const a=start+j,b=a+length/2;
          const vr=re[b]*ur-im[b]*ui,vi=re[b]*ui+im[b]*ur;
          re[b]=re[a]-vr;im[b]=im[a]-vi;re[a]+=vr;im[a]+=vi;
          const next=ur*wr-ui*wi;ui=ur*wi+ui*wr;ur=next;
        }
      }
    }
    for(let y=0;y<height;y++){
      const bin=Math.round((height-1-y)/(height-1)*(size/2-1));
      const db=20*Math.log10(Math.max(1e-8,Math.hypot(re[bin],im[bin])/(size/2)));
      const level=Math.min(1,Math.max(0,(db+80)/80)),at=(y*width+x)*4;
      pixels[at]=Math.round(20+205*level);pixels[at+1]=Math.round(28+155*Math.sqrt(level));pixels[at+2]=Math.round(36+80*(1-level));pixels[at+3]=255;
    }
  }
  self.postMessage({width,height,pixels,envelope,rms:Math.sqrt(sum/Math.max(1,samples.length)),peak,sampleRate,fftSize:size,duration:samples.length/sampleRate});
};
