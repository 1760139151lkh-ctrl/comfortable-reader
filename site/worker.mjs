import {leastSquares} from './engine.mjs';
self.addEventListener('message',event=>{
  const {requestId,rows,parameters}=event.data??{};
  try {self.postMessage({requestId,ok:true,result:leastSquares(rows,parameters)});}
  catch(error){self.postMessage({requestId,ok:false,error:String(error)});}
});
