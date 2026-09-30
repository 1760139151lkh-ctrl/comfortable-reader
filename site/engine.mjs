/** Trusted, deterministic host capability. Book data never supplies JavaScript. */
export function parsePairs(csv) {
  const lines=csv.replace(/^\uFEFF/,'').trim().split(/\r?\n/);
  if (lines[0]?.trim()!=='x,y' || lines.length<3 || lines.length>10001) throw new Error('输入数据需要表头 x,y 和 2—10000 行数值');
  return lines.slice(1).map((line,index)=>{
    const fields=line.split(',');
    const decimal=/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
    if(fields.length!==2 || fields.some(x=>!decimal.test(x.trim())||!Number.isFinite(Number(x)))) throw new Error(`第 ${index+2} 行需要两个有限十进制数值`);
    return [Number(fields[0]),Number(fields[1])];
  });
}
export function leastSquares(rows,{rate=0.12,steps=30}={}) {
  if(!Array.isArray(rows)||rows.length<2||rows.length>10000||rows.some(r=>!Array.isArray(r)||r.length!==2||r.some(v=>!Number.isFinite(v)))) throw new Error('输入记录超出计算范围');
  if(!Number.isFinite(rate)||rate<=0||rate>0.3||!Number.isInteger(steps)||steps<1||steps>200) throw new Error('步长或更新次数超出登记范围');
  let w=0,b=0;const trace=[];
  for(let step=1;step<=steps;step++){
    let sumSquare=0,sumSlope=0,sumOffset=0;
    for(const [x,y] of rows){const error=(w*x+b)-y;sumSquare+=error*error;sumSlope+=error*x;sumOffset+=error;}
    const loss=sumSquare/rows.length,gradW=2*sumSlope/rows.length,gradB=2*sumOffset/rows.length;
    const before={w,b};w-=rate*gradW;b-=rate*gradB;
    if(![w,b,loss].every(Number.isFinite)) throw new Error('计算出现非有限数值；没有保存伪结果');
    trace.push({step,before,loss_before:loss,gradient:{w:gradW,b:gradB},after:{w,b}});
  }
  return {identity:'browser_builtin_least_squares_v1',rate,steps,records:rows.length,weights:{w,b},predictions:rows.map(([x,y])=>({x,target:y,prediction:w*x+b})),trace};
}
