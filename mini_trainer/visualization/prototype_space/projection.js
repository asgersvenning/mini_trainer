/* Coordinates are fitted offline. This canvas changes only the viewing transform. */
let projectionCase = null;
let projectionView = {x:0, y:0, scale:1};
let projectionDrag = null;
let projectionHits = [];
const projectionCanvas = $('projection-map');
function currentProjection() { return data.projections?.[$('projection-plane').value]; }
function projectionPoint(point) {return [600+(point[0]-projectionView.x)*projectionView.scale,300-(point[1]-projectionView.y)*projectionView.scale];}
function fitProjection(ids) {
 const p=currentProjection();if(!p)return;
 const points=ids.map(i=>p.coordinates[i]);
 let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
 for(const [x,y] of points){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);}
 projectionView={x:(x0+x1)/2,y:(y0+y1)/2,scale:Math.min(1050/Math.max(.0001,x1-x0),450/Math.max(.0001,y1-y0))};
}
function drawProjection() {
 const ctx=projectionCanvas.getContext('2d');ctx.clearRect(0,0,1200,600);
 if(projectionCase!==data){
  projectionCase=data;$('projection-plane').replaceChildren();
  for(const name of Object.keys(data.projections||{})){const option=document.createElement('option');option.value=option.textContent=name;$('projection-plane').append(option);}
  if(data.projections?.['Angular t-SNE'])$('projection-plane').value='Angular t-SNE';
  fitProjection(data.names.map((_,i)=>i));
 }
 const p=currentProjection();if(!p){$('projection-quality').textContent='Regenerate this report to include spatial projections.';return;}
 projectionHits=p.coordinates.map(projectionPoint);
 const original=new Set(data.neighbours[selected]),projected=new Set(p.neighbours[selected]);
 ctx.lineWidth=1;ctx.strokeStyle='#dce3e9';ctx.fillStyle='#52687b';ctx.font='12px system-ui';
 // Both axes use the same units per pixel, including after pan/zoom.
 for(let i=0;i<=4;i++){
  const x=60+i*270,y=30+i*135;
  ctx.beginPath();ctx.moveTo(x,20);ctx.lineTo(x,570);ctx.stroke();
  ctx.fillText((projectionView.x+(x-600)/projectionView.scale).toPrecision(3),x+3,590);
  ctx.beginPath();ctx.moveTo(35,y);ctx.lineTo(1180,y);ctx.stroke();
  ctx.fillText((projectionView.y-(y-300)/projectionView.scale).toPrecision(3),2,y+13);
 }
 ctx.fillText(p.axes[0]+' →',1120,15);ctx.fillText(p.axes[1],5,15);
 for(let i=0;i<projectionHits.length;i++){
  const [x,y]=projectionHits[i];if(x<0||x>1200||y<0||y>600)continue;
  ctx.fillStyle='#7c90a34d';ctx.beginPath();ctx.arc(x,y,2,0,2*Math.PI);ctx.fill();
 }
 const [sx,sy]=projectionHits[selected];
 if($('projection-edges').checked){ctx.strokeStyle='#087e8b88';for(const i of original){const [x,y]=projectionHits[i];ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(x,y);ctx.stroke();}}
 for(const i of projected){if(original.has(i))continue;const [x,y]=projectionHits[i];ctx.strokeStyle='#c87719';ctx.lineWidth=1.8;ctx.beginPath();ctx.arc(x,y,5,0,2*Math.PI);ctx.stroke();}
 ctx.font='12px system-ui';
 for(const i of original){const [x,y]=projectionHits[i];ctx.fillStyle='#087e8b';ctx.beginPath();ctx.arc(x,y,4.5,0,2*Math.PI);ctx.fill();if(!$('map-photos').checked)ctx.fillText(data.names[i],x+7,y-5);}
 ctx.fillStyle='#c2344b';ctx.beginPath();ctx.arc(sx,sy,6,0,2*Math.PI);ctx.fill();ctx.font='bold 13px system-ui';if(!$('map-photos').checked)ctx.fillText(data.names[selected],sx+9,sy+17);
 const fractions=p.variance_fraction?.map(v=>(v*100).toFixed(2));
 const summary=fractions?`Variance retained: ${(100*p.variance_fraction.reduce((a,b)=>a+b,0)).toFixed(2)}% (${p.axes[0]} ${fractions[0]}%, ${p.axes[1]} ${fractions[1]}%). `:`Angular t-SNE · perplexity ${p.parameters.perplexity.toFixed(1)} · seed ${p.parameters.seed}. Map gaps and areas do not measure spherical distances or areas. `;
 $('projection-quality').textContent=`${summary}Original top-${data.stats.k} neighbours retained among plane top-${data.stats.k}: ${(100*p.mean_retained_fraction).toFixed(1)}% averaged over classes; ${(100*p.retained_fraction[selected]).toFixed(1)}% for selected class. Exact plane-distance ties use class order.`;
 if(typeof scheduleMapThumbnails==='function')scheduleMapThumbnails();
}
function projectionMouse(event){const r=projectionCanvas.getBoundingClientRect();return [(event.clientX-r.left)*1200/r.width,(event.clientY-r.top)*600/r.height];}
function projectionHit(point){if(typeof mapThumbnailHit==='function'){const hit=mapThumbnailHit(point);if(hit>=0)return hit;}let best=-1,distance=100;projectionHits.forEach(([x,y],i)=>{const d=(x-point[0])**2+(y-point[1])**2;if(d<distance){best=i;distance=d;}});return best;}
projectionCanvas.onpointerdown=e=>{if(e.button!==0)return;projectionDrag={point:projectionMouse(e),view:{...projectionView},moved:false};projectionCanvas.setPointerCapture(e.pointerId);};
projectionCanvas.onpointermove=e=>{
 const point=projectionMouse(e);
 if(projectionDrag){const dx=point[0]-projectionDrag.point[0],dy=point[1]-projectionDrag.point[1];if(Math.hypot(dx,dy)>3)projectionDrag.moved=true;projectionView.x=projectionDrag.view.x-dx/projectionView.scale;projectionView.y=projectionDrag.view.y+dy/projectionView.scale;drawProjection();return;}
 const id=projectionHit(point);
 if(typeof showMapPhotoCredit==='function')showMapPhotoCredit(id);
 $('projection-hover').textContent=id<0?'Hover a point for its class ID.':`Class ${data.names[id]} · checkpoint row ${id}${id===selected?' · selected':data.neighbours[selected].includes(id)?' · original-space neighbour':currentProjection().neighbours[selected].includes(id)?' · nearby in this plane only':''}`;
};
projectionCanvas.onpointerup=e=>{if(!projectionDrag)return;const click=!projectionDrag.moved;projectionDrag=null;if(projectionCanvas.hasPointerCapture(e.pointerId))projectionCanvas.releasePointerCapture(e.pointerId);if(click){const id=projectionHit(projectionMouse(e));if(id>=0)selectClass(id);}};
projectionCanvas.onpointercancel=()=>{projectionDrag=null;};
projectionCanvas.addEventListener('wheel',e=>{
 e.preventDefault();const [x,y]=projectionMouse(e),old=projectionView.scale;
 const scale=Math.max(1,Math.min(1e7,old*Math.exp(-Math.max(-300,Math.min(300,e.deltaY))*.002)));
 projectionView.x+=(x-600)*(1/old-1/scale);projectionView.y-=(y-300)*(1/old-1/scale);projectionView.scale=scale;drawProjection();
},{passive:false});
$('projection-plane').onchange=()=>{fitProjection(data.names.map((_,i)=>i));drawProjection();};
$('projection-reset').onclick=()=>{fitProjection(data.names.map((_,i)=>i));drawProjection();};
$('projection-focus').onclick=()=>{fitProjection([selected,...data.neighbours[selected]]);drawProjection();};
$('projection-edges').onchange=drawProjection;
drawProjection();
