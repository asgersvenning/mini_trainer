/* Optional label relaxation changes thumbnail positions only, never prototype coordinates. */
let mapThumbTimer = null, mapThumbController = null, mapThumbGeneration = 0;
let mapThumbLayout = [];
const mapThumbImages = new Map();
const mapThumbFailures = new Set();
function thumbnailSharedArea(a,b){
 return Math.max(0,Math.min(a.left+a.width,b.left+b.width)-Math.max(a.left,b.left))*Math.max(0,Math.min(a.top+a.height,b.top+b.height)-Math.max(a.top,b.top));
}
function thumbnailEnergy(boxes,gradient=false){
 let energy=0,overlap=0,pairs=0;
 // Diagonal contact-curvature estimate preconditions crowded and free axes
 // differently; a floor bounds steps along weakly constrained directions.
 const curvature=gradient?boxes.map(()=>[.05,.05]):null;
 const forces=gradient?boxes.map(b=>[(b.anchorX-b.x)*.003,(b.anchorY-b.y)*.003]):null;
 for(const b of boxes)energy+=.0015*((b.x-b.anchorX)**2+(b.y-b.anchorY)**2);
 for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){
  const a=boxes[i],b=boxes[j],dx=b.x-a.x,dy=b.y-a.y;
  const px=(a.width+b.width)/2+2-Math.abs(dx),py=(a.height+b.height)/2+2-Math.abs(dy);
  if(px<=0||py<=0)continue;
  if(px>2&&py>2){pairs++;overlap=Math.max(overlap,(px-2)*(py-2)/Math.min(a.width*a.height,b.width*b.height));}
  // Smooth minimum penetration of the actual border boxes. The forces below
  // are the negative gradient of this same, time-independent contact energy.
  const depth=Math.min(px,py),wx=Math.exp(-(px-depth)/2),wy=Math.exp(-(py-depth)/2);
  const contact=Math.max(0,depth-2*Math.log(wx+wy));energy+=.5*contact*contact;
  if(!gradient||contact===0)continue;
  const angle=((a.id*137.508+b.id*71.3)%360)*Math.PI/180;
  const sx=dx===0?(Math.cos(angle)<0?-1:1):Math.sign(dx),sy=dy===0?(Math.sin(angle)<0?-1:1):Math.sign(dy);
  const fx=sx*contact*wx/(wx+wy),fy=sy*contact*wy/(wx+wy);
  const hx=(wx/(wx+wy))**2,hy=(wy/(wx+wy))**2;curvature[i][0]+=hx;curvature[j][0]+=hx;curvature[i][1]+=hy;curvature[j][1]+=hy;
  forces[i][0]-=fx;forces[i][1]-=fy;forces[j][0]+=fx;forces[j][1]+=fy;
 }
 return {energy,overlap,pairs,forces,curvature};
}
function stepThumbnailForce(boxes,width,height,maxShift,state={}){
 if(typeof state!=='object')state={};
 const before=thumbnailEnergy(boxes,true),positions=boxes.map(b=>[b.x,b.y]);
 let rate=state.rate||1,after=before,speed=0,accepted=false;
 for(let trial=0;trial<14;trial++){
  speed=0;
  boxes.forEach((b,i)=>{
   const momentum=trial===0?.55:0;
   let dx=before.forces[i][0]*rate/before.curvature[i][0]+(b.vx||0)*momentum,dy=before.forces[i][1]*rate/before.curvature[i][1]+(b.vy||0)*momentum;
   const distance=Math.hypot(dx,dy);if(distance>8){dx*=8/distance;dy*=8/distance;}
   let x=Math.max(b.width/2,Math.min(width-b.width/2,positions[i][0]+dx)),y=Math.max(b.height/2,Math.min(height-b.height/2,positions[i][1]+dy));
   const offset=Math.hypot(x-b.anchorX,y-b.anchorY);
   if(offset>maxShift){x=b.anchorX+(x-b.anchorX)*maxShift/offset;y=b.anchorY+(y-b.anchorY)*maxShift/offset;}
   b.x=x;b.y=y;speed=Math.max(speed,Math.hypot(x-positions[i][0],y-positions[i][1]));
  });
  after=thumbnailEnergy(boxes);
  if(after.energy<=before.energy+1e-9){accepted=true;break;}
  // Retry without inertia before shortening the gradient step.
  if(trial>0)rate*=.5;
 }
 if(!accepted){boxes.forEach((b,i)=>{b.x=positions[i][0];b.y=positions[i][1];});after=before;speed=0;}
 boxes.forEach((b,i)=>{b.vx=b.x-positions[i][0];b.vy=b.y-positions[i][1];b.left=b.x-b.width/2;b.top=b.y-b.height/2;});
 state.rate=Math.min(4,rate*1.1);state.steps=(state.steps||0)+1;
 const small=speed<.04&&Math.abs(after.energy-before.energy)/Math.max(1,boxes.length)<.002&&Math.abs(after.overlap-before.overlap)<1e-4;
 state.quiet=small?(state.quiet||0)+1:0;
 state.metrics={energy:after.energy,overlap:after.overlap,pairs:after.pairs,speed};
 state.done=state.quiet>=8||state.steps>=480;
 state.reason=state.quiet>=8?'resting':'work limit';
 return speed;
}
function relaxThumbnails(boxes,width,height,maxShift){
 const state={};while(!state.done)stepThumbnailForce(boxes,width,height,maxShift,state);
 return boxes;
}
// Cascaded time-domain filters separate solver convergence from visible motion.
// A speed-limited guide and its continuous follower use CSS pixels and seconds.
// Convex combinations preserve viewport and anchor-radius bounds; retaining the
// guide on each box preserves velocity when loading or navigation reheats layout.
function stepThumbnailPresentation(boxes,targets,seconds){
 const omega=8,maxSpeed=140;
 let remaining=Math.min(.05,Math.max(0,seconds));
 while(remaining>1e-9){
  const dt=Math.min(1/240,remaining);remaining-=dt;
  boxes.forEach((b,i)=>{
   const target=targets[i];b.guideX??=b.x;b.guideY??=b.y;
   const dx=target.x-b.guideX,dy=target.y-b.guideY,distance=Math.hypot(dx,dy);
   const fraction=distance?Math.min(1-Math.exp(-omega*dt),maxSpeed*dt/distance):0;
   const decay=Math.exp(-omega*dt);
   for(const [key,guide] of [['x','guideX'],['y','guideY']]){
    const start=b[guide],velocity=(target[key]-start)*fraction/dt;
    // Exact follower response to a linearly moving guide in this substep.
    b[key]=start+velocity*dt-velocity/omega+(b[key]-start+velocity/omega)*decay;
    b[guide]=start+velocity*dt;
   }
  });
 }
 let speed=0,error=0;
 boxes.forEach((b,i)=>{
  b.left=b.x-b.width/2;b.top=b.y-b.height/2;
  speed=Math.max(speed,omega*Math.hypot((b.guideX??b.x)-b.x,(b.guideY??b.y)-b.y));
  error=Math.max(error,Math.hypot(targets[i].x-b.x,targets[i].y-b.y),Math.hypot(targets[i].x-(b.guideX??b.x),targets[i].y-(b.guideY??b.y)));
 });
 return {speed,error,done:speed<.02&&error<.005};
}
let mapThumbMotion=null;
function stopThumbnailMotion(){
 if(!mapThumbMotion)return;
 cancelAnimationFrame(mapThumbMotion.frame);
 mapThumbMotion.resolve();mapThumbMotion=null;
}
function paintThumbnailMotion(){
 for(const box of mapThumbLayout){
  box.button.style.left=box.x+'px';box.button.style.top=box.y+'px';
  if(!box.tether)continue;
  const line=box.tether.querySelector('line');
  line.setAttribute('x1',box.anchorX);line.setAttribute('y1',box.anchorY);
  const dot=box.tether.querySelector('circle');dot.setAttribute('cx',box.anchorX);dot.setAttribute('cy',box.anchorY);
  line.setAttribute('x2',Math.max(box.left,Math.min(box.left+box.width,box.anchorX)));
  line.setAttribute('y2',Math.max(box.top,Math.min(box.top+box.height,box.anchorY)));
  box.tether.style.visibility=Math.hypot(box.x-box.anchorX,box.y-box.anchorY)<.5?'hidden':'visible';
 }
}
let mapThumbRest=null;
function animateThumbnails(config){
 // Image arrivals join the existing frame loop; they never reset its clock,
 // numerical destinations, or presentation velocity.
 if(mapThumbMotion)return mapThumbMotion.promise;
 mapThumbRest=null;
 const motion={frame:0,state:{},last:0,targets:[],byId:new Map()};
 mapThumbMotion=motion;
 return motion.promise=new Promise(resolve=>{
  motion.resolve=resolve;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finish=()=>{
   const measured=thumbnailEnergy(mapThumbLayout);
   mapThumbRest={...motion.state.metrics,energy:measured.energy,overlap:measured.overlap,pairs:measured.pairs,reason:motion.state.reason,steps:motion.state.steps,presentationSpeed:motion.presentation?.speed||0};
   paintThumbnailMotion();mapThumbMotion=null;resolve();
  };
  const syncTargets=time=>{
   const active=mapThumbLayout.filter(b=>reduced||!b.readyAt||time>=b.readyAt);
   const changed=active.length!==motion.byId.size||active.some(b=>!motion.byId.has(b.id));
   motion.targets=active.map(b=>motion.byId.get(b.id)||({...b,vx:0,vy:0}));
   motion.byId=new Map(motion.targets.map(b=>[b.id,b]));
   if(changed)motion.state={};
   return active.length===mapThumbLayout.length;
  };
  const advance=()=>stepThumbnailForce(motion.targets,config.width,config.height,2*config.size,motion.state);
  if(reduced){
   syncTargets(Infinity);
   while(!motion.state.done)advance();
   mapThumbLayout.forEach((b,i)=>{b.x=motion.targets[i].x;b.y=motion.targets[i].y;b.guideX=b.x;b.guideY=b.y;b.left=b.x-b.width/2;b.top=b.y-b.height/2;});
   finish();return;
  }
  const tick=time=>{
   if(mapThumbMotion!==motion)return;
   const dt=motion.last?(time-motion.last)/1000:0;motion.last=time;
   const allReady=syncTargets(time);
   for(let i=0;i<2&&!motion.state.done;i++)advance();
   // Each new image waits independently; existing contacts and destinations
   // continue evolving while it fades in. Match by ID, not admission order.
   const targets=mapThumbLayout.map(b=>motion.byId.get(b.id)||b);
   motion.presentation=stepThumbnailPresentation(mapThumbLayout,targets,dt);
   paintThumbnailMotion();
   if(allReady&&motion.state.done&&motion.presentation.done){finish();return;}
   motion.frame=requestAnimationFrame(tick);
  };
  motion.frame=requestAnimationFrame(tick);
 });
}
function thumbnailFootprint(size,labels,aspect=1){
 const ratio=Number.isFinite(aspect)&&aspect>0?aspect:1;
 const imageWidth=size*Math.min(1,ratio),imageHeight=size*Math.min(1,1/ratio);
 return {imageWidth,imageHeight,width:imageWidth+(labels?8:0),height:imageHeight+(labels?24:0)};
}
function thumbnailLayout(points, priority, {width,height,size,density,overlap,labels=true,push=false,aspects=[],unknownOverlap=overlap}) {
 const budget=Math.min(200,Math.max(1,Math.floor(width*height*density/1e6)));
 const chosen=[],used=new Set();
 for(const id of priority){
  if(used.has(id))continue;used.add(id);const [x,y]=points[id];
  const {width:boxWidth,height:boxHeight}=thumbnailFootprint(size,labels,aspects[id]);
  const box={id,x,y,anchorX:x,anchorY:y,left:x-boxWidth/2,top:y-boxHeight/2,width:boxWidth,height:boxHeight};
  if(box.left<0||box.top<0||box.left+boxWidth>width||box.top+boxHeight>height)continue;
  const area=boxWidth*boxHeight;
  const blocked=chosen.some(other=>{const limit=aspects[id]>0&&aspects[other.id]>0?overlap:unknownOverlap;return thumbnailSharedArea(box,other)/Math.min(area,other.width*other.height)>limit+1e-12;});
  if(!blocked)chosen.push(box);if(chosen.length>=budget)break;
 }
 const boxes=push?relaxThumbnails(chosen,width,height,2*size):chosen;
 return {boxes,budget,culled:chosen.length-boxes.length};
}
function thumbnailTether(box,always=false){
 if(!always&&Math.hypot(box.x-box.anchorX,box.y-box.anchorY)<.5)return null;
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
 svg.classList.add('map-thumb-tether');svg.setAttribute('aria-hidden','true');svg.style.zIndex='201';
 const line=document.createElementNS(ns,'line');
 const edgeX=Math.max(box.left,Math.min(box.left+box.width,box.anchorX)),edgeY=Math.max(box.top,Math.min(box.top+box.height,box.anchorY));
 for(const [key,value] of Object.entries({x1:box.anchorX,y1:box.anchorY,x2:edgeX,y2:edgeY,stroke:'#435b70','stroke-width':1}))line.setAttribute(key,String(value));
 const dot=document.createElementNS(ns,'circle');
 for(const [key,value] of Object.entries({cx:box.anchorX,cy:box.anchorY,r:3,fill:'#ffffff',stroke:'#243b50','stroke-width':1.5}))dot.setAttribute(key,String(value));
 svg.append(line,dot);return svg;
}
function mapThumbnailHit(point){
 const rect=projectionCanvas.getBoundingClientRect(),x=point[0]*rect.width/1200,y=point[1]*rect.height/600;
 const box=mapThumbLayout.find(b=>x>=b.left&&x<=b.left+b.width&&y>=b.top&&y<=b.top+b.height);
 return box?.id??-1;
}
function showMapPhotoCredit(index){
 const node=$('map-photo-credit');node.replaceChildren();if(index<0)return;
 const id=data.names[index],album=photoAlbums.get(id);if(!album?.photos.length)return;
 const photo=album.photos[(photoChoices.get(id)||0)%album.photos.length];
 node.append(photoText('span',`${album.display_name} · class ${id} · ${photo.creator} · ${photo.license} · `),photoLink('Image source',photo.source),photoText('span',' · '),photoLink('GBIF record','https://www.gbif.org/occurrence/'+photo.occurrence_id));
}
let mapThumbCase=null,mapThumbKey='',mapThumbOffsets=new Map();
function retireThumbnail(box){
 // Remove interaction immediately; the short visual exit is independent of
 // solver membership and cannot delay newly visible thumbnails.
 box.button.disabled=true;box.button.style.pointerEvents='none';
 box.button.setAttribute('aria-hidden','true');
 for(const node of [box.button,box.tether]){
  if(!node)continue;
  if(matchMedia('(prefers-reduced-motion: reduce)').matches){node.remove();continue;}
  const opacity=getComputedStyle(node).opacity;
  const fade=node.animate([{opacity},{opacity:0}],{duration:100,fill:'forwards'});
  fade.onfinish=()=>node.remove();
 }
}
function reanchorThumbnails(points,width,height){
 const kept=[];
 for(const box of mapThumbLayout){
  const point=points[box.id];if(!point){retireThumbnail(box);continue;}
  const dx=box.x-box.anchorX,dy=box.y-box.anchorY;
  if(box.guideX!==undefined){box.guideX+=point[0]-box.anchorX;box.guideY+=point[1]-box.anchorY;}
  box.anchorX=point[0];box.anchorY=point[1];box.x=point[0]+dx;box.y=point[1]+dy;
  box.left=box.x-box.width/2;box.top=box.y-box.height/2;
  if(box.left<0||box.top<0||box.left+box.width>width||box.top+box.height>height){retireThumbnail(box);continue;}
  kept.push(box);
 }
 mapThumbLayout=kept;paintThumbnailMotion();
}
function setThumbnailAnchors(){
 $('map-thumbnail-layer').classList.toggle('hide-thumb-anchors',!$('map-photo-anchors').checked);
}
function scheduleMapThumbnails(){
 clearTimeout(mapThumbTimer);stopThumbnailMotion();mapThumbController?.abort();++mapThumbGeneration;
 const size=Number($('map-photo-size').value),density=Number($('map-photo-density').value),overlap=Number($('map-photo-overlap').value);
 const key=[size,$('map-photo-labels').checked,$('map-photo-push').checked,$('map-photo-aspect').checked,$('projection-plane').value].join(':');
 if($('map-photos').checked&&mapThumbCase===data&&mapThumbKey===key){
  const rect=projectionCanvas.getBoundingClientRect();
  reanchorThumbnails(projectionHits.map(([x,y])=>[x*rect.width/1200,y*rect.height/600]),rect.width,rect.height);
  mapThumbOffsets=new Map(mapThumbLayout.map(b=>[b.id,[b.x-b.anchorX,b.y-b.anchorY]]));
 }else{mapThumbOffsets.clear();mapThumbLayout=[];$('map-thumbnail-layer').replaceChildren();}
 mapThumbCase=data;mapThumbKey=key;
 $('map-size-value').value=size+' px';$('map-density-value').value=density+' / MP';$('map-overlap-value').value=overlap+'%';
 if(!$('map-photos').checked){$('map-photo-status').textContent='Enable map thumbnails to interpret numeric class IDs as GBIF taxa. Push apart connects displaced images to their fixed prototype points.';return;}
 if(data.metadata.synthetic){$('map-photo-status').textContent='Synthetic case: no GBIF photo requests.';return;}
 $('map-photo-status').textContent='Updating visible thumbnails…';
 mapThumbTimer=setTimeout(renderMapThumbnails,180);
}
async function renderMapThumbnails(){
 const generation=mapThumbGeneration,caseData=data;
 mapThumbController=new AbortController();const signal=mapThumbController.signal;
 const rect=projectionCanvas.getBoundingClientRect(),size=Number($('map-photo-size').value);
 const config={width:rect.width,height:rect.height,size,density:Number($('map-photo-density').value),overlap:Number($('map-photo-overlap').value)/100,labels:$('map-photo-labels').checked,push:$('map-photo-push').checked,aspect:$('map-photo-aspect').checked};
 if(config.aspect)config.aspects=data.names.map(id=>{const album=photoAlbums.get(id),photo=album?.photos[(photoChoices.get(id)||0)%(album?.photos.length||1)],image=photo&&mapThumbImages.get(photo.image_path);return image?.naturalWidth/image?.naturalHeight;});
 const points=projectionHits.map(([x,y])=>[x*rect.width/1200,y*rect.height/600]);
 // Selected class first, then its true neighbours; other classes have stable
 // mixed row order so dense regions do not win simply by checkpoint ordering.
 const remaining=data.names.map((_,i)=>i).sort((a,b)=>(Math.imul(a,2654435761)>>>0)-(Math.imul(b,2654435761)>>>0));
 const priority=[selected,...data.neighbours[selected],...remaining].filter(i=>photoNumber(data.names[i])&&!mapThumbFailures.has(data.names[i]));
 const {boxes,budget}=thumbnailLayout(points,priority,{...config,push:false,unknownOverlap:config.aspect&&!config.push?0:config.overlap});
 const ranks=new Map(boxes.map((b,i)=>[b.id,i]));
 const retained=new Map();
 for(const box of mapThumbLayout){
  if(ranks.has(box.id)&&box.photoChoice===(photoChoices.get(data.names[box.id])||0))retained.set(box.id,box);
  else{retireThumbnail(box);}
 }
 mapThumbLayout=[...retained.values()].sort((a,b)=>ranks.get(a.id)-ranks.get(b.id));

 let motionFinished=false;
 let loaded=0,unavailable=0;
 const status=()=>{$('map-photo-status').textContent=`${loaded} images loaded · ${motionFinished?mapThumbLayout.length:boxes.length} visible slots · budget ${budget} for this viewport. ${config.push?`Push apart: ${motionFinished?`${mapThumbRest?.reason||'resting'} · ${mapThumbRest?.pairs||0} overlapping pairs retained`:'adjusting rectangle contacts'}. Images move at most ${2*size} px.`:`Maximum pairwise rectangle overlap ${Math.round(config.overlap*100)}%.`} ${unavailable} unavailable. Hover for names and credits; click to inspect. Prototype coordinates and neighbour ranks stay fixed.`;};
 if(!boxes.length){status();return;}
 try{if(boxes.some(b=>!retained.has(b.id))){const health=await fetch('/api/health',{signal});if(!health.ok||!(await health.json()).photo_api)throw Error('Photo server unavailable');}}
 catch(error){if(!signal.aborted)$('map-photo-status').textContent='Map photos require the local GBIF photo server. Numerical views remain available.';return;}
 const loadBox=async box=>{
  if(signal.aborted||generation!==mapThumbGeneration||data!==caseData)return;
  const id=data.names[box.id];
  if(retained.has(box.id)){loaded++;const old=retained.get(box.id);old.button.style.zIndex=String(boxes.length-ranks.get(box.id));old.button.style.borderColor=box.id===selected?'#c2344b':data.neighbours[selected].includes(box.id)?'#087e8b':'#7c90a3';return;}
  try{
   let album=photoAlbums.get(id);
   if(!album){const response=await fetch('/api/gbif/'+id,{signal});album=await response.json();if(!response.ok)throw Error(album.error||'Photo lookup failed');photoAlbums.set(id,album);}
   if(signal.aborted||generation!==mapThumbGeneration)return;
   if(!album.photos.length){mapThumbFailures.add(id);unavailable++;status();return;}
   const choice=(photoChoices.get(id)||0)%album.photos.length,photo=album.photos[choice];
   let image=mapThumbImages.get(photo.image_path)?.cloneNode();
   const button=photoText('button','','map-thumb');button.type='button';button.style.left=box.x+'px';button.style.top=box.y+'px';button.style.width=box.width+'px';button.style.height=box.height+'px';button.style.zIndex=String(boxes.length-boxes.indexOf(box));button.style.borderColor=box.id===selected?'#c2344b':data.neighbours[selected].includes(box.id)?'#087e8b':'#7c90a3';button.setAttribute('aria-label',`Inspect ${album.display_name}, class ${id}`);button.onclick=()=>selectClass(box.id);button.onfocus=()=>showMapPhotoCredit(box.id);
   button.classList.toggle('image-only',!config.labels);button.classList.toggle('original-aspect',config.aspect);button.title=`${album.display_name} · class ${id}`;
   if(!image){image=new Image();image.decoding='async';}
   image.draggable=false;image.alt=album.display_name;image.style.height=size+'px';
   button.append(image);if(config.labels)button.append(photoText('span',id));
   const tether=thumbnailTether(box,config.push);
   const okay=(image.complete&&image.naturalWidth>0)||await loadReferenceImage(image,photo.image_path,signal);
   if(signal.aborted||generation!==mapThumbGeneration)return;
   if(okay){
    const footprint=thumbnailFootprint(size,config.labels,config.aspect?image.naturalWidth/image.naturalHeight:1);
    box.width=footprint.width;box.height=footprint.height;box.left=box.x-box.width/2;box.top=box.y-box.height/2;
    button.style.width=box.width+'px';button.style.height=box.height+'px';image.style.height=footprint.imageHeight+'px';
    box.button=button;box.tether=tether;box.photoChoice=photoChoices.get(id)||0;box.readyAt=performance.now()+80;
    const offset=config.push?mapThumbOffsets.get(box.id):null;
    if(offset){box.x=Math.max(box.width/2,Math.min(config.width-box.width/2,box.anchorX+offset[0]));box.y=Math.max(box.height/2,Math.min(config.height-box.height/2,box.anchorY+offset[1]));box.left=box.x-box.width/2;box.top=box.y-box.height/2;button.style.left=box.x+'px';button.style.top=box.y+'px';}
    if(tether)$('map-thumbnail-layer').append(tether);$('map-thumbnail-layer').append(button);
    const bounds=button.getBoundingClientRect();box.width=bounds.width;box.height=bounds.height;box.left=box.x-box.width/2;box.top=box.y-box.height/2;
    mapThumbLayout.push(box);mapThumbLayout.sort((a,b)=>ranks.get(a.id)-ranks.get(b.id));
    if(config.push)animateThumbnails(config);
    loaded++;mapThumbImages.delete(photo.image_path);mapThumbImages.set(photo.image_path,image);while(mapThumbImages.size>128)mapThumbImages.delete(mapThumbImages.keys().next().value);}
   else{button.remove();tether?.remove();mapThumbLayout=mapThumbLayout.filter(b=>b!==box);mapThumbFailures.add(id);unavailable++;}
   status();if(box.id===selected)showMapPhotoCredit(selected);
  }catch(error){if(signal.aborted)return;mapThumbFailures.add(id);unavailable++;status();}
 };
 // Bound in-flight work; a slow image no longer stalls every later class.
 let next=0;
 await Promise.all(Array.from({length:Math.min(4,boxes.length)},async()=>{
  while(next<boxes.length&&!signal.aborted&&generation===mapThumbGeneration)await loadBox(boxes[next++]);
 }));
 if(config.push&&!signal.aborted&&generation===mapThumbGeneration)await animateThumbnails(config);
 if(generation===mapThumbGeneration){motionFinished=true;status();showMapPhotoCredit(selected);}
}
for(const name of ['map-photos','map-photo-size','map-photo-density','map-photo-overlap','map-photo-labels','map-photo-push','map-photo-aspect'])$(name).oninput=drawProjection;
$('map-photo-retry').onclick=()=>{mapThumbFailures.clear();scheduleMapThumbnails();};
$('map-photo-anchors').oninput=setThumbnailAnchors;setThumbnailAnchors();
new ResizeObserver(scheduleMapThumbnails).observe(projectionCanvas);
scheduleMapThumbnails();
