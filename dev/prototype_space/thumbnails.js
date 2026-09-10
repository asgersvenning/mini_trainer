/* Optional label relaxation changes thumbnail positions only, never prototype coordinates. */
let mapThumbTimer = null, mapThumbController = null, mapThumbGeneration = 0;
let mapThumbLayout = [];
const mapThumbImages = new Map();
const mapThumbFailures = new Set();
function thumbnailSharedArea(a,b){
 return Math.max(0,Math.min(a.left+a.width,b.left+b.width)-Math.max(a.left,b.left))*Math.max(0,Math.min(a.top+a.height,b.top+b.height)-Math.max(a.top,b.top));
}
function relaxThumbnails(boxes,width,height,maxShift){
 const move=(box,dx,dy)=>{
  let x=Math.max(box.width/2,Math.min(width-box.width/2,box.x+dx));
  let y=Math.max(box.height/2,Math.min(height-box.height/2,box.y+dy));
  const distance=Math.hypot(x-box.anchorX,y-box.anchorY);
  if(distance>maxShift){x=box.anchorX+(x-box.anchorX)*maxShift/distance;y=box.anchorY+(y-box.anchorY)*maxShift/distance;}
  box.x=x;box.y=y;box.left=x-box.width/2;box.top=y-box.height/2;
 };
 // Bounded deterministic pairwise repulsion. Any unresolved collisions are
 // culled in original priority order, rather than silently retaining overlap.
 for(let pass=0;pass<48;pass++){
  let collisions=0;
  for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){
   const a=boxes[i],b=boxes[j],dx=b.x-a.x,dy=b.y-a.y;
   const ox=(a.width+b.width)/2-Math.abs(dx),oy=(a.height+b.height)/2-Math.abs(dy);
   if(ox<=0||oy<=0)continue;collisions++;
   if(ox<=oy){const push=(ox+.5)/2*(dx<0?-1:1);move(a,-push,0);move(b,push,0);}
   else{const push=(oy+.5)/2*(dy<0?-1:1);move(a,0,-push);move(b,0,push);}
  }
  if(!collisions)break;
 }
 const kept=[];
 for(const box of boxes)if(!kept.some(other=>thumbnailSharedArea(box,other)>1e-8))kept.push(box);
 return kept;
}
function thumbnailLayout(points, priority, {width,height,size,density,overlap,labels=true,push=false}) {
 const budget=Math.min(200,Math.max(1,Math.floor(width*height*density/1e6)));
 const chosen=[],used=new Set();const boxWidth=labels?size+8:size,boxHeight=labels?size+24:size;
 for(const id of priority){
  if(used.has(id))continue;used.add(id);const [x,y]=points[id];
  const box={id,x,y,anchorX:x,anchorY:y,left:x-boxWidth/2,top:y-boxHeight/2,width:boxWidth,height:boxHeight};
  if(box.left<0||box.top<0||box.left+boxWidth>width||box.top+boxHeight>height)continue;
  const area=boxWidth*boxHeight;
  const blocked=chosen.some(other=>thumbnailSharedArea(box,other)/area>overlap+1e-12);
  if(!blocked)chosen.push(box);if(chosen.length>=budget)break;
 }
 const boxes=push?relaxThumbnails(chosen,width,height,2*size):chosen;
 return {boxes,budget,culled:chosen.length-boxes.length};
}
function thumbnailTether(box){
 if(Math.hypot(box.x-box.anchorX,box.y-box.anchorY)<.5)return null;
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
function scheduleMapThumbnails(){
 clearTimeout(mapThumbTimer);mapThumbController?.abort();++mapThumbGeneration;
 mapThumbLayout=[];$('map-thumbnail-layer').replaceChildren();
 const size=Number($('map-photo-size').value),density=Number($('map-photo-density').value),overlap=Number($('map-photo-overlap').value);
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
 const config={width:rect.width,height:rect.height,size,density:Number($('map-photo-density').value),overlap:Number($('map-photo-overlap').value)/100,labels:$('map-photo-labels').checked,push:$('map-photo-push').checked};
 const points=projectionHits.map(([x,y])=>[x*rect.width/1200,y*rect.height/600]);
 // Selected class first, then its true neighbours; other classes have stable
 // mixed row order so dense regions do not win simply by checkpoint ordering.
 const remaining=data.names.map((_,i)=>i).sort((a,b)=>(Math.imul(a,2654435761)>>>0)-(Math.imul(b,2654435761)>>>0));
 const priority=[selected,...data.neighbours[selected],...remaining].filter(i=>photoNumber(data.names[i])&&!mapThumbFailures.has(data.names[i]));
 const {boxes,budget,culled}=thumbnailLayout(points,priority,config);
 let loaded=0,unavailable=0;
 const status=()=>{$('map-photo-status').textContent=`${loaded} images loaded · ${boxes.length} visible slots · budget ${budget} for this viewport. ${config.push?`Push apart: overlap admits candidates; displayed rectangles do not overlap. Images move at most ${2*size} px; ${culled} unresolved collisions culled.`:`Maximum pairwise rectangle overlap ${Math.round(config.overlap*100)}%.`} ${unavailable} unavailable. Hover for names and credits; click to inspect. Prototype coordinates and neighbour ranks stay fixed.`;};
 if(!boxes.length){status();return;}
 try{const health=await fetch('/api/health',{signal});if(!health.ok||!(await health.json()).photo_api)throw Error('Photo server unavailable');}
 catch(error){if(!signal.aborted)$('map-photo-status').textContent='Map photos require the local GBIF photo server. Numerical views remain available.';return;}
 for(const box of boxes){
  if(signal.aborted||generation!==mapThumbGeneration||data!==caseData)return;
  const id=data.names[box.id];
  try{
   let album=photoAlbums.get(id);
   if(!album){const response=await fetch('/api/gbif/'+id,{signal});album=await response.json();if(!response.ok)throw Error(album.error||'Photo lookup failed');photoAlbums.set(id,album);}
   if(signal.aborted||generation!==mapThumbGeneration)return;
   if(!album.photos.length){mapThumbFailures.add(id);unavailable++;status();continue;}
   const choice=(photoChoices.get(id)||0)%album.photos.length,photo=album.photos[choice];
   let image=mapThumbImages.get(photo.image_path)?.cloneNode();
   const button=photoText('button','','map-thumb');button.type='button';button.style.left=box.x+'px';button.style.top=box.y+'px';button.style.width=box.width+'px';button.style.height=box.height+'px';button.style.zIndex=String(boxes.length-boxes.indexOf(box));button.style.borderColor=box.id===selected?'#c2344b':data.neighbours[selected].includes(box.id)?'#087e8b':'#7c90a3';button.setAttribute('aria-label',`Inspect ${album.display_name}, class ${id}`);button.onclick=()=>selectClass(box.id);button.onfocus=()=>showMapPhotoCredit(box.id);
   button.classList.toggle('image-only',!config.labels);button.title=`${album.display_name} · class ${id}`;
   if(!image){image=new Image();image.decoding='async';}
   image.draggable=false;image.alt=album.display_name;image.style.height=size+'px';
   button.append(image);if(config.labels)button.append(photoText('span',id));
   const tether=thumbnailTether(box);if(tether)$('map-thumbnail-layer').append(tether);
   $('map-thumbnail-layer').append(button);
   mapThumbLayout.push(box); // Placeholders have the same culled footprint.
   const okay=(image.complete&&image.naturalWidth>0)||await loadReferenceImage(image,photo.image_path,signal);
   if(signal.aborted||generation!==mapThumbGeneration)return;
   if(okay){loaded++;mapThumbImages.delete(photo.image_path);mapThumbImages.set(photo.image_path,image);while(mapThumbImages.size>128)mapThumbImages.delete(mapThumbImages.keys().next().value);}
   else{button.remove();tether?.remove();mapThumbLayout=mapThumbLayout.filter(b=>b!==box);mapThumbFailures.add(id);unavailable++;}
   status();if(box.id===selected)showMapPhotoCredit(selected);
  }catch(error){if(signal.aborted)return;mapThumbFailures.add(id);unavailable++;status();}
 }
 if(generation===mapThumbGeneration){status();showMapPhotoCredit(selected);}
}
for(const name of ['map-photos','map-photo-size','map-photo-density','map-photo-overlap','map-photo-labels','map-photo-push'])$(name).oninput=drawProjection;
$('map-photo-retry').onclick=()=>{mapThumbFailures.clear();scheduleMapThumbnails();};
new ResizeObserver(scheduleMapThumbnails).observe(projectionCanvas);
scheduleMapThumbnails();
