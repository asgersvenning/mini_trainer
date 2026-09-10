/* Image rectangles stay anchored to projected coordinates; culling never moves points. */
let mapThumbTimer = null, mapThumbController = null, mapThumbGeneration = 0;
let mapThumbLayout = [];
const mapThumbImages = new Map();
const mapThumbFailures = new Set();
function thumbnailLayout(points, priority, {width,height,size,density,overlap}) {
 const budget=Math.min(200,Math.max(1,Math.floor(width*height*density/1e6)));
 const chosen=[],used=new Set();const boxWidth=size+8,boxHeight=size+24;
 for(const id of priority){
  if(used.has(id))continue;used.add(id);const [x,y]=points[id];
  const box={id,x,y,left:x-boxWidth/2,top:y-boxHeight/2,width:boxWidth,height:boxHeight};
  if(box.left<0||box.top<0||box.left+boxWidth>width||box.top+boxHeight>height)continue;
  const area=boxWidth*boxHeight;
  const blocked=chosen.some(other=>{
   const shared=Math.max(0,Math.min(box.left+box.width,other.left+other.width)-Math.max(box.left,other.left))*Math.max(0,Math.min(box.top+box.height,other.top+other.height)-Math.max(box.top,other.top));
   return shared/area>overlap+1e-12;
  });
  if(!blocked)chosen.push(box);if(chosen.length>=budget)break;
 }
 return {boxes:chosen,budget};
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
 if(!$('map-photos').checked){$('map-photo-status').textContent='Enable map thumbnails to interpret numeric class IDs as GBIF taxa. Images stay at their projected positions.';return;}
 if(data.metadata.synthetic){$('map-photo-status').textContent='Synthetic case: no GBIF photo requests.';return;}
 $('map-photo-status').textContent='Updating visible thumbnails…';
 mapThumbTimer=setTimeout(renderMapThumbnails,180);
}
async function renderMapThumbnails(){
 const generation=mapThumbGeneration,caseData=data;
 mapThumbController=new AbortController();const signal=mapThumbController.signal;
 const rect=projectionCanvas.getBoundingClientRect(),size=Number($('map-photo-size').value);
 const config={width:rect.width,height:rect.height,size,density:Number($('map-photo-density').value),overlap:Number($('map-photo-overlap').value)/100};
 const points=projectionHits.map(([x,y])=>[x*rect.width/1200,y*rect.height/600]);
 // Selected class first, then its true neighbours; other classes have stable
 // mixed row order so dense regions do not win simply by checkpoint ordering.
 const remaining=data.names.map((_,i)=>i).sort((a,b)=>(Math.imul(a,2654435761)>>>0)-(Math.imul(b,2654435761)>>>0));
 const priority=[selected,...data.neighbours[selected],...remaining].filter(i=>photoNumber(data.names[i])&&!mapThumbFailures.has(data.names[i]));
 const {boxes,budget}=thumbnailLayout(points,priority,config);
 let loaded=0,unavailable=0;
 const status=()=>{$('map-photo-status').textContent=`${loaded} images loaded · ${boxes.length} visible slots · budget ${budget} for this viewport. Maximum pairwise rectangle overlap ${Math.round(config.overlap*100)}%. ${unavailable} unavailable. Hover for credits; click to inspect. Colours/positions and neighbour ranks are unchanged.`;};
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
   if(!image){image=new Image();image.decoding='async';}
   image.draggable=false;image.alt=album.display_name;image.style.height=size+'px';
   button.append(image,photoText('span',id));$('map-thumbnail-layer').append(button);
   mapThumbLayout.push(box); // Placeholders have the same culled footprint.
   const okay=(image.complete&&image.naturalWidth>0)||await loadReferenceImage(image,photo.image_path,signal);
   if(signal.aborted||generation!==mapThumbGeneration)return;
   if(okay){loaded++;mapThumbImages.delete(photo.image_path);mapThumbImages.set(photo.image_path,image);while(mapThumbImages.size>128)mapThumbImages.delete(mapThumbImages.keys().next().value);}
   else{button.remove();mapThumbLayout=mapThumbLayout.filter(b=>b!==box);mapThumbFailures.add(id);unavailable++;}
   status();if(box.id===selected)showMapPhotoCredit(selected);
  }catch(error){if(signal.aborted)return;mapThumbFailures.add(id);unavailable++;status();}
 }
 if(generation===mapThumbGeneration){status();showMapPhotoCredit(selected);}
}
for(const name of ['map-photos','map-photo-size','map-photo-density','map-photo-overlap'])$(name).oninput=drawProjection;
$('map-photo-retry').onclick=()=>{mapThumbFailures.clear();scheduleMapThumbnails();};
new ResizeObserver(scheduleMapThumbnails).observe(projectionCanvas);
scheduleMapThumbnails();
