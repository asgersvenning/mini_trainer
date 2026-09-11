/* Predictions use the exported forward; PCA queries use the fitted prototype transform. */
let inferenceWorker=null,inferenceManifest=null,inferenceQuery=null,inferenceSequence=0,predictionPhotos=null,lastPrediction=null,inferencePreviewURL=null;
const inferenceStatus=message=>$('inference-status').textContent=message;
function resetInferenceModel(){++inferenceSequence;inferenceWorker?.terminate();inferenceWorker=null;inferenceManifest=null;inferenceQuery=null;lastPrediction=null;predictionPhotos?.abort();$('inference-results').replaceChildren();$('inference-image').disabled=true;$('inference-preview').hidden=true;if(inferencePreviewURL){URL.revokeObjectURL(inferencePreviewURL);inferencePreviewURL=null;}inferenceStatus('Load the matching browser bundle to begin.');}

$('inference-load').onclick=async()=>{
 inferenceWorker?.terminate();inferenceWorker=null;inferenceQuery=null;inferenceManifest=null;lastPrediction=null;predictionPhotos?.abort();$('inference-results').replaceChildren();
 $('inference-image').disabled=true;++inferenceSequence;
 try{
  const url=new URL($('inference-url').value,location.href);
  if(!['http:','https:'].includes(url.protocol))throw Error('Use an HTTP or HTTPS browser-bundle manifest URL.');
  inferenceStatus('Loading model metadata…');
  const response=await fetch(url);if(!response.ok)throw Error(`Manifest HTTP ${response.status}`);
  const m=await response.json();
  if(m.schema!=='mini-trainer-browser-v1'||m.checkpoint_sha256!==data.metadata.checkpoint_sha256)throw Error('This bundle does not match the displayed checkpoint.');
  if(m.classes[0].length!==data.names.length||m.classes[0].some((name,i)=>name!==data.names[i]))throw Error('Model class order differs from the explorer.');
  const worker=new Worker(new URL('inference_worker.js',url));inferenceWorker=worker;
  worker.onerror=event=>inferenceStatus(`Browser worker failed: ${event.message}. Check same-origin hosting and JavaScript MIME types.`);
  worker.onmessage=({data:result})=>{
   if(worker!==inferenceWorker)return;
   if(result.kind==='ready'){inferenceManifest=m;$('inference-image').disabled=false;inferenceStatus('Model ready. Choose an image; inference runs on this device.');}
   if(result.kind==='error'){inferenceStatus(result.message);$('inference-image').disabled=!inferenceManifest;}
   if(result.kind==='result'&&result.id===inferenceSequence){showInference(result,m);$('inference-image').disabled=false;}
  };
  worker.postMessage({kind:'load',base:url.href,manifest:m,coordinates:data.projections?.['Angular t-SNE']?.coordinates});inferenceStatus('Loading the ONNX model into this browser…');
 }catch(error){inferenceStatus(`${error.message} Cross-origin hosts must allow CORS; this model bundle can be served beside the explorer.`);}
};
$('inference-cancel').onclick=()=>{++inferenceSequence;inferenceWorker?.terminate();inferenceWorker=null;inferenceManifest=null;$('inference-image').disabled=true;inferenceStatus('Stopped. Load the model to continue.');};
$('inference-image').onchange=async event=>{
 const file=event.target.files[0];if(!file||!inferenceManifest)return;
 if(inferencePreviewURL)URL.revokeObjectURL(inferencePreviewURL);inferencePreviewURL=URL.createObjectURL(file);$('inference-preview').src=inferencePreviewURL;$('inference-preview').hidden=false;
 const id=++inferenceSequence;$('inference-image').disabled=true;inferenceStatus('Preparing image…');
 try{
  rejectRotatedExif(await file.arrayBuffer());
  const bitmap=await createImageBitmap(file,{imageOrientation:'none',premultiplyAlpha:'none'});
  if(bitmap.width*bitmap.height>40000000){bitmap.close();throw Error('Use an image smaller than 40 megapixels.');}
  const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;
  const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(bitmap,0,0);bitmap.close();
  const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);
  const tensor=browserPreprocess(pixels,inferenceManifest);
  if(id!==inferenceSequence)return;
  inferenceStatus('Running inference locally…');inferenceWorker.postMessage({kind:'infer',id,tensor},[tensor.buffer]);
 }catch(error){inferenceStatus(error.message);$('inference-image').disabled=false;}
};
function browserPreprocess(image,m){
 if(m.preprocessing!=='nearest-square-uint8-bilinear-center-imagenet-v1')throw Error('Unsupported preprocessing contract.');
 const n=m.size,r=m.resize;const square=new Uint8Array(n*n*3);
 for(let y=0;y<n;y++)for(let x=0;x<n;x++){
  const sy=Math.min(image.height-1,Math.floor(Math.fround(y*Math.fround(image.height/n))));
  const sx=Math.min(image.width-1,Math.floor(Math.fround(x*Math.fround(image.width/n))));
  const source=4*(sy*image.width+sx);if(image.data[source+3]!==255)throw Error('Transparent images are not supported by this bundle.');
  for(let c=0;c<3;c++)square[(y*n+x)*3+c]=image.data[source+c];
 }
 const output=new Float32Array(3*n*n),offset=Math.round((r-n)/2);
 const mean=[.485,.456,.406],std=[.229,.224,.225];
 for(let y=0;y<n;y++)for(let x=0;x<n;x++){
  const fy=Math.max(0,(y+offset+.5)*n/r-.5),fx=Math.max(0,(x+offset+.5)*n/r-.5);
  const y0=Math.floor(fy),x0=Math.floor(fx),y1=Math.min(n-1,y0+1),x1=Math.min(n-1,x0+1),dy=fy-y0,dx=fx-x0;
  for(let c=0;c<3;c++){
   const a=square[(y0*n+x0)*3+c]*(1-dx)+square[(y0*n+x1)*3+c]*dx;
   const b=square[(y1*n+x0)*3+c]*(1-dx)+square[(y1*n+x1)*3+c]*dx;
   const value=Math.round(a*(1-dy)+b*dy);
   output[c*n*n+y*n+x]=Math.fround((Math.fround(value/255)-mean[c])/std[c]);
  }
 }
 return output;
}
function renderInferenceResults(){
 if(!lastPrediction)return;const {result,m}=lastPrediction;
 predictionPhotos?.abort();predictionPhotos=new AbortController();const signal=predictionPhotos.signal;
 const container=$('inference-results');container.replaceChildren();
 result.outputs.forEach((scores,level)=>{
  const max=Math.max(...scores),exp=scores.map(x=>Math.exp(x-max)),sum=exp.reduce((a,b)=>a+b,0);
  const heading=document.createElement('h3');heading.textContent=`Level ${level}`;container.append(heading);
  const list=document.createElement('ol');list.setAttribute('aria-label',`Level ${level} predictions`);list.setAttribute('data-level',level);
  scores.map((score,i)=>({score,i})).sort((a,b)=>b.score-a.score||a.i-b.i).slice(0,5).forEach(({i})=>{
   const row=document.createElement('li'),button=document.createElement('button'),id=m.classes[level][i];const label=level===0?classLabel(i):id;button.textContent=`${label}${label===id?'':' · '+id} · ${(100*exp[i]/sum).toFixed(2)}%`;
   button.onclick=()=>{if(level===0)selectClass(i);};if(level!==0){button.disabled=true;button.title='Higher-rank prediction; no individual species selected.';}row.append(button);list.append(row);
   if(gbifEnabled()){
    const id=m.classes[level][i];
    gbifClient.name(id).then(value=>{if(!signal.aborted&&gbifEnabled()&&!(level===0&&(importedNames.get(data)?.has(i)||data.display_names?.[i])))button.textContent=`${value.display_name} · ${id} · ${(100*exp[i]/sum).toFixed(2)}%`;}).catch(()=>{});
    if(level===0)showPredictionPhoto(row,id,signal);
   }
  });container.append(list);
 });
}
function showInference(result,m){
 lastPrediction={result,m};renderInferenceResults();
 inferenceQuery={embedding:result.embedding,checkpoint:m.checkpoint_sha256,insertion:result.insertion};
 const plane=result.insertion?'Angular t-SNE':Object.keys(data.projections).find(name=>data.projections[name].transform);
 if(plane){$('projection-plane').value=plane;drawProjection();}
 inferenceStatus(`Complete. Scores are not calibrated reliability. Purple diamond: ${result.insertion?`fixed-map t-SNE insertion; ${(100*result.insertion.retained_fraction).toFixed(0)}% of top-12 angular neighbours retained, ${result.insertion.milliseconds.toFixed(0)} ms`:'PCA projection'}. Prototype positions remain fixed.`);
}
function drawInferenceQuery(ctx,p,pixels){
 if(!inferenceQuery||inferenceQuery.checkpoint!==data.metadata.checkpoint_sha256)return;
 if(!p.transform){
  if(p!==data.projections['Angular t-SNE']||!inferenceQuery.insertion)return;
  paintInferencePoint(ctx,inferenceQuery.insertion.coordinates,pixels);return;
 }
 const e=inferenceQuery.embedding,t=p.transform;if(e.length!==t.mean.length)return;
 const norm=t.normalize?Math.hypot(...e):1;const point=[0,0];
 for(let i=0;i<e.length;i++)for(let j=0;j<2;j++)point[j]+=(e[i]/norm-t.mean[i])*t.basis[i][j];
 paintInferencePoint(ctx,point,pixels);
}
function paintInferencePoint(ctx,point,pixels){
 const [x,y]=projectionPoint(point);ctx.fillStyle='#8a26bb';ctx.beginPath();ctx.moveTo(x,y-9*pixels);ctx.lineTo(x+9*pixels,y);ctx.lineTo(x,y+9*pixels);ctx.lineTo(x-9*pixels,y);ctx.closePath();ctx.fill();
}

function rejectRotatedExif(buffer){
 const d=new DataView(buffer);if(d.byteLength<4||d.getUint16(0)!==0xffd8)return;
 let p=2;
 while(p+4<=d.byteLength){
  const marker=d.getUint16(p);p+=2;if(marker===0xffda||marker===0xffd9)return;
  const size=d.getUint16(p);if(size<2||p+size>d.byteLength)throw Error('Malformed JPEG metadata.');
  if(marker===0xffe1&&size>=16&&d.getUint32(p+2)===0x45786966){
   const t=p+8,le=d.getUint16(t)===0x4949,ifd=t+d.getUint32(t+4,le);
   if(ifd+2>p+size)throw Error('Malformed EXIF metadata.');
   const count=d.getUint16(ifd,le);
   for(let i=0;i<count;i++){const e=ifd+2+12*i;if(e+12>p+size)throw Error('Malformed EXIF entries.');
    if(d.getUint16(e,le)===0x112&&d.getUint16(e+8,le)!==1)throw Error('This image has EXIF rotation. Export an unrotated RGB copy for this model.');
   }
  }p+=size;
 }
}

async function showPredictionPhoto(row,id,signal){
 const figure=document.createElement('figure'),image=document.createElement('img'),caption=document.createElement('figcaption');
 image.width=88;image.height=88;image.style.objectFit='contain';image.loading='lazy';image.alt='Reference example for class '+id;
 caption.textContent='Loading reference example…';figure.append(image,caption);row.append(figure);
 try{
  const album=await referenceAlbum(id,signal);if(signal.aborted||!gbifEnabled())return;
  const photo=album.photos[0];if(!photo){image.remove();caption.textContent='No reference image available.';return;}
  caption.replaceChildren(photoText('span',`Reference example · ${photo.creator} · ${photo.license} · `),photoLink('Source',photo.source),photoText('span',' · '),photoLink('GBIF record','https://www.gbif.org/occurrence/'+photo.occurrence_id));
  const loaded=await loadReferenceImage(image,photo.image_path,signal);
  if(!loaded&&!signal.aborted){image.remove();caption.prepend(photoText('span','Image unavailable. '));}
 }catch(error){if(!signal.aborted){image.remove();caption.textContent='Reference image unavailable.';}}
}
