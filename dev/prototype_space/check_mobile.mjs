// Focused image ingestion and multi-touch regression; camera hardware still needs manual review.
import {spawn} from 'node:child_process';
import {mkdtempSync,writeFileSync,readFileSync,rmSync} from 'node:fs';
if(!process.env.CHROME_PATH||!process.argv[2]||!process.argv[3])throw Error('Usage: CHROME_PATH=/path/to/chrome node check_mobile.mjs URL LARGE_JPEG [OUTPUT_JSON]');
const profile=mkdtempSync('/tmp/browser-mobile-check-');
const chrome=spawn(process.env.CHROME_PATH,['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-first-run','--disable-background-networking','--remote-debugging-pipe','--user-data-dir='+profile],{stdio:['ignore','ignore','ignore','pipe','pipe']});
let serial=0,buffer=Buffer.alloc(0);const pending=new Map(),errors=[];
chrome.stdio[4].on('data',chunk=>{buffer=Buffer.concat([buffer,chunk]);let end;while((end=buffer.indexOf(0))!==-1){const msg=JSON.parse(buffer.subarray(0,end));buffer=buffer.subarray(end+1);if(msg.id){const p=pending.get(msg.id);pending.delete(msg.id);msg.error?p.reject(Error(JSON.stringify(msg.error))):p.resolve(msg.result);}else if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params.exceptionDetails);}});
function send(method,params={},sessionId){const id=++serial;return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');});}
const timeout=setTimeout(()=>{chrome.kill();process.exitCode=1;console.error('Timeout');},60000);
try{
 const {targetId}=await send('Target.createTarget',{url:'about:blank'});const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});const call=(m,p)=>send(m,p,sessionId);
 await call('Runtime.enable');await call('Page.enable');await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 await call('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:5});await call('Page.navigate',{url:process.argv[2]});
 const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 for(let i=0;i<150;i++){if(await evaluate('typeof captureViewerState==="function"'))break;await new Promise(r=>setTimeout(r,100));}
 if(!await evaluate('typeof prepareInferenceImage==="function"'))throw Error('Viewer did not load');
 const checks=await evaluate(`(async()=>{
 const checks=[],check=(v,name)=>{if(!v)throw Error(name);checks.push(name)};
 const canvas=document.createElement('canvas');canvas.width=80;canvas.height=40;const ctx=canvas.getContext('2d');
 for(let y=0;y<40;y++)for(let x=0;x<80;x++){ctx.fillStyle='rgb('+x*3+','+y*6+','+(x+y)%256+')';ctx.fillRect(x,y,1,1);}
 const blob=await new Promise(r=>canvas.toBlob(r,'image/jpeg',1));const raw=new Uint8Array(await blob.arrayBuffer());
 const original=await createImageBitmap(blob);ctx.drawImage(original,0,0);original.close();const pixels=ctx.getImageData(0,0,80,40);
 const m={preprocessing:'nearest-square-uint8-bilinear-center-imagenet-v1',size:16,resize:18};
 const sampled=await prepareInferenceImage(blob,16),before=browserPreprocess(pixels,m),after=browserPreprocess(sampled,m);
 check(before.every((v,i)=>v===after[i]),'opaque JPEG preserves model input exactly');
 const mappings=[null,(x,y)=>[x,y],(x,y)=>[79-x,y],(x,y)=>[79-x,39-y],(x,y)=>[x,39-y],(x,y)=>[y,x],(x,y)=>[y,39-x],(x,y)=>[79-y,39-x],(x,y)=>[79-y,x]];
 for(let orientation=1;orientation<=8;orientation++){
  const exif=new Uint8Array([255,225,0,34,69,120,105,102,0,0,73,73,42,0,8,0,0,0,1,0,18,1,3,0,1,0,0,0,orientation,0,0,0,0,0,0,0]);
  const jpeg=new Blob([raw.slice(0,2),exif,raw.slice(2)],{type:'image/jpeg'}),image=await prepareInferenceImage(jpeg,8);
  let same=true;for(let y=0;y<8;y++)for(let x=0;x<8;x++){
   const [sx,sy]=mappings[orientation](x*(orientation>=5?40:80)/8,y*(orientation>=5?80:40)/8);
   for(let c=0;c<3;c++)if(image.data[4*(y*8+x)+c]!==pixels.data[4*(sy*80+sx)+c])same=false;
  }
  check(same,'EXIF orientation '+orientation);
 }
 canvas.width=canvas.height=8;ctx.clearRect(0,0,8,8);const png=await new Promise(r=>canvas.toBlob(r));const white=await prepareInferenceImage(png,8);
 check(white.data.every(v=>v===255),'transparent PNG composited on white');
 const webp=await new Promise(r=>canvas.toBlob(r,'image/webp'));check((await prepareInferenceImage(webp,8)).width===8,'WebP accepted');
 const bitmap=window.createImageBitmap;try{window.createImageBitmap=async()=>{throw Error('simulate decoder gap')};check((await prepareInferenceImage(blob,8)).width===8,'native image decoder fallback');}finally{window.createImageBitmap=bitmap;}
 let error='';try{await prepareInferenceImage(new File(['invalid'],'example.heic',{type:'image/heic'}),8);}catch(e){error=e.message;}
 check(error.includes('HEIC/HEIF')&&error.includes('JPEG'),'unsupported format guidance');
 check($('inference-camera').getAttribute('capture')==='environment'&&$('inference-camera').onchange===$('inference-image').onchange,'camera and gallery share inference path');
 setInferenceInputDisabled(false);let opened=false;const click=$('inference-camera').click;$('inference-camera').click=()=>{opened=true};$('inference-camera-button').click();$('inference-camera').click=click;
 check(opened,'camera button activates capture input');setInferenceInputDisabled(true);
 return checks;
 })()`);
 const large=readFileSync(process.argv[3]).toString('base64');
 const largeResult=await evaluate(`(async()=>{const blob=await (await fetch('data:image/jpeg;base64,${large}')).blob();const source=await createImageBitmap(blob);const megapixels=source.width*source.height/1e6;source.close();if(megapixels<=40)throw Error('Supply a JPEG larger than 40 MP');const start=performance.now();const image=await prepareInferenceImage(blob,384);return {megapixels,width:image.width,height:image.height,ms:performance.now()-start};})()`);
 if(largeResult.width!==384||largeResult.height!==384)throw Error('Large image sampling failed');checks.push('42 MP JPEG sampled to model dimensions');
 await evaluate('setWorkspaceView("projection");window.scrollTo(0,0)');await new Promise(r=>setTimeout(r,300));
 const start=await evaluate('({view:{...projectionView},selected,rect:projectionCanvas.getBoundingClientRect().toJSON(),coordinates:JSON.stringify(currentProjection().coordinates)})');
 const cx=start.rect.x+start.rect.width/2,cy=start.rect.y+Math.min(start.rect.height/2,300);
 const touch=async(type,points)=>call('Input.dispatchTouchEvent',{type,touchPoints:points.map(([id,x,y])=>({id,x,y,radiusX:4,radiusY:4,force:1}))});
 await touch('touchStart',[[1,cx-35,cy],[2,cx+35,cy]]);await touch('touchMove',[[1,cx-70,cy],[2,cx+70,cy]]);await touch('touchEnd',[]);
 const zoomed=await evaluate('({view:{...projectionView},selected,coordinates:JSON.stringify(currentProjection().coordinates),pointers:projectionPointers.size})');
 if(Math.abs(zoomed.view.scale/start.view.scale-2)>0.03||zoomed.selected!==start.selected||zoomed.coordinates!==start.coordinates||zoomed.pointers!==0)throw Error('Pinch invariants failed: '+JSON.stringify(zoomed.view));
 checks.push('real touch pinch doubles scale without selection or geometry changes');
 await touch('touchStart',[[1,cx,cy]]);await touch('touchMove',[[1,cx+20,cy+15]]);await touch('touchEnd',[]);
 const panned=await evaluate('({...projectionView})');if(panned.x===zoomed.view.x||panned.scale!==zoomed.view.scale)throw Error('Pan after pinch failed');checks.push('single finger pan after pinch');
 await touch('touchStart',[[1,cx,cy],[2,cx+40,cy]]);await touch('touchCancel',[]);
 if(await evaluate('projectionPointers.size!==0||projectionDrag!==null'))throw Error('Cancelled gesture retained pointers');checks.push('touch cancellation clears gesture');
 if(errors.length)throw Error(JSON.stringify(errors));const result={checks,largeResult,errors};if(process.argv[4])writeFileSync(process.argv[4],JSON.stringify(result,null,2));console.log(JSON.stringify(result));await send('Browser.close');
}catch(error){console.error(error);chrome.kill();process.exitCode=1;}finally{clearTimeout(timeout);chrome.once('exit',()=>rmSync(profile,{recursive:true,force:true}));}
