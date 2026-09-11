// Optional browser smoke check. Set CHROMIUM_BIN; pass a report URL/path and output directory.
import {spawn} from 'node:child_process';
import {writeFileSync,mkdtempSync,rmSync,mkdirSync,existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {pathToFileURL} from 'node:url';
import {resolve,join} from 'node:path';
const executable=process.env.CHROMIUM_BIN;
if(!executable||!existsSync(executable)||!process.argv[2])throw Error('Set CHROMIUM_BIN and pass report URL/path [output directory]');
const output=resolve(process.argv[3]||'tmp/prototype-browser-check');mkdirSync(output,{recursive:true});
const profile=mkdtempSync(tmpdir()+'/prototype-browser-');
const chrome=spawn(executable,['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-extensions','--remote-debugging-pipe','--user-data-dir='+profile],{stdio:['ignore','ignore','ignore','pipe','pipe']});
let sequence=0,buffer=Buffer.alloc(0);const pending=new Map();const errors=[];
chrome.stdio[4].on('data',chunk=>{buffer=Buffer.concat([buffer,chunk]);let end;while((end=buffer.indexOf(0))!==-1){const msg=JSON.parse(buffer.subarray(0,end));buffer=buffer.subarray(end+1);if(msg.id){const cb=pending.get(msg.id);if(cb){pending.delete(msg.id);msg.error?cb.reject(Error(JSON.stringify(msg.error))):cb.resolve(msg.result);}}else if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params.exceptionDetails);}});
function send(method,params={},sessionId){const id=++sequence;return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');});}
const deadline=setTimeout(()=>{console.error('Browser smoke timed out');chrome.kill('SIGTERM');process.exitCode=1;},60000);
try{
 const {targetId}=await send('Target.createTarget',{url:'about:blank'});const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});
 const call=(m,p)=>send(m,p,sessionId);await call('Page.enable');await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:process.argv[2]});
 let value;
 for(let i=0;i<100;i++){
  const result=await call('Runtime.evaluate',{expression:`typeof data!=='undefined'&&document.querySelectorAll('#neighbours tr').length?{classes:data.names.length,checkpoint:data.metadata.checkpoint,sha:data.metadata.checkpoint_sha256,points:projectionHits.length,projection:document.getElementById('projection-plane').value}:null`,returnByValue:true});
  value=result.result.value;if(value)break;await new Promise(r=>setTimeout(r,100));
 }
 if(!value||value.classes!==12632||value.points!==12632||!value.checkpoint.includes('epoch26')||value.projection!=='Angular t-SNE'||errors.length)throw Error(JSON.stringify({value,errors}));
 await call('Runtime.evaluate',{expression:`setWorkspaceView('projection')`});await new Promise(r=>setTimeout(r,300));
 const metrics=await call('Runtime.evaluate',{expression:`(async()=>{
 const samples=[],frameTimes=[];for(let frame=0;frame<20;frame++){
  await new Promise(requestAnimationFrame);projectionDrag={point:[600,300],view:{...projectionView},moved:true};
  const r=projectionCanvas.getBoundingClientRect(),start=performance.now();
  for(let i=0;i<30;i++)projectionCanvas.onpointermove({clientX:r.left+r.width/2+i,clientY:r.top+r.height/2});
  samples.push(performance.now()-start);projectionDrag=null;await new Promise(requestAnimationFrame);frameTimes.push(performance.now()-start);
 }
 projectionDrag=null;fitProjection(data.names.map((_,i)=>i));drawProjection();
 const originalFetch=window.fetch,pixel='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
 let calls=0,aborts=0,first=null;
 window.fetch=async(url,options={})=>{
  if(url==='/api/health')return {ok:true,json:async()=>({photo_api:true})};
  calls++;await new Promise((resolve,reject)=>{const timer=setTimeout(resolve,500);options.signal?.addEventListener('abort',()=>{clearTimeout(timer);aborts++;reject(new DOMException('Aborted','AbortError'));},{once:true});});
  return {ok:true,json:async()=>({class_id:String(url).split('/').pop(),display_name:'Controlled latency fixture',photos:[{image_path:pixel,creator:'Fixture',license:'Fixture',source:'https://example.org',occurrence_id:'1'}]})};
 };
 photoAlbums.clear();mapThumbImages.clear();mapThumbFailures.clear();$('map-photos').checked=true;$('map-photo-density').value='20';$('map-photo-push').checked=false;
 const loadStart=performance.now();drawProjection();
 const observe=setInterval(()=>{if(first===null&&mapThumbLayout.length)first=performance.now()-loadStart;},10);
 for(let i=0;i<3;i++){await new Promise(r=>setTimeout(r,250));projectionView.x+=.1;drawProjection();}
 let finished=false;
 for(let i=0;i<160;i++){await new Promise(r=>setTimeout(r,50));const counts=$('map-photo-status').textContent.match(/([0-9]+) images loaded · ([0-9]+) visible slots/);if(counts&&Number(counts[1])===Number(counts[2])&&Number(counts[1])>0){finished=true;break;}}
 clearInterval(observe);const loading={firstVisibleMs:first,elapsedMs:performance.now()-loadStart,visible:mapThumbLayout.length,calls,aborts,finished,fixture:'500 ms metadata; tiny local image; 3 navigation updates'};
 $('map-photos').checked=false;drawProjection();window.fetch=originalFetch;
 samples.sort((a,b)=>a-b);return {loading,frameTimes,samples,median:samples[10],p95:samples[Math.ceil(samples.length*.95)-1],classes:data.names.length,browser:navigator.userAgent};
})()`,returnByValue:true,awaitPromise:true});
 if(metrics.exceptionDetails)throw Error(JSON.stringify(metrics.exceptionDetails));
 writeFileSync(join(output,'performance.json'),JSON.stringify(metrics.result.value,null,2));
 console.log(JSON.stringify(metrics.result.value));
 const shot=await call('Page.captureScreenshot',{format:'png'});writeFileSync(join(output,'epoch26.png'),Buffer.from(shot.data,'base64'));
 console.log(JSON.stringify({value,javascriptErrors:errors.length}));await send('Browser.close');
}catch(e){console.error(e);process.exitCode=1;chrome.kill('SIGTERM');}finally{clearTimeout(deadline);setTimeout(()=>rmSync(profile,{recursive:true,force:true}),1000);}
