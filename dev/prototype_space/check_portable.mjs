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
 await call('Page.navigate',{url:process.argv[2].startsWith('http')?process.argv[2]:pathToFileURL(process.argv[2]).href});
 let ready=false;for(let i=0;i<100;i++){const r=await call('Runtime.evaluate',{expression:'typeof captureViewerState === "function" ? document.querySelectorAll("#neighbours tr").length : 0',returnByValue:true});if(r.result.value>0){ready=true;break;}await new Promise(r=>setTimeout(r,100));}
 if(!ready)throw Error('Inspector did not render: '+JSON.stringify(errors));
 const result=await call('Runtime.evaluate',{expression:`(async()=>{
 const checks=[];const check=(v,s)=>{if(!v)throw Error(s);checks.push(s)};
 check(typeof BrowserGBIF==='function','browser client available');
 check(!document.getElementById('gbif-names'),'one namespace setting');
 check(document.body.dataset.view==='projection','explore default');
 document.getElementById('activity-predict').click();
 check(!document.getElementById('inference-panel').hidden,'predict navigation');
 const state=captureViewerState();check(state.version===2,'versioned state');
 document.getElementById('activity-explore').click();
 check(document.body.dataset.view==='projection','explore navigation');
 let terminated=false;inferenceWorker={terminate(){terminated=true;}};resetInferenceModel();check(terminated&&inferenceManifest===null&&$('inference-image').disabled,'model reset cancels stale inference');
 const id=data.names[0];let requests=0;
 gbifClient.fetcher=async url=>{requests++;const taxon=url.includes('occurrence/search')?new URL(url).searchParams.get('taxonKey'):url.split('/').pop();return {ok:true,json:async()=>url.includes('occurrence/search')?{results:[{key:1,speciesKey:Number(taxon),media:[{type:'StillImage',identifier:location.origin+'/fixture.jpg',creator:'Fixture author',license:'CC BY'}]}]}:{key:Number(taxon),canonicalName:'Example '+taxon}};};
 $('class-id-namespace').value='gbif';$('class-id-namespace').onchange();
 check((await gbifClient.name(id)).display_name==='Example '+id,'browser names');
 const album=await referenceAlbum(id,new AbortController().signal);
 check(album.photos[0].creator==='Fixture author','browser photo attribution');
 const legacy=captureViewerState();legacy.version=1;delete legacy.controls['class-id-namespace'];legacy.controls['gbif-names']=true;
 applyViewerState(legacy);check($('class-id-namespace').value==='gbif','legacy namespace migration');
 await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
 let rejected=false;try{applyViewerState({...captureViewerState(),version:1,controls:null});}catch{rejected=true;}check(rejected,'malformed legacy state rejected');
 const before={...projectionView};const classes=[data.names,...[1,2].map(level=>[...new Set(data.groups.map(group=>group[level]))])];
 lastPrediction={result:{outputs:classes.map(names=>names.map((_,i)=>i===0?10:0))},m:{classes}};renderInferenceResults();
 await new Promise(resolve=>setTimeout(resolve,300));
 check($('inference-results').querySelectorAll('figure').length===5,'species prediction thumbnails');
 check($('inference-results').textContent.includes('Fixture author'),'prediction credits');
 check(JSON.stringify(before)===JSON.stringify(projectionView),'prediction labels preserve map transform');
 $('class-id-namespace').value='generic';$('class-id-namespace').onchange();
 const count=requests;await new Promise(resolve=>setTimeout(resolve,300));
 check(requests===count,'generic mode prevents new GBIF requests');
 check(!$('inference-results').querySelector('figure'),'generic prediction fallback');
 $('class-id-namespace').value='gbif';$('class-id-namespace').onchange();setWorkspaceView('predict');
 await new Promise(resolve=>setTimeout(resolve,300));
 return checks;
 })()`,returnByValue:true,awaitPromise:true});
 if(result.exceptionDetails)throw Error(JSON.stringify({exception:result.exceptionDetails,errors}));
 for(const [name,width,height] of [['desktop',1440,1000],['mobile',390,844]]){
  await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
  await new Promise(resolve=>setTimeout(resolve,500));
  const shot=await call('Page.captureScreenshot',{format:'png'});writeFileSync(join(output,name+'.png'),Buffer.from(shot.data,'base64'));
 }
 if(process.argv.includes('--live')){
  const live=await call('Runtime.evaluate',{expression:`(async()=>{
   const client=new BrowserGBIF();const name=await client.name('1775152');const album=await client.album('1775152');
   if(!album.photos.length)throw Error('No live reference images');
   const image=document.createElement('img');document.body.append(image);
   let loaded=false;for(const photo of album.photos){if(await loadReferenceImage(image,photo.image_path,new AbortController().signal)){loaded=true;break;}}
   image.remove();if(!loaded)throw Error('Live reference images unavailable');
   return {name:name.display_name,photos:album.photos.length,imageLoaded:loaded};
  })()`,awaitPromise:true,returnByValue:true});
  if(live.exceptionDetails)throw Error(JSON.stringify(live.exceptionDetails));
  console.log(JSON.stringify({liveGBIF:live.result.value}));
 }
 if(errors.length)throw Error(JSON.stringify(errors));
 const report={checks:result.result.value,errors:errors.length};writeFileSync(join(output,'portable-check.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
 await send('Browser.close');
}catch(e){console.error(e);process.exitCode=1;chrome.kill('SIGTERM');}finally{clearTimeout(deadline);setTimeout(()=>rmSync(profile,{recursive:true,force:true}),1000);}
