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
const deadline=setTimeout(()=>{console.error('Browser smoke timed out');chrome.kill('SIGTERM');process.exitCode=1;},45000);
try{
 const {targetId}=await send('Target.createTarget',{url:'about:blank'});const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});
 const call=(m,p)=>send(m,p,sessionId);await call('Page.enable');await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:process.argv[2].startsWith('http')?process.argv[2]:pathToFileURL(process.argv[2]).href});
 let ready=false;for(let i=0;i<100;i++){const r=await call('Runtime.evaluate',{expression:'document.querySelectorAll("#neighbours tr").length',returnByValue:true});if(r.result.value>0){ready=true;break;}await new Promise(r=>setTimeout(r,100));}
 if(!ready)throw Error('Inspector did not render: '+JSON.stringify(errors));
 const result=await call('Runtime.evaluate',{expression:`(()=>{const checks=[];function check(v,s){if(!v)throw Error(s);checks.push(s)}
 check(document.querySelectorAll('#stats .stat').length===4,'production summary');
 check(projectionHits.length===data.names.length,'projection contains every class');
 check(document.getElementById('projection-quality').textContent.includes('neighbours retained'),'projection reports distortion');
 const projectionBefore=document.getElementById('projection-map').toDataURL();
 document.getElementById('projection-plane').selectedIndex=1;document.getElementById('projection-plane').dispatchEvent(new Event('change'));
 check(document.getElementById('projection-map').toDataURL()!==projectionBefore,'alternate PCA plane');
 document.getElementById('projection-plane').selectedIndex=0;document.getElementById('projection-plane').dispatchEvent(new Event('change'));
 const allScale=projectionView.scale;document.getElementById('projection-focus').click();check(projectionView.scale>=allScale,'fit selected projected neighbourhood');
 document.getElementById('projection-reset').click();
 const canvas=document.getElementById('projection-map'),bounds=canvas.getBoundingClientRect();
 const oldScale=projectionView.scale;canvas.dispatchEvent(new WheelEvent('wheel',{clientX:bounds.left+bounds.width/2,clientY:bounds.top+bounds.height/2,deltaY:-150,cancelable:true}));check(projectionView.scale>oldScale,'projection zoom');
 const oldX=projectionView.x;projectionDrag={point:projectionMouse({clientX:bounds.left+100,clientY:bounds.top+100}),view:{...projectionView},moved:false};canvas.onpointermove({clientX:bounds.left+150,clientY:bounds.top+100});canvas.onpointerup({pointerId:1});check(projectionView.x!==oldX,'projection pan');
 document.getElementById('projection-reset').click();
 const target=projectionHits.findIndex(([x,y],i)=>i!==selected&&x>20&&x<1180&&y>20&&y<580&&projectionHit([x,y])===i);check(target>=0,'projected class is selectable');
 const [px,py]=projectionHits[target];const event={button:0,pointerId:1,clientX:bounds.left+px*bounds.width/1200,clientY:bounds.top+py*bounds.height/600};projectionDrag={point:projectionMouse(event),view:{...projectionView},moved:false};canvas.onpointerup(event);
 check(selected===target&&document.getElementById('selected-name').textContent===data.names[target],'projection selection updates inspector');

 check(document.querySelectorAll('#neighbours tr').length===data.stats.k,'production neighbours');
 check(Number.isFinite(Number(document.querySelector('#neighbours tr').children[3].textContent)),'finite float32 log tail in neighbour table');
 document.getElementById('range-preset').value='full';document.getElementById('range-preset').dispatchEvent(new Event('change'));const logImage=document.getElementById('matrix-min').toDataURL();document.getElementById('global-metric').value='baseline';document.getElementById('global-metric').dispatchEvent(new Event('change'));check(document.getElementById('matrix-min').toDataURL()!==logImage,'legacy versus log-domain global matrix');document.getElementById('global-metric').value='logtail';document.getElementById('global-metric').dispatchEvent(new Event('change'));
 document.getElementById('metric').value='logtail';document.getElementById('metric').dispatchEvent(new Event('change'));check(document.getElementById('local-scale').textContent.includes('−log₁₀ tail'),'stable log-tail matrix');
 const rawFirst=globalValues.logmin.find(Number.isFinite);document.getElementById('range-preset').value='focus';document.getElementById('range-preset').dispatchEvent(new Event('change'));check(document.getElementById('matrix-min').toDataURL()!==logImage,'focused versus full colour range');check(globalValues.logmin.find(Number.isFinite)===rawFirst,'display clipping preserves raw values');document.getElementById('range-preset').value='robust';document.getElementById('range-preset').dispatchEvent(new Event('change'));check(Number(document.getElementById('range-low').value)>data.matrix_log10_floor,'observed percentile window');document.getElementById('range-low').value='1';document.getElementById('range-high').value='0';document.getElementById('apply-range').click();check(document.getElementById('range-status').textContent.includes('lower < upper'),'invalid display range feedback');document.getElementById('range-preset').value='focus';document.getElementById('range-preset').dispatchEvent(new Event('change'));

 document.getElementById('crowded').click();check(document.getElementById('selected-name').textContent===data.names[data.profiles.reduce((best,p,i)=>p[0]>data.profiles[best][0]?i:best,0)],'strongest neighbour selection');
 document.getElementById('metric').value='z';document.getElementById('metric').dispatchEvent(new Event('change'));check(document.getElementById('local-scale').textContent.includes('z '),'z metric toggle');
 document.getElementById('tree-locate').click();check(document.querySelectorAll('#tree .tree-label').length<=32,'tree locate and collapse');
 document.getElementById('tree-root').click();const collapsed=[...document.querySelectorAll('#tree .tree-label')].find(n=>n.textContent.includes('expand'));if(collapsed){collapsed.dispatchEvent(new MouseEvent('click'));check(size(treeFocus)<data.names.length,'subtree drilldown');}
 const searchId=data.names[1];document.getElementById('search').value=searchId;document.getElementById('find').click();check(document.getElementById('selected-name').textContent===searchId,'exact class search');
 const before=document.getElementById('selected-name').textContent;document.querySelector('#neighbours button').click();check(document.getElementById('selected-name').textContent!==before,'neighbour navigation');
 document.getElementById('search').value='not-a-class';document.getElementById('find').click();check(document.getElementById('search-status').textContent.includes('No matching'),'missing class feedback');
 for(const name of ['Independent unit vectors','Four planted groups','Algebraic edge cases']){document.getElementById('case').value=name;document.getElementById('case').dispatchEvent(new Event('change'));check(document.querySelectorAll('#neighbours tr').length>0,name);}
 document.getElementById('case').value='Production checkpoint';document.getElementById('case').dispatchEvent(new Event('change'));
 return checks;})()`,returnByValue:true});
 if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));if(errors.length)throw Error(JSON.stringify(errors));
 await call('Runtime.evaluate',{expression:'document.getElementById("projection-panel").scrollIntoView()'});
 const projectionShot=await call('Page.captureScreenshot',{format:'png'});writeFileSync(join(output,'projection.png'),Buffer.from(projectionShot.data,'base64'));
 await call('Runtime.evaluate',{expression:'window.scrollTo(0,0)'});
 const shot=await call('Page.captureScreenshot',{format:'png'});writeFileSync(join(output,'overview.png'),Buffer.from(shot.data,'base64'));
 await call('Runtime.evaluate',{expression:'document.getElementById("inspector").scrollIntoView()'});
 const local=await call('Page.captureScreenshot',{format:'png'});writeFileSync(join(output,'neighbourhood.png'),Buffer.from(local.data,'base64'));
 await call('Runtime.evaluate',{expression:'document.getElementById("global-metric").closest(".panel").scrollIntoView()'});const globalShot=await call('Page.captureScreenshot',{format:'png'});writeFileSync(join(output,'matrix.png'),Buffer.from(globalShot.data,'base64'));const report={checks:result.result.value,javascript_errors:errors.length};writeFileSync(join(output,'browser-check.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 const photos=await call('Runtime.evaluate',{expression:`(async()=>{
 const checks=[];const check=(value,label)=>{if(!value)throw Error(label);checks.push(label)};
 const originalFetch=window.fetch;
 const pixel='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
 window.fetch=async url=>({ok:true,json:async()=>url==='/api/health'?{photo_api:true}:{class_id:url.split('/').pop(),accepted_id:url.split('/').pop(),display_name:'Fixture class',photos:[0,1].map(i=>({image_path:pixel,creator:'Fixture photographer',license:'Fixture media license',source:'https://www.gbif.org',occurrence_id:'1'}))}});
 document.getElementById('photo-enabled').checked=true;await renderPhotos();
 check(document.querySelectorAll('.photo-card').length===data.stats.k+1,'photo neighbourhood cards');
 check(document.querySelector('.photo-card img').naturalWidth===1,'photo image loaded');
 check(document.querySelector('.photo-credit').textContent.includes('Fixture photographer'),'photo attribution');
 const next=document.querySelector('.photo-card button:not(.photo-open)');await next.onclick();
 check(document.querySelector('.photo-card .caption + .caption').textContent.includes('2 / 2'),'cycle class example');
 const expected=data.neighbours[selected][0];document.querySelectorAll('.photo-open')[1].click();
 check(selected===expected,'photo navigates to neighbour');
 document.getElementById('case').value='Algebraic edge cases';document.getElementById('case').dispatchEvent(new Event('change'));await renderPhotos();
 check(document.getElementById('photo-status').textContent.includes('No online lookup'),'synthetic case skips photo lookup');
 document.getElementById('photo-enabled').checked=false;await renderPhotos();
 document.getElementById('case').value='Production checkpoint';document.getElementById('case').dispatchEvent(new Event('change'));
 const fixturePoints=[[60,60],[100,60],[260,100],[450,50]], config={width:400,height:200,size:64,density:200,overlap:0};
 check(thumbnailLayout(fixturePoints,[0,1,2,3],config).boxes.map(b=>b.id).join(',')==='0,2','thumbnail viewport and overlap culling');
 check(thumbnailLayout(fixturePoints,[0,1,2,3],{...config,overlap:.5}).boxes.length===3,'thumbnail allowable overlap');
 check(thumbnailLayout(fixturePoints,[0,1,2,3],{...config,density:5}).boxes.length===1,'thumbnail density budget');
 check(thumbnailLayout(fixturePoints,[0,1,2,3],{...config,size:140}).boxes.length===1,'thumbnail size changes culling');
 document.getElementById('map-photos').checked=true;scheduleMapThumbnails();clearTimeout(mapThumbTimer);await renderMapThumbnails();
 check(document.querySelectorAll('.map-thumb').length>0,'map thumbnails load');
 check([...document.querySelectorAll('.map-thumb img')].every(img=>img.naturalWidth>0),'map thumbnails decode');
 const first=mapThumbLayout[0];check(mapThumbnailHit([first.x*1200/projectionCanvas.getBoundingClientRect().width,first.y*600/projectionCanvas.getBoundingClientRect().height])===first.id,'thumbnail hit testing');
 showMapPhotoCredit(first.id);check(document.getElementById('map-photo-credit').textContent.includes('Fixture photographer'),'map thumbnail attribution');
 document.querySelector('.map-thumb').click();check(selected===first.id,'thumbnail selection updates class');
 document.getElementById('map-photos').checked=false;scheduleMapThumbnails();check(document.querySelectorAll('.map-thumb').length===0,'map thumbnail toggle clears images');

 const fixtureFetch=window.fetch;let release;
 window.fetch=()=>new Promise(resolve=>{release=()=>resolve({ok:true,json:async()=>({photo_api:true})})});
 document.getElementById('map-photos').checked=true;scheduleMapThumbnails();clearTimeout(mapThumbTimer);const pendingMap=renderMapThumbnails();
 document.getElementById('map-photos').checked=false;scheduleMapThumbnails();release();await pendingMap;
 check(document.querySelectorAll('.map-thumb').length===0,'cancelled thumbnail load stays hidden');
 let syntheticRequests=0;window.fetch=async()=>{syntheticRequests++;throw Error('Unexpected synthetic lookup')};
 document.getElementById('case').value='Algebraic edge cases';document.getElementById('case').dispatchEvent(new Event('change'));document.getElementById('map-photos').checked=true;scheduleMapThumbnails();
 check(syntheticRequests===0&&document.getElementById('map-photo-status').textContent.includes('Synthetic'),'synthetic map skips photo requests');
 document.getElementById('map-photos').checked=false;scheduleMapThumbnails();window.fetch=fixtureFetch;
 window.fetch=originalFetch;
 document.getElementById('case').value='Production checkpoint';document.getElementById('case').dispatchEvent(new Event('change'));
 return checks;
 })()`,awaitPromise:true,returnByValue:true});
 if(photos.exceptionDetails)throw Error(JSON.stringify(photos.exceptionDetails));
 report.checks.push(...photos.result.value);report.javascript_errors=errors.length;
 writeFileSync(join(output,'browser-check.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({photo_checks:photos.result.value,javascript_errors:errors.length}));
 await send('Browser.close');
}catch(e){console.error(e);process.exitCode=1;chrome.kill('SIGTERM');}finally{clearTimeout(deadline);setTimeout(()=>rmSync(profile,{recursive:true,force:true}),1000);}
