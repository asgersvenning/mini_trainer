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
 const forcePair=(size,gap,dy=0)=>[{id:0,x:200,y:200,anchorX:200,anchorY:200,width:size,height:size},{id:1,x:200+gap,y:200+dy,anchorX:200+gap,anchorY:200+dy,width:size,height:size}];
 const distant=forcePair(64,70);stepThumbnailForce(distant,1000,800,128,0);
 check(distant.every(b=>b.vx===0&&b.vy===0),'separate thumbnail boxes exert no repulsion beyond two-pixel clearance');
 const small=forcePair(32,50),large=forcePair(64,50);stepThumbnailForce(small,1000,800,128,0);stepThumbnailForce(large,1000,800,128,0);
 check(small.every(b=>b.vx===0)&&large.some(b=>Math.abs(b.vx)>0),'contact uses thumbnail image size');
 const corners=forcePair(64,67,67);stepThumbnailForce(corners,1000,800,128,0);
 check(corners.every(b=>b.vx===0&&b.vy===0),'diagonally separated rectangles do not repel as oversized ellipses');
 const translated=forcePair(64,50),base=forcePair(64,50);for(const b of translated){b.x+=100;b.anchorX+=100;b.y+=50;b.anchorY+=50;}stepThumbnailForce(base,1000,800,128,0);stepThumbnailForce(translated,1000,800,128,0);
 check(base.every((b,i)=>Math.abs(b.vx-translated[i].vx)<1e-10&&Math.abs(b.vy-translated[i].vy)<1e-10),'screen-space forces are invariant to viewport translation');

 check(thumbnailLayout(fixturePoints,[0,1,2,3],config).boxes.map(b=>b.id).join(',')==='0,2','thumbnail viewport and overlap culling');
 check(thumbnailLayout(fixturePoints,[0,1,2,3],{...config,overlap:.5}).boxes.length===3,'thumbnail allowable overlap');
 check(thumbnailLayout(fixturePoints,[0,1,2,3],{...config,density:5}).boxes.length===1,'thumbnail density budget');
 check(thumbnailLayout(fixturePoints,[0,1,2,3],{...config,size:140}).boxes.length===1,'thumbnail size changes culling');
 const compact=thumbnailLayout([[32,32],[96,32]],[0,1],{...config,width:128,height:64,labels:false,density:2000});
 check(compact.boxes.length===2&&compact.boxes.every(b=>b.width===64&&b.height===64),'image-only culling has no label or padding footprint');
 const wideFootprint=thumbnailFootprint(80,false,2),tallFootprint=thumbnailFootprint(80,true,.5);
 check(wideFootprint.width===80&&wideFootprint.height===40&&tallFootprint.width===48&&tallFootprint.height===104,'original aspect footprints include image proportions and optional labels');
 const aspectSlots=thumbnailLayout([[40,20],[40,60]],[0,1],{width:80,height:80,size:80,density:1000,overlap:0,labels:false,aspects:[2,2]});
 check(aspectSlots.boxes.length===2,'aspect-aware culling uses rectangular footprints');
 check(thumbnailLayout([[100,100],[130,100]],[0,1],{width:200,height:200,size:64,density:200,overlap:.75,unknownOverlap:0,labels:false}).boxes.length===1,'unknown aspect ratios reserve nonoverlapping boxes when forces are disabled');
 check(thumbnailLayout([[100,100],[100,100]],[0,1],{width:200,height:200,size:64,density:200,overlap:.5,labels:false,aspects:[4,1]}).boxes.length===1,'overlap limit protects the smaller rectangle from being covered');

 const originalPoints=JSON.stringify(fixturePoints),pushed=thumbnailLayout(fixturePoints,[0,1,2,3],{...config,overlap:.5,push:true});
 check(pushed.boxes.length===3&&pushed.boxes.some(b=>b.x!==b.anchorX||b.y!==b.anchorY),'push apart retains and separates overlapping candidates');
 const separate=boxes=>boxes.every((b,i)=>boxes.slice(i+1).every(other=>thumbnailSharedArea(b,other)<1e-8));
 check(separate(pushed.boxes),'pushed rectangles do not overlap');
 check(pushed.boxes.every(b=>b.left>=0&&b.top>=0&&b.left+b.width<=config.width&&b.top+b.height<=config.height&&Math.hypot(b.x-b.anchorX,b.y-b.anchorY)<=2*config.size+1e-8),'push apart respects viewport and displacement bounds');
 check(JSON.stringify(fixturePoints)===originalPoints&&JSON.stringify(thumbnailLayout(fixturePoints,[0,1,2,3],{...config,overlap:.5,push:true}))===JSON.stringify(pushed),'push apart is deterministic and preserves source points');
 const dense=Array.from({length:60},(_,i)=>[40+(i%10)*18,40+Math.floor(i/10)*18]);
 const tight=thumbnailLayout(dense,dense.map((_,i)=>i),{...config,width:260,height:180,labels:false,overlap:.75,push:true,density:2000});
 check(tight.boxes.length>0&&tight.culled===0&&tight.boxes.every(b=>b.left>=0&&b.top>=0&&b.left+b.width<=260&&b.top+b.height<=180),'crowded boundary fixture retains all admitted boxes within bounds');
 document.getElementById('map-photos').checked=true;scheduleMapThumbnails();clearTimeout(mapThumbTimer);await renderMapThumbnails();
 check(document.querySelectorAll('.map-thumb').length>0,'map thumbnails load');
 check([...document.querySelectorAll('.map-thumb img')].every(img=>img.naturalWidth>0),'map thumbnails decode');
 const first=mapThumbLayout[0];check(mapThumbnailHit([first.x*1200/projectionCanvas.getBoundingClientRect().width,first.y*600/projectionCanvas.getBoundingClientRect().height])===first.id,'thumbnail hit testing');
 showMapPhotoCredit(first.id);check(document.getElementById('map-photo-credit').textContent.includes('Fixture photographer'),'map thumbnail attribution');
 document.querySelector('.map-thumb').click();check(selected===first.id,'thumbnail selection updates class');
 const coordinatesBefore=JSON.stringify(projectionHits);
 document.getElementById('map-photo-labels').checked=false;document.getElementById('map-photo-push').checked=true;document.getElementById('map-photo-overlap').value='75';document.getElementById('map-photo-density').value='200';
 scheduleMapThumbnails();clearTimeout(mapThumbTimer);await renderMapThumbnails();
 check(document.querySelectorAll('.map-thumb').length>0&&[...document.querySelectorAll('.map-thumb')].every(b=>!b.querySelector('span')&&getComputedStyle(b).padding==='0px'&&getComputedStyle(b).borderWidth==='0px'),'image-only toggle removes labels and whitespace');
 check(mapThumbRest&&mapThumbRest.pairs===thumbnailEnergy(mapThumbLayout).pairs&&document.getElementById('map-photo-status').textContent.includes('overlapping pairs retained'),'settled status reports residual overlap without hiding it');
 check(JSON.stringify(projectionHits)===coordinatesBefore,'thumbnail relaxation preserves projected coordinates');
 const displaced=mapThumbLayout.find(b=>Math.hypot(b.x-b.anchorX,b.y-b.anchorY)>=.5);
 check(displaced&&document.querySelectorAll('.map-thumb-tether').length>0,'displaced photos retain visible anchor markers');
 const anchorNodes=[...document.querySelectorAll('.map-thumb-tether')];
 check(anchorNodes.every(n=>getComputedStyle(n).display==='none'),'thumbnail anchors default to hidden');
 const layoutBeforeToggle=mapThumbLayout;
 document.getElementById('map-photo-anchors').checked=true;document.getElementById('map-photo-anchors').dispatchEvent(new Event('input'));
 check(anchorNodes.every(n=>getComputedStyle(n).display!=='none')&&mapThumbLayout===layoutBeforeToggle,'anchor toggle does not restart or reload thumbnails');
 document.getElementById('map-photo-anchors').checked=false;setThumbnailAnchors();
 const beforeView={...projectionView},beforeNodes=new Map(mapThumbLayout.map(b=>[b.id,{button:b.button,x:b.x,anchorX:b.anchorX,offset:b.x-b.anchorX,width:b.width}]));
 projectionView.x+=2/projectionView.scale;projectionView.scale*=1.01;drawProjection();clearTimeout(mapThumbTimer);
 check(mapThumbLayout.length>0&&mapThumbLayout.every(b=>b.button===beforeNodes.get(b.id).button&&Math.abs(b.x-b.anchorX-beforeNodes.get(b.id).offset)<1e-8&&b.width===beforeNodes.get(b.id).width),'pan and zoom preserve loaded DOM nodes, pixel offsets and image dimensions');
 const fetchBefore=window.fetch;let refreshRequests=0;window.fetch=async(...args)=>{refreshRequests++;return fetchBefore(...args);};await renderMapThumbnails();window.fetch=fetchBefore;
 check(mapThumbLayout.some(b=>beforeNodes.get(b.id)?.button===b.button),'settled viewport refresh reuses surviving thumbnail nodes');
 projectionView=beforeView;drawProjection();clearTimeout(mapThumbTimer);await renderMapThumbnails();

 const motionConfig={width:projectionCanvas.getBoundingClientRect().width,height:projectionCanvas.getBoundingClientRect().height,size:64};
 for(const b of mapThumbLayout){b.x=b.anchorX;b.y=b.anchorY;b.left=b.x-b.width/2;b.top=b.y-b.height/2;b.vx=0;b.vy=0;}
 const startPositions=mapThumbLayout.map(b=>[b.x,b.y]);const moving=animateThumbnails(motionConfig);
 await new Promise(resolve=>setTimeout(resolve,120));
 check(mapThumbMotion!==null&&mapThumbLayout.some((b,i)=>Math.hypot(b.x-startPositions[i][0],b.y-startPositions[i][1])>.1),'force layout advances visibly over animation frames');
 check(mapThumbLayout.every(b=>Math.abs(parseFloat(b.button.style.left)-b.x)<.01&&Math.abs(parseFloat(b.button.style.top)-b.y)<.01),'animated images and hit rectangles move together');
 await moving;check(mapThumbMotion===null&&mapThumbLayout.length===startPositions.length,'solver stops without removing thumbnails');
 const originalMatchMedia=window.matchMedia;window.matchMedia=()=>({matches:true});await animateThumbnails(motionConfig);window.matchMedia=originalMatchMedia;
 check(mapThumbMotion===null&&mapThumbLayout.length===startPositions.length,'reduced motion preserves the same thumbnail population');
 const savedLayout=mapThumbLayout;
 const makeCrowded=()=>[0,1].map(i=>{
  const b={id:i,x:32,y:32,anchorX:32,anchorY:32,left:0,top:0,width:64,height:64,vx:0,vy:0};
  b.button=savedLayout[0].button.cloneNode(true);b.tether=thumbnailTether(b,true);document.getElementById('map-thumbnail-layer').append(b.button,b.tether);return b;
 });
 mapThumbLayout=makeCrowded();const crowded=mapThumbLayout;await animateThumbnails({width:64,height:64,size:64});
 check(mapThumbLayout===crowded&&crowded.every(b=>b.button.isConnected&&b.button.style.opacity===''),'impossible packing keeps both thumbnails visible at rest');
 check(mapThumbRest.reason==='resting'&&mapThumbRest.pairs===1&&mapThumbRest.steps<480,'constrained equilibrium stops on measured stability and reports overlap');
 await new Promise(resolve=>setTimeout(resolve,300));
 check(crowded.every(b=>b.button.isConnected),'no deferred settling-time removal');
 for(const b of crowded){b.button.remove();b.tether.remove();}mapThumbLayout=savedLayout;paintThumbnailMotion();
 const fading={...savedLayout[0],id:-100,button:savedLayout[0].button.cloneNode(true),tether:null};
 document.getElementById('map-thumbnail-layer').append(fading.button);
 // Exercise viewport culling with a valid, fully offscreen projected position.
 mapThumbLayout=[{...fading,id:0}];reanchorThumbnails([[-1000,-1000]],motionConfig.width,motionConfig.height);
 check(mapThumbLayout.length===0&&fading.button.isConnected&&fading.button.disabled&&fading.button.style.pointerEvents==='none','viewport culling removes hit targets immediately while keeping the exit visible');
 check(fading.button.getAnimations().some(a=>a.effect.getTiming().duration===100),'culled thumbnail gets a fast exit animation');
 await new Promise(resolve=>setTimeout(resolve,180));
 check(!fading.button.isConnected,'culled thumbnail is removed after fading');
 mapThumbLayout=savedLayout;paintThumbnailMotion();
 const cancelled=animateThumbnails(motionConfig);stopThumbnailMotion();await cancelled;
 check(mapThumbMotion===null,'force animation cancellation resolves pending work');
 document.getElementById('map-photos').checked=false;scheduleMapThumbnails();document.getElementById('map-photos').checked=true;document.getElementById('map-photo-density').value='5';mapThumbImages.clear();
 const originalLoad=loadReferenceImage;let activeImages=0,maxImages=0;
 loadReferenceImage=async(...args)=>{activeImages++;maxImages=Math.max(maxImages,activeImages);await new Promise(resolve=>setTimeout(resolve,100));const result=await originalLoad(...args);activeImages--;return result;};
 scheduleMapThumbnails();clearTimeout(mapThumbTimer);const delayed=renderMapThumbnails();await new Promise(resolve=>setTimeout(resolve,40));
 check(activeImages>1&&activeImages<=4&&mapThumbLayout.length===0&&mapThumbMotion===null,'bounded concurrent image loads do not move unloaded thumbnails');
 await delayed;loadReferenceImage=originalLoad;
 check(maxImages<=4&&mapThumbLayout.length>0&&mapThumbLayout.every(b=>b.button.querySelector('img').naturalWidth>0),'only visible decoded images enter force layout');
 const wideImage='data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="120" height="60"><rect width="120" height="60" fill="teal"/></svg>');
 const portraitImage='data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="60" height="120"><rect width="60" height="120" fill="orange"/></svg>');
 let albumIndex=0;for(const album of photoAlbums.values()){for(const photo of album.photos)photo.image_path=albumIndex%2?wideImage:portraitImage;albumIndex++;}
 document.getElementById('map-photo-aspect').checked=true;document.getElementById('map-photo-aspect').dispatchEvent(new Event('input'));clearTimeout(mapThumbTimer);await renderMapThumbnails();
 check(mapThumbLayout.length>0&&mapThumbLayout.every(b=>{const image=b.button.querySelector('img'),rect=image.getBoundingClientRect();return Math.abs(rect.width/rect.height-image.naturalWidth/image.naturalHeight)<.002&&Math.abs(b.width-rect.width)<.02&&Math.abs(b.height-rect.height)<.02;}),'original aspect option displays uncropped images with matching collision boxes');
 document.getElementById('map-photo-aspect').checked=false;document.getElementById('map-photo-aspect').dispatchEvent(new Event('input'));clearTimeout(mapThumbTimer);await renderMapThumbnails();
 check(mapThumbLayout.every(b=>Math.abs(b.width-b.height)<.01&&getComputedStyle(b.button.querySelector('img')).objectFit==='cover'),'square crop remains selectable');



 const currentBox=mapThumbLayout[0];check(mapThumbnailHit([currentBox.x*1200/projectionCanvas.getBoundingClientRect().width,currentBox.y*600/projectionCanvas.getBoundingClientRect().height])===currentBox.id,'displaced thumbnail hit testing');
 document.getElementById('map-photo-labels').checked=true;document.getElementById('map-photo-push').checked=false;document.getElementById('map-photo-overlap').value='0';document.getElementById('map-photo-density').value='40';
 scheduleMapThumbnails();clearTimeout(mapThumbTimer);await renderMapThumbnails();
 check([...document.querySelectorAll('.map-thumb')].every(b=>b.querySelector('span'))&&document.querySelectorAll('.map-thumb-tether').length===0,'label and fixed-position modes restore');
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
 const focusChecks=[];
 for(const [width,height] of [[1800,900],[900,1500],[390,844]]){
  await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
  const focused=await call('Runtime.evaluate',{expression:`(async()=>{
   const checks=[],check=(v,label)=>{if(!v)throw Error(label);checks.push(label)};
   const before={selected,treeFocus,plane:$('projection-plane').value,x:projectionView.x,y:projectionView.y};
   setWorkspaceView('projection');await new Promise(r=>setTimeout(r,250));
   const rect=projectionCanvas.getBoundingClientRect();
   check(! $('inspector').hidden&&!$('photo-panel').hidden&&$('tree-panel').hidden,'projection groups related panels');
   check(rect.width>innerWidth*.8&&rect.height>=300,'projection uses viewport width and height');
   check(Math.abs(rect.width/1200-rect.height/projectionHeight)<1e-8,'projection preserves equal screen scale');
   check(selected===before.selected&&treeFocus===before.treeFocus&&$('projection-plane').value===before.plane,'focus preserves selection tree and plane');
   check(projectionView.x===before.x&&projectionView.y===before.y,'focus preserves projection center');
   const origin=projectionPoint([projectionView.x,projectionView.y]);
   const hit=projectionMouse({clientX:rect.left+origin[0]*rect.width/1200,clientY:rect.top+origin[1]*rect.height/projectionHeight});
   check(Math.hypot(hit[0]-origin[0],hit[1]-origin[1])<1e-8,'pointer conversion follows focused aspect ratio');
   setWorkspaceView('tree');await new Promise(r=>setTimeout(r,100));
   check(!$('tree-panel').hidden&&!$('inspector').hidden&&$('projection-panel').hidden,'tree groups selected-class context');
   check($('tree').viewBox.baseVal.width>=760,'tree preserves readable labels');
   check(document.documentElement.scrollWidth<=innerWidth+1,'focused layout avoids page-wide horizontal scrolling');
   setWorkspaceView('matrix');check(!$('matrix-panel').hidden&&!$('inspector').hidden&&$('tree-panel').hidden,'matrix groups local inspection');
   setWorkspaceView('all');await new Promise(r=>setTimeout(r,100));
   check([...document.querySelectorAll('main .panel')].every(p=>!p.hidden),'all views restored');
   return checks;
  })()`,returnByValue:true,awaitPromise:true});
  if(focused.exceptionDetails)throw Error(JSON.stringify(focused.exceptionDetails));
  focusChecks.push({width,height,checks:focused.result.value});
 }
 if(errors.length)throw Error(JSON.stringify(errors));
 report.focus_checks=focusChecks;report.javascript_errors=errors.length;
 writeFileSync(join(output,'browser-check.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({focusChecks,javascript_errors:errors.length}));
 await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:2,mobile:false});
 const enhanced=await call('Runtime.evaluate',{expression:`(async()=>{
  const checks=[],check=(v,label)=>{if(!v)throw Error(label);checks.push(label)};
  setWorkspaceView('projection');await new Promise(r=>setTimeout(r,200));
  const before={selected,plane:$('projection-plane').value,x:projectionView.x,y:projectionView.y,names:JSON.stringify(data.names)};
  $('class-names').checked=true;
  const local=new Map([[selected,'Selected species'],[data.neighbours[selected][0],'Neighbour species']]);importedNames.set(data,local);refreshClassLabels();
  check($('selected-name').textContent.includes('Selected species'),'selected class shows optional name and ID');
  check($('neighbours').textContent.includes('Neighbour species'),'neighbour table resolves names');
  check($('local').textContent.includes('Neighbour species'),'matrix axes and hover resolve names');
  check($('profile').textContent.includes('Neighbour species'),'profile retains rank with neighbour name');
  $('search').value='Neighbour species';$('find').click();check(selected===data.neighbours[before.selected][0],'manual selector searches resolved names');selectClass(before.selected);
  treeFocus=selected;drawTree();check($('tree').textContent.includes('Selected species'),'dendrogram leaves resolve names');
  check(JSON.stringify(data.names)===before.names,'names never replace checkpoint IDs');
  $('score-display').value='tail';$('score-display').onchange();
  check($('score-column').textContent.includes('Φ')&&$('metric').value==='tail','probability toggle updates table and matrix');
  check(!$('score-note').hidden&&$('local-scale').textContent.includes('Upper-tail'),'probability reference and logarithmic colours disclosed');
  check($('profile').textContent.includes('Upper-tail'),'profile probability axis');
  $('score-display').value='z';$('score-display').onchange();check($('score-column').textContent==='z','z display restores');
  const original=window.fetch,ids=[selected,...data.neighbours[selected]],pending=[];let inFlight=0,peak=0;
  importedNames.delete(data);resolvedNames.clear();nameFailures.clear();
  window.fetch=async url=>{if(String(url).startsWith('/api/taxon/')){inFlight++;peak=Math.max(peak,inFlight);pending.push(url);await new Promise(r=>setTimeout(r,30));inFlight--;return {ok:true,json:async()=>({class_id:String(url).split('/').pop(),display_name:'Async taxon '+String(url).split('/').pop()})};}return original(url);};
  $('gbif-names').checked=true;requestClassNames();
  for(let i=0;i<60&&!resolvedNames.size;i++)await new Promise(r=>setTimeout(r,30));
  await new Promise(r=>setTimeout(r,200));
  check(resolvedNames.size>0&&pending.every(url=>url.startsWith('/api/taxon/')),'asynchronous names need no photo lookup');
  check(peak===1&&pending.length<30,'name lookup is bounded and prioritizes a small visible set');
  $('class-names').checked=false;$('gbif-names').checked=false;await new Promise(r=>setTimeout(r,100));window.fetch=original;refreshClassLabels();
  check($('selected-name').textContent===data.names[selected],'IDs restore without losing selected class');
  const rect=projectionCanvas.getBoundingClientRect();check(projectionCanvas.width===Math.round(rect.width*devicePixelRatio)&&projectionCanvas.height===Math.round(rect.height*devicePixelRatio),'canvas backing matches physical screen pixels');
  check(rect.height>=300,'focused map retains vertical workspace');
  const geometry=()=>{const r=projectionCanvas.getBoundingClientRect();return JSON.stringify([r.x,r.y,r.width,r.height,projectionView.x,projectionView.y,projectionView.scale]);};
  const stableGeometry=geometry(),savedCredit=$('map-photo-credit').innerHTML;
  for(const expanded of [false,true]){
   $('projection-details').open=expanded;
   for(const text of ['', 'Long taxon name and attribution '.repeat(100),'Short credit']){
    $('map-photo-credit').textContent=text;$('map-photo-status').textContent=text;
    await new Promise(r=>setTimeout(r,60));
    check(geometry()===stableGeometry,'hover and status preserve map geometry with details '+expanded);
   }
  }
  $('thumbnail-options').open=true;await new Promise(r=>setTimeout(r,60));
  check(geometry()===stableGeometry,'thumbnail settings overlay preserves map geometry');
  $('thumbnail-options').open=false;$('projection-details').open=false;$('map-photo-credit').innerHTML=savedCredit;
  check(rect.height>innerHeight*.75,'focused map uses over three quarters of landscape height');
  return checks;
 })()`,returnByValue:true,awaitPromise:true});
 if(enhanced.exceptionDetails)throw Error(JSON.stringify(enhanced.exceptionDetails));
 console.log(JSON.stringify({enhancedChecks:enhanced.result.value,javascript_errors:errors.length}));
 await send('Browser.close');
}catch(e){console.error(e);process.exitCode=1;chrome.kill('SIGTERM');}finally{clearTimeout(deadline);setTimeout(()=>rmSync(profile,{recursive:true,force:true}),1000);}
