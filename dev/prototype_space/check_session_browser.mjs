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
 const persisted=await call('Runtime.evaluate',{expression:`(async()=>{
  for(let i=0;i<100&&!viewerServerToken;i++)await new Promise(r=>setTimeout(r,50));
  if(!viewerServerToken)throw Error('Server state API not available');
  setWorkspaceView('projection');selectClass(1);await new Promise(r=>setTimeout(r,100));
  projectionView.scale*=1.25;drawProjection();saveViewerState();await viewerStateWrites;
  const remote=await (await fetch('/api/view-state?key='+encodeURIComponent(viewerDocumentKey))).json();
  if(remote.state.view.selected!==data.names[1])throw Error('State not persisted by local server');
  localStorage.clear();restoringViewerState=true;
 })()`,awaitPromise:true});
 if(persisted.exceptionDetails)throw Error(JSON.stringify(persisted.exceptionDetails));
 await call('Page.reload');
 let restored=false;
 for(let i=0;i<100;i++){
  const result=await call('Runtime.evaluate',{expression:`typeof viewerServerToken!=='undefined'&&viewerServerToken&&selected===1&&document.body.dataset.view==='projection'`,returnByValue:true});
  if(result.result.value){restored=true;break;}await new Promise(r=>setTimeout(r,100));
 }
 if(!restored||errors.length)throw Error('Server-backed reload did not restore state: '+JSON.stringify(errors));
 console.log('Server-backed state restored after clearing browser storage');
 const shot=await call('Page.captureScreenshot',{format:'png'});writeFileSync(join(output,'epoch26.png'),Buffer.from(shot.data,'base64'));
 console.log(JSON.stringify({value,javascriptErrors:errors.length}));await send('Browser.close');
}catch(e){console.error(e);process.exitCode=1;chrome.kill('SIGTERM');}finally{clearTimeout(deadline);setTimeout(()=>rmSync(profile,{recursive:true,force:true}),1000);}
