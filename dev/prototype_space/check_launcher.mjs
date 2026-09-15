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
 const call=(m,p)=>send(m,p,sessionId);await call('Page.enable');await call('Runtime.enable');
 await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:process.argv[2]});
 const evaluate=async expression=>{const result=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;};
 for(let i=0;i<100;i++){if(await evaluate('!!document.getElementById("weights")'))break;await new Promise(r=>setTimeout(r,100));}
 const {root}=await call('DOM.getDocument');const {nodeId}=await call('DOM.querySelector',{nodeId:root.nodeId,selector:'#weights'});
 await call('DOM.setFileInputFiles',{nodeId,files:[resolve(process.argv[4])]});
 await evaluate('document.getElementById("open").requestSubmit()');
 let ready=false;
 for(let i=0;i<400;i++){
  if(await evaluate('!document.getElementById("view").hidden')){ready=true;break;}
  const text=await evaluate('document.getElementById("status").textContent');if(text.startsWith('Could not'))throw Error(text);
  await new Promise(r=>setTimeout(r,100));
 }
 if(!ready)throw Error('File picker did not finish generation');
 await evaluate('document.getElementById("view").click()');
 for(let i=0;i<100;i++){if(await evaluate('!!document.getElementById("projection-map")'))break;await new Promise(r=>setTimeout(r,100));}
 const result=await evaluate(`(async()=>{
  for(let i=0;i<50&&document.getElementById('open-model').hidden;i++)await new Promise(r=>setTimeout(r,50));
  return {classes:data.names.length,names:data.names,checkpoint:data.metadata.checkpoint,projection:document.getElementById('projection-plane').value,canOpenAnother:!document.getElementById('open-model').hidden,saveLink:!!document.querySelector('a[download]')};
 })()`);
 if(!result.canOpenAnother||!result.saveLink||result.classes<2||result.projection!=='Angular t-SNE')throw Error(JSON.stringify(result));
 const shot=await call('Page.captureScreenshot',{format:'png'});writeFileSync(join(output,'uploaded-model.png'),Buffer.from(shot.data,'base64'));
 await evaluate('document.getElementById("open-model").click()');
 for(let i=0;i<50;i++){if(await evaluate('!!document.getElementById("weights")'))break;await new Promise(r=>setTimeout(r,100));}
 if(!await evaluate('!!document.getElementById("weights")'))throw Error('Cannot return to file picker');
 if(errors.length)throw Error(JSON.stringify(errors));
 console.log(JSON.stringify({browserUpload:result,returnedToPicker:true,javascriptErrors:errors.length}));
 await send('Browser.close');
}catch(e){console.error(e);process.exitCode=1;chrome.kill('SIGTERM');}finally{clearTimeout(deadline);setTimeout(()=>rmSync(profile,{recursive:true,force:true}),1000);}
