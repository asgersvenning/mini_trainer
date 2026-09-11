import {spawn} from 'node:child_process';
import {mkdtempSync,writeFileSync,mkdirSync} from 'node:fs';
if(!process.env.CHROME_PATH||!process.argv[2]||!process.argv[3])throw Error('Usage: CHROME_PATH=/path/to/chrome node check_inference.mjs URL IMAGE_PATH');
const chrome=spawn(process.env.CHROME_PATH,['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-first-run','--disable-background-networking','--remote-debugging-pipe','--user-data-dir='+mkdtempSync('/tmp/browser-inference-check-')],{stdio:['ignore','ignore','ignore','pipe','pipe']});
let serial=0,buffer=Buffer.alloc(0);const pending=new Map(),errors=[];
chrome.stdio[4].on('data',chunk=>{buffer=Buffer.concat([buffer,chunk]);let end;while((end=buffer.indexOf(0))!==-1){const msg=JSON.parse(buffer.subarray(0,end));buffer=buffer.subarray(end+1);if(msg.id){const p=pending.get(msg.id);pending.delete(msg.id);msg.error?p.reject(Error(JSON.stringify(msg.error))):p.resolve(msg.result);}else if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params.exceptionDetails);}});
function send(method,params={},sessionId){const id=++serial;return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');});}
const timeout=setTimeout(()=>{chrome.kill();process.exitCode=1;console.error('Timeout');},240000);
try{
 const {targetId}=await send('Target.createTarget',{url:'about:blank'});const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});const call=(m,p)=>send(m,p,sessionId);
 await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url:process.argv[2]});
 const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 for(let i=0;i<150;i++){if(await evaluate('typeof captureViewerState==="function" && !!document.getElementById("inference-load")'))break;await new Promise(r=>setTimeout(r,200));}
 if(errors.length)throw Error(JSON.stringify(errors));
 await evaluate('setWorkspaceView("predict");document.getElementById("inference-load").click()');
 for(let i=0;i<600;i++){const status=await evaluate('document.getElementById("inference-status").textContent');if(i%25===0)console.log(status);if(status.startsWith('Model ready'))break;if(/failed|Unsupported|Error|HTTP|no available/.test(status))throw Error(status);await new Promise(r=>setTimeout(r,200));}
 const parity=[];
 const {root}=await call('DOM.getDocument');const {nodeId}=await call('DOM.querySelector',{nodeId:root.nodeId,selector:'#inference-image'});await call('DOM.setFileInputFiles',{nodeId,files:[process.argv[3]]});
 for(let i=0;i<600;i++){const status=await evaluate('document.getElementById("inference-status").textContent');if(i%50===0)console.log(status);if(status.startsWith('Complete.'))break;await new Promise(r=>setTimeout(r,200));}
 const result=await evaluate(`({status:document.getElementById('inference-status').textContent,rows:document.querySelectorAll('#inference-results li').length,embedding:inferenceQuery?.embedding.length,insertion:inferenceQuery?.insertion,plane:document.getElementById('projection-plane').value})`);
 console.log(JSON.stringify({result,errors}));if(process.argv[4])writeFileSync(process.argv[4],JSON.stringify({result,errors},null,2));
 if(!result.status.startsWith('Complete.')||result.rows!==15||!result.embedding||errors.length)throw Error('Browser acceptance failed');
 if(process.argv[5]){
  mkdirSync(process.argv[5],{recursive:true});
  await evaluate('document.getElementById("class-id-namespace").value="gbif";document.getElementById("class-id-namespace").onchange()');
  await new Promise(resolve=>setTimeout(resolve,12000));
  for(const [name,width,height] of [['desktop',1440,1000],['mobile',390,844]]){
   await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
   await evaluate('window.scrollTo(0,0)');await new Promise(resolve=>setTimeout(resolve,400));
   const shot=await call('Page.captureScreenshot',{format:'png'});writeFileSync(process.argv[5]+'/'+name+'.png',Buffer.from(shot.data,'base64'));
  }
 }
 await send('Browser.close');
}catch(error){console.error(error);chrome.kill();process.exitCode=1;}finally{clearTimeout(timeout);}
