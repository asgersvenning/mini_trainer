/* Local WASM inference. Model files are fetched; user pixels never leave this worker. */
let session, manifest, prototypes, coordinates;
self.onmessage = async ({data: message}) => {
 try {
  if(message.kind==='load') {
   manifest=message.manifest;coordinates=message.coordinates;
   importScripts(new URL('insertion.js',message.base).href);
   const response=await fetch(new URL(manifest.prototypes,message.base));if(!response.ok)throw Error('Could not load prototypes.');
   prototypes=new Float32Array(await response.arrayBuffer());
   importScripts(new URL('runtime/ort.wasm.min.js',message.base).href);
   ort.env.wasm.numThreads=1;
   ort.env.wasm.wasmPaths={mjs:new URL('runtime/ort-wasm-simd-threaded.js',message.base).href,wasm:new URL('runtime/ort-wasm-simd-threaded.wasm',message.base).href};
   session=await ort.InferenceSession.create(new URL(manifest.model,message.base).href,{
    executionProviders:['wasm'],externalData:(manifest.external_data||[]).map(path=>({path,data:new URL(path,message.base).href}))
   });
   self.postMessage({kind:'ready'});
  } else if(message.kind==='infer') {
   if(!session)throw Error('Load a model first.');
   const result=await session.run({images:new ort.Tensor('float32',message.tensor,[1,3,manifest.size,manifest.size])});
   const outputs=manifest.outputs.map(name=>Array.from(result[name].data));
   const embedding=Array.from(result[manifest.embedding].data);
   const insertion=coordinates?insertIntoFixedTSNE(embedding,prototypes,coordinates):null;
   self.postMessage({kind:'result',id:message.id,outputs,embedding,insertion});
  }
 } catch(error) {self.postMessage({kind:'error',id:message.id,message:String(error.message||error)});}
};
