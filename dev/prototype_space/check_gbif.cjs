/* Deterministic transport checks; no public API calls. */
const assert=require('node:assert/strict');
const {BrowserGBIF}=require('../../mini_trainer/visualization/prototype_space/gbif.js');
(async()=>{
 let calls=0,active=0,peak=0;
 const client=new BrowserGBIF({concurrency:2,fetcher:async url=>{
  calls++;active++;peak=Math.max(peak,active);
  await new Promise(resolve=>setTimeout(resolve,5));active--;
  return {ok:true,json:async()=>({key:Number(url.split('/').pop()),canonicalName:'Example'})};
 }});
 assert.equal(client.seed({schema:'mini-trainer-gbif-v1',taxa:{'42':{key:42,canonicalName:'Cached species'},'43':{key:99}}}),1);
 assert.equal((await client.name('42')).display_name,'Cached species');assert.equal(calls,0);
 const [a,b]=await Promise.all([client.name('123'),client.name('123')]);
 assert.deepEqual(a,b);assert.equal(calls,1);await client.name('123');assert.equal(calls,1);
 await Promise.all(['1','2','3','4'].map(id=>client.name(id)));assert.equal(peak,2);
 await assert.rejects(client.name('not-a-taxon'));
 const album=BrowserGBIF.albumFromRecords('1',{acceptedKey:2},[
  {key:10,speciesKey:9,media:[{type:'StillImage',identifier:'https://example.org/wrong.jpg'}]},
  {key:11,speciesKey:2,media:[{type:'StillImage',identifier:'javascript:alert(1)'},{type:'StillImage',identifier:'https://example.org/photo.jpg',creator:'Author',license:'CC BY'}]}
 ]);
 assert.equal(album.class_id,'1');assert.equal(album.accepted_id,'2');assert.equal(album.photos.length,1);
 assert.equal(album.photos[0].creator,'Author');assert.equal(album.photos[0].license,'CC BY');
 let release;
 const slow=new BrowserGBIF({fetcher:()=>new Promise(resolve=>release=resolve)});
 const pending=slow.name('1');slow.reset();release({ok:true,json:async()=>({key:1})});
 await assert.rejects(pending,{name:'AbortError'});assert.equal(slow.cache.size,0);
 const limited=new BrowserGBIF({fetcher:async()=>({ok:false,status:429})});
 await assert.rejects(limited.name('1'),/request limit/);assert.equal(limited.pending.size,0);
 console.log('GBIF checks passed: deduplication, cache, concurrency, IDs, taxonomy, URLs, attribution, cancellation, rate limit.');
})().catch(error=>{console.error(error);process.exitCode=1;});
