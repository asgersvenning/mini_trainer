/* Shared browser-only GBIF metadata. Original model IDs never change. */
class BrowserGBIF {
 constructor({fetcher=globalThis.fetch.bind(globalThis),concurrency=4,timeout=12000}={}) {
  this.fetcher=fetcher;this.concurrency=concurrency;this.timeout=timeout;
  this.cache=new Map();this.pending=new Map();this.queue=[];this.active=0;
  this.controller=new AbortController();
 }
 seed(snapshot){
  if(snapshot?.schema!=='mini-trainer-gbif-v1'||!snapshot.taxa||typeof snapshot.taxa!=='object')throw Error('Invalid GBIF snapshot');
  let count=0;
  for(const [id,taxon] of Object.entries(snapshot.taxa)){
   if(!/^\d+$/.test(id)||!taxon||String(taxon.key)!==id)continue;
   this.cache.set('species/'+id,taxon);count++;
  }
  return count;
 }
 reset(){
  this.controller.abort();this.controller=new AbortController();
  this.cache.clear();
 }
 request(path){
  const signal=this.controller.signal,key=path;
  const existing=this.pending.get(key);
  if(this.cache.has(key))return Promise.resolve(this.cache.get(key));
  if(existing&&existing.signal===signal)return existing.promise;
  const job={key,signal};
  job.promise=new Promise((resolve,reject)=>Object.assign(job,{resolve,reject}));
  this.pending.set(key,job);this.queue.push(job);this.pump();return job.promise;
 }
 pump(){
  while(this.active<this.concurrency&&this.queue.length){
   const job=this.queue.shift();
   if(job.signal.aborted){job.reject(new DOMException('Lookup cancelled','AbortError'));if(this.pending.get(job.key)===job)this.pending.delete(job.key);continue;}
   this.active++;
   (async()=>{
    try{
     const signal=AbortSignal.any([job.signal,AbortSignal.timeout(this.timeout)]);
     const response=await this.fetcher('https://api.gbif.org/v1/'+job.key,{signal,credentials:'omit'});
     if(!response.ok)throw Error(response.status===429?'GBIF request limit reached; retry later.':`GBIF HTTP ${response.status}`);
     const value=await response.json();
     if(job.signal.aborted)throw new DOMException('Lookup cancelled','AbortError');
     if(!value||typeof value!=='object'||Array.isArray(value))throw Error('Invalid GBIF response');
     this.cache.set(job.key,value);job.resolve(value);
    }catch(error){job.reject(error);}
    finally{this.active--;if(this.pending.get(job.key)===job)this.pending.delete(job.key);this.pump();}
   })();
  }
 }
 async taxon(id){
  id=String(id);if(!/^\d+$/.test(id))throw Error('Not a GBIF taxon ID');
  const value=await this.request('species/'+id);
  if(String(value.key)!==id)throw Error('GBIF returned a different taxon');
  return value;
 }
 async name(id){
  const taxon=await this.taxon(id);
  return {class_id:String(id),display_name:taxon.canonicalName||taxon.scientificName||String(id)};
 }
 async album(id){
  id=String(id);const generation=this.controller.signal;
  const taxon=await this.taxon(id);
  if(generation.aborted)throw new DOMException('Lookup cancelled','AbortError');
  const records=await this.request('occurrence/search?taxonKey='+id+'&mediaType=StillImage&limit=20');
  if(generation.aborted)throw new DOMException('Lookup cancelled','AbortError');
  if(!Array.isArray(records.results))throw Error('Invalid GBIF occurrence response');
  return BrowserGBIF.albumFromRecords(id,taxon,records.results);
 }
 static webURL(value){
  if(typeof value!=='string')return false;
  try{return ['https:','http:'].includes(new URL(value).protocol);}catch{return false;}
 }
 static albumFromRecords(id,taxon,records){
  const accepted=String(taxon.acceptedKey||id),photos=[],seen=new Set();
  for(const record of records){
   if(!record||!/^\d+$/.test(String(record.key)))continue;
   const members=['taxonKey','acceptedTaxonKey','speciesKey','genusKey','familyKey','orderKey','classKey','phylumKey','kingdomKey'].map(key=>String(record[key]));
   if(!members.includes(id)&&!members.includes(accepted))continue;
   for(const media of Array.isArray(record.media)?record.media:[]){
    if(!media||media.type!=='StillImage'||!this.webURL(media.identifier)||seen.has(media.identifier))continue;
    seen.add(media.identifier);
    photos.push({occurrence_id:String(record.key),image_path:media.identifier,original:media.identifier,
     source:this.webURL(media.references)?media.references:'https://www.gbif.org/occurrence/'+record.key,
     creator:media.creator||media.rightsHolder||'Creator not supplied',license:media.license||'Image license not supplied',scientific_name:record.scientificName||''});
    if(photos.length===6)break;
   }
   if(photos.length===6)break;
  }
  return {class_id:id,accepted_id:accepted,display_name:taxon.canonicalName||taxon.scientificName||id,taxonomic_status:taxon.taxonomicStatus,photos,selection:'GBIF class examples'};
 }
}
if(typeof module!=='undefined')module.exports={BrowserGBIF};
