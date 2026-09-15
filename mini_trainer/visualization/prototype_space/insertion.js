/* Fixed-map t-SNE insertion. Only the query moves; all reference coordinates stay fixed.
 * Minimize KL(P(query -> prototypes) || normalized Student-t(query -> map)).
 * This is an out-of-sample approximation, not a refit of the joint t-SNE map.
 */
function insertionAffinities(distances,perplexity=30){
 const ids=Array.from(distances,(_,i)=>i).sort((a,b)=>distances[a]-distances[b]||a-b).slice(0,Math.min(distances.length,Math.ceil(3*perplexity)));
 const target=Math.log(Math.min(perplexity,ids.length));let beta=1,lo=0,hi=Infinity,probabilities=[];
 const minimum=distances[ids[0]]**2;
 for(let iteration=0;iteration<80;iteration++){
  probabilities=ids.map(i=>Math.exp(-beta*(distances[i]**2-minimum)));
  const total=probabilities.reduce((a,b)=>a+b,0);probabilities=probabilities.map(p=>p/total);
  const entropy=-probabilities.reduce((s,p)=>s+(p>0?p*Math.log(p):0),0);
  if(Math.abs(entropy-target)<1e-7)break;
  if(entropy>target){lo=beta;beta=Number.isFinite(hi)?(lo+hi)/2:beta*2;}else{hi=beta;beta=(lo+hi)/2;}
 }
 return {ids,probabilities,perplexity:Math.exp(-probabilities.reduce((s,p)=>s+(p?p*Math.log(p):0),0))};
}
function insertionObjective(point,coordinates,affinities){
 let sum=0,rx=0,ry=0,loss=0,gx=0,gy=0;
 for(const q of coordinates){const dx=point[0]-q[0],dy=point[1]-q[1],w=1/(1+dx*dx+dy*dy);sum+=w;rx+=w*w*dx;ry+=w*w*dy;}
 affinities.ids.forEach((id,i)=>{const p=affinities.probabilities[i];if(!p)return;const q=coordinates[id],dx=point[0]-q[0],dy=point[1]-q[1],w=1/(1+dx*dx+dy*dy);loss+=p*Math.log(p/w);gx+=p*w*dx;gy+=p*w*dy;});
 return {loss:loss+Math.log(sum),gradient:[2*(gx-rx/sum),2*(gy-ry/sum)]};
}
function insertIntoFixedTSNE(embedding,prototypes,coordinates,{perplexity=30,iterations=200,k=12}={}){
 const began=performance.now(),n=coordinates.length,d=embedding.length;
 if(prototypes.length!==n*d||n<2)throw Error('Prototype dimensions do not match the t-SNE map.');
 const norm=Math.hypot(...embedding);if(!Number.isFinite(norm)||norm===0)throw Error('Invalid image embedding.');
 const distances=new Float64Array(n);
 for(let i=0;i<n;i++){let dot=0,length=0;for(let j=0;j<d;j++){const w=prototypes[i*d+j];dot+=w*embedding[j];length+=w*w;}distances[i]=Math.acos(Math.max(-1+1e-7,Math.min(1-1e-7,dot/(Math.sqrt(length)*norm))));}
 const affinities=insertionAffinities(distances,perplexity);
 const centroid=[0,0];affinities.ids.forEach((id,i)=>{for(let j=0;j<2;j++)centroid[j]+=affinities.probabilities[i]*coordinates[id][j];});
 const starts=[centroid,...affinities.ids.slice(0,3).map(id=>coordinates[id].slice())];let best=null;
 for(const start of starts){
  let point=start.slice(),state=insertionObjective(point,coordinates,affinities),rate=10;const initialLoss=state.loss;
  for(let iteration=0;iteration<iterations;iteration++){
   const norm=Math.hypot(...state.gradient);if(norm<1e-7)break;
   let accepted=false,step=Math.min(rate,2/norm);
   for(let search=0;search<20;search++){
    const next=point.map((v,j)=>v-step*state.gradient[j]),candidate=insertionObjective(next,coordinates,affinities);
    if(candidate.loss<=state.loss-1e-4*step*norm*norm){point=next;state=candidate;rate=Math.min(100,step*1.5);accepted=true;break;}step*=.5;
   }if(!accepted)break;
  }
  if(!best||state.loss<best.kl)best={coordinates:point,kl:state.loss,initial_kl:initialLoss};
 }
 const count=Math.min(k,n),original=affinities.ids.slice(0,count);
 const nearby=coordinates.map((p,i)=>({i,d:(p[0]-best.coordinates[0])**2+(p[1]-best.coordinates[1])**2})).sort((a,b)=>a.d-b.d||a.i-b.i).slice(0,count).map(p=>p.i);
 return {...best,neighbours:original,projected_neighbours:nearby,retained_fraction:original.filter(i=>nearby.includes(i)).length/count,perplexity:affinities.perplexity,milliseconds:performance.now()-began,method:'fixed-map conditional KL insertion'};
}
if(typeof module!=='undefined')module.exports={insertionAffinities,insertionObjective,insertIntoFixedTSNE};
