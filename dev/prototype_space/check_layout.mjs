// Dependency-free numerical checks for the browser's pure layout functions.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('./thumbnails.js',import.meta.url),'utf8');
const {thumbnailEnergy,stepThumbnailForce}=Function(source.slice(source.indexOf('function thumbnailSharedArea'),source.indexOf('let mapThumbMotion'))+'return {thumbnailEnergy,stepThumbnailForce};')();
const box=(id,x,y,width=64,height=64)=>({id,x,y,anchorX:x,anchorY:y,width,height});
const gradientCase=[box(0,100,100,72,60),box(1,149,127,64,80),box(2,210,150,50,64)];
gradientCase[0].anchorX-=12;
const gradient=thumbnailEnergy(gradientCase,true).forces;
for(let i=0;i<gradientCase.length;i++)for(const [axis,key] of ['x','y'].entries()){
 const original=gradientCase[i][key],epsilon=1e-5;
 gradientCase[i][key]=original+epsilon;const plus=thumbnailEnergy(gradientCase).energy;
 gradientCase[i][key]=original-epsilon;const minus=thumbnailEnergy(gradientCase).energy;
 gradientCase[i][key]=original;
 assert.ok(Math.abs((plus-minus)/(2*epsilon)+gradient[i][axis])<2e-5,'force must be the negative energy gradient');
}
const results=[];
for(const fixture of [
 {name:'separated',boxes:[box(0,100,100),box(1,300,100)],width:600,height:400},
 {name:'feasible 30-box crowd',boxes:Array.from({length:30},(_,i)=>box(i,200+i%6*50,180+Math.floor(i/6)*50)),width:900,height:700},
 {name:'impossible two-box viewport',boxes:[box(0,32,32),box(1,32,32)],width:64,height:64}
]){
 const state={},ids=fixture.boxes.map(b=>b.id),anchors=fixture.boxes.map(b=>[b.anchorX,b.anchorY]);
 let previous=thumbnailEnergy(fixture.boxes).energy;
 while(!state.done){
  stepThumbnailForce(fixture.boxes,fixture.width,fixture.height,128,state);
  assert.ok(state.metrics.energy<=previous+1e-8,'accepted steps must not increase energy');previous=state.metrics.energy;
  assert.deepEqual(fixture.boxes.map(b=>b.id),ids,'solver must preserve every admitted thumbnail');
  for(const b of fixture.boxes){assert.ok(b.left>=-1e-8&&b.top>=-1e-8&&b.left+b.width<=fixture.width+1e-8&&b.top+b.height<=fixture.height+1e-8);assert.ok(Math.hypot(b.x-b.anchorX,b.y-b.anchorY)<=128+1e-8);}
 }
 assert.deepEqual(fixture.boxes.map(b=>[b.anchorX,b.anchorY]),anchors);
 assert.equal(state.reason,'resting');assert.ok(state.quiet>=8);
 if(fixture.name.startsWith('feasible'))assert.equal(state.metrics.pairs,0);
 if(fixture.name.startsWith('impossible'))assert.equal(state.metrics.pairs,1);
 results.push({fixture:fixture.name,steps:state.steps,overlappingPairs:state.metrics.pairs});
}
console.log(JSON.stringify({gradientChecks:6,fixtures:results}));
