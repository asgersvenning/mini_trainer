// Dependency-free numerical checks for the browser's pure layout functions.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('./thumbnails.js',import.meta.url),'utf8');
const {thumbnailEnergy,stepThumbnailForce,stepThumbnailPresentation}=Function(source.slice(source.indexOf('function thumbnailSharedArea'),source.indexOf('let mapThumbMotion'))+'return {thumbnailEnergy,stepThumbnailForce,stepThumbnailPresentation};')();
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

// Measure the complete visible trajectory, including startup, a target reversal,
// and the deceleration tail, using real seconds at different display rates.
const trajectories=[];
for(const hz of [30,60,120]){
 const boxes=[box(0,100,100)],target=[box(0,228,100)];
 let previousX=100,previousVelocity=0,peakSpeed=0,peakAcceleration=0,firstStep=0,result;
 const samples=[];
 for(let frame=1;frame<=hz*5;frame++){
  if(frame===hz+1)target[0].x=68;
  result=stepThumbnailPresentation(boxes,target,1/hz);
  const velocity=(boxes[0].x-previousX)*hz;
  peakSpeed=Math.max(peakSpeed,Math.abs(velocity));
  peakAcceleration=Math.max(peakAcceleration,Math.abs(velocity-previousVelocity)*hz);
  if(frame===1)firstStep=boxes[0].x-100;
  assert.ok(boxes[0].x>=68&&boxes[0].x<=228,'display remains inside feasible convex region');
  previousX=boxes[0].x;previousVelocity=velocity;
  if(frame%(hz/10)===0)samples.push(boxes[0].x);
 }
 assert.ok(peakSpeed<=140.01,'visible speed bounded in CSS pixels per second');
 assert.ok(peakAcceleration<2240,'acceleration bounded even at target reversal');
 assert.ok(firstStep<.6,'startup must ease in instead of jumping a solver step');
 assert.ok(result.done&&result.speed<.02,'motion stops only after the slow tail');
 trajectories.push({hz,peakSpeed,peakAcceleration,firstStep,samples});
}
for(let i=1;i<trajectories.length;i++)assert.ok(trajectories[i].samples.every((x,j)=>Math.abs(x-trajectories[0].samples[j])<1e-8),'same elapsed time yields same presentation across refresh rates');
console.log(JSON.stringify({trajectories:trajectories.map(({samples,...metrics})=>metrics)}));

// Drive the real animation coordinator with deterministic browser frames. An
// arriving image must not cancel a frame, rewind destinations, or pause others.
let queued=null,clock=0,cancellations=0;
const coordinator=Function('requestAnimationFrame','cancelAnimationFrame','matchMedia',
 source.slice(0,source.indexOf('function thumbnailFootprint'))+`
 paintThumbnailMotion=()=>{};
 return {set:boxes=>{mapThumbLayout=boxes;},start:animateThumbnails,
 motion:()=>mapThumbMotion,stop:stopThumbnailMotion};
`)(callback=>{queued=callback;return 1;},()=>{queued=null;cancellations++;},()=>({matches:false}));
const actors=[box(0,200,200),box(1,220,200)];coordinator.set(actors);
const config={width:600,height:400,size:64};
const running=coordinator.start(config);
const frame=()=>{clock+=1000/60;const callback=queued;queued=null;callback(clock);};
for(let i=0;i<6;i++)frame();
const original=coordinator.motion(),destination=original.byId.get(0),lastTime=original.last;
const newcomer=box(2,500,300);newcomer.readyAt=clock+100;actors.unshift(newcomer);
assert.equal(coordinator.start(config),running,'arrivals share the running completion promise');
assert.equal(coordinator.motion(),original,'arrival preserves the animation loop');
assert.equal(original.last,lastTime,'arrival does not reset frame time');
const beforeArrival=actors[1].x,steps=original.state.steps;
for(let i=0;i<4;i++)frame();
assert.equal(cancellations,0,'image appearance must not cancel frames');
assert.equal(original.byId.get(0),destination,'existing numerical targets survive arrival and reordering');
assert.ok(original.state.steps>steps,'existing contacts advance while a newcomer waits');
assert.ok(Math.abs(actors[1].x-beforeArrival)>.5,'existing image keeps moving during appearance delay');
assert.equal(newcomer.x,500,'new image stays still until its own appearance delay');
assert.ok(!original.byId.has(2),'waiting image exerts no collision force');
for(let i=0;i<4;i++)frame();
assert.ok(original.byId.has(2),'new image joins the same solver after appearing');
assert.equal(original.byId.get(0),destination,'joining contacts preserve existing destinations');
for(let i=0;i<600&&coordinator.motion();i++)frame();
assert.equal(coordinator.motion(),null,'joined animation eventually settles');
await running;
console.log(JSON.stringify({arrivalContinuityChecks:12}));
