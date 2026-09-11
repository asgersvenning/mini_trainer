/* Versioned presentation state. Numerical data and transient image motion are excluded. */
const viewerStateControls=['workspace-view','projection-plane','metric','global-metric','range-preset','range-low','range-high','score-display','class-names','gbif-names','photo-enabled','map-photos','map-photo-size','map-photo-density','map-photo-overlap','map-photo-labels','map-photo-push','map-photo-aspect','map-photo-anchors','projection-edges'];
let restoringViewerState=false,viewerStateTimer,viewerServerToken=null,viewerStateChanged=false,viewerStateWrites=Promise.resolve();
const viewerDocumentKey='prototype-case:'+(Object.values(cases).find(c=>!c.metadata.synthetic)?.metadata.checkpoint_sha256||Object.keys(cases).join(':'));
function viewerIdentity(){
 let hash=2166136261;for(const name of data.names)for(const c of name+'\0')hash=Math.imul(hash^c.charCodeAt(0),16777619);
 return [data.metadata.checkpoint_sha256||'synthetic',data.metadata.seed||0,$('case').value,data.names.length,hash>>>0].join(':');
}
function captureViewerState(){
 return {version:1,identity:viewerIdentity(),selected:data.names[selected],tree:treeFocus,view:{...projectionView},controls:Object.fromEntries(viewerStateControls.map(id=>[id,$(id).type==='checkbox'?$(id).checked:$(id).value]))};
}
function stateMessage(message){$('viewer-state-status').textContent=message;}
function saveViewerState(){
 if(restoringViewerState)return;
 try{localStorage.setItem('prototype-view:'+viewerIdentity(),JSON.stringify(captureViewerState()));localStorage.setItem(viewerDocumentKey,$('case').value);stateMessage('View saved locally.');}
 catch{stateMessage('Browser storage unavailable. Export state to keep this view.');}
 if(viewerServerToken){const body=JSON.stringify({key:viewerDocumentKey,state:{case:$('case').value,view:captureViewerState()}});viewerStateWrites=viewerStateWrites.then(()=>fetch('/api/view-state',{method:'POST',headers:{'Content-Type':'application/json','X-Explorer-Token':viewerServerToken},body,keepalive:true})).catch(()=>stateMessage('Local server state unavailable; browser save/export remains available.'));}
}
function applyViewerState(state){
 if(state?.version!==1||state.identity!==viewerIdentity())throw Error('State belongs to a different model/case or unsupported version.');
 const index=data.names.indexOf(state.selected),v=state.view;
 if(index<0||!v||![v.x,v.y,v.scale].every(Number.isFinite)||v.scale<=0||!Number.isInteger(state.tree)||state.tree<0||state.tree>=data.names.length*2-1)throw Error('Invalid saved selection or geometry.');
 if(!state.controls||typeof state.controls!=='object')throw Error('Missing saved controls.');
 for(const [id,value] of Object.entries(state.controls)){
  if(!viewerStateControls.includes(id))continue;const node=$(id);
  if(node.type==='checkbox'){if(typeof value!=='boolean')throw Error('Invalid checkbox state.');}
  else if(node.tagName==='SELECT'){if(![...node.options].some(o=>o.value===value))throw Error('Unavailable saved view or projection.');}
  else if(!Number.isFinite(Number(value))||(node.min!==''&&Number(value)<Number(node.min))||(node.max!==''&&Number(value)>Number(node.max)))throw Error('Invalid saved control range.');
 }
 restoringViewerState=true;
 try{
  for(const [id,value] of Object.entries(state.controls)){if(!viewerStateControls.includes(id))continue;const node=$(id);if(node.type==='checkbox')node.checked=value;else node.value=value;}
  setWorkspaceView($('workspace-view').value);treeFocus=state.tree;selectClass(index);
  $('score-note').hidden=$('score-display').value!=='tail';drawHistogram();drawGlobal();setThumbnailAnchors();
  // Wait for the new focused layout's ResizeObserver before restoring its transform.
  requestAnimationFrame(()=>requestAnimationFrame(()=>{projectionView={...v};drawProjection();restoringViewerState=false;stateMessage('Saved view restored.');}));
 }catch(error){restoringViewerState=false;throw error;}
}
function restoreViewerState(){
 try{const raw=localStorage.getItem('prototype-view:'+viewerIdentity());if(raw)applyViewerState(JSON.parse(raw));else stateMessage('No saved view for this model/case.');}
 catch(error){stateMessage('Saved view not restored: '+error.message);}
}
function queueViewerState(){if(restoringViewerState)return;viewerStateChanged=true;clearTimeout(viewerStateTimer);viewerStateTimer=setTimeout(saveViewerState,400);}
const stateMenu=document.createElement('details');stateMenu.className='label-settings';stateMenu.innerHTML='<summary>Saved view</summary><div><button type="button" id="viewer-state-export">Export state</button><label>Import state <input id="viewer-state-import" type="file" accept=".json"></label><button type="button" id="viewer-state-reset">Reset saved view</button><span id="viewer-state-status" class="caption" role="status"></span></div>';
document.querySelector('.controls').append(stateMenu);
$('viewer-state-export').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(captureViewerState(),null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='prototype-view.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('viewer-state-import').onchange=async()=>{try{const file=$('viewer-state-import').files[0];if(file)applyViewerState(JSON.parse(await file.text()));}catch(error){stateMessage(error.message);}};
$('viewer-state-reset').onclick=async()=>{restoringViewerState=true;clearTimeout(viewerStateTimer);try{localStorage.removeItem('prototype-view:'+viewerIdentity());await viewerStateWrites;if(viewerServerToken)await fetch('/api/view-state',{method:'POST',headers:{'Content-Type':'application/json','X-Explorer-Token':viewerServerToken},body:JSON.stringify({key:viewerDocumentKey,state:null})});}catch{}location.reload();};
document.addEventListener('change',event=>{if(event.target.id==='case')restoreViewerState();else queueViewerState();});
document.addEventListener('click',queueViewerState);document.addEventListener('pointerup',queueViewerState);document.addEventListener('wheel',queueViewerState,{passive:true});
window.addEventListener('pagehide',saveViewerState);
try{const savedCase=localStorage.getItem(viewerDocumentKey);if(savedCase&&cases[savedCase]&&$('case').value!==savedCase){$('case').value=savedCase;setCase();}}catch{}
restoreViewerState();

// The local launcher can change port between sessions, unlike browser storage.
if(location.protocol.startsWith('http'))fetch('/api/view-state?key='+encodeURIComponent(viewerDocumentKey)).then(r=>r.ok?r.json():null).then(result=>{
 if(!result)return;viewerServerToken=result.token;
 if(!viewerStateChanged&&result.state?.view&&cases[result.state.case]){
  $('case').value=result.state.case;setCase();applyViewerState(result.state.view);
 }
}).catch(()=>{});
