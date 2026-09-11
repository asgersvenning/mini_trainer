/* Optional reference photos using the shared browser GBIF client. */
let photoController = null;
let photoGeneration = 0;
const photoAlbums = new Map();
const photoChoices = new Map();
const photoNumber = id => /^\d+$/.test(id);

// Share metadata between map generations and the neighbourhood cards. Navigation
// discards queued work; up to four started lookups finish into the session cache.
const albumRequests=new Map(),albumQueue=[];
let activeAlbumRequests=0;
function referenceAlbum(id,signal){
 if(photoAlbums.has(id))return Promise.resolve(photoAlbums.get(id));
 const existing=albumRequests.get(id);if(existing&&existing.generation===gbifClient.controller.signal){existing.signals.push(signal);return existing.promise;}
 const job={id,signals:[signal],generation:gbifClient.controller.signal};job.promise=new Promise((resolve,reject)=>{job.resolve=resolve;job.reject=reject;});
 albumRequests.set(id,job);albumQueue.push(job);pumpAlbumRequests();return job.promise;
}
function pumpAlbumRequests(){
 while(activeAlbumRequests<4&&albumQueue.length){
  const job=albumQueue.shift();
  if(job.generation.aborted||job.signals.every(signal=>signal.aborted)){if(albumRequests.get(job.id)===job)albumRequests.delete(job.id);job.reject(new DOMException('Navigation cancelled','AbortError'));continue;}
  activeAlbumRequests++;
  (async()=>{try{
   if(!gbifEnabled())throw new DOMException('GBIF disabled','AbortError');
   const album=await gbifClient.album(job.id);
   photoAlbums.set(job.id,album);job.resolve(album);
  }catch(error){job.reject(error);}finally{if(albumRequests.get(job.id)===job)albumRequests.delete(job.id);activeAlbumRequests--;pumpAlbumRequests();}})();
 }
}

function photoText(tag, text, className = '') {
  const node = document.createElement(tag);
  node.textContent = text;
  node.className = className;
  return node;
}

function photoLink(text, href) {
  const node = photoText('a', text);
  node.href = href;
  node.target = '_blank';
  node.rel = 'noopener noreferrer';
  return node;
}

function loadReferenceImage(image, src, signal) {
  return new Promise(resolve => {
    let timer;
    const done = ok => {
      clearTimeout(timer);
      image.onload = image.onerror = null;
      signal.removeEventListener('abort', abort);
      resolve(ok);
    };
    const abort = () => { image.removeAttribute('src'); done(false); };
    if (signal.aborted) { done(false); return; }
    signal.addEventListener('abort', abort, {once: true});
    image.onload = () => done(true);
    image.onerror = () => done(false);
    timer = setTimeout(abort, 18000);
    // Keep the image visible during loading so progressive decoding can appear.
    image.src = src;
  });
}

async function showReference(card, album, classIndex, generation, signal) {
  const photos = album.photos;
  card.title.textContent = album.display_name;
  if (!photos.length) { card.status.textContent = 'No matching class examples in the first 20 GBIF records.'; return; }
  const choice = (photoChoices.get(album.class_id) || 0) % photos.length;
  const photo = photos[choice];
  card.credit.replaceChildren(
    photoText('span', photo.creator + ' · ' + photo.license + ' '),
    photoLink('Source', photo.source),
    photoText('span', ' · '),
    photoLink('GBIF record', 'https://www.gbif.org/occurrence/' + photo.occurrence_id)
  );
  card.image.alt = `GBIF example of ${album.display_name}; class ${album.class_id}`;
  card.status.textContent = `Loading example ${choice + 1} / ${photos.length}…`;
  card.next.hidden = photos.length < 2;
  card.next.disabled = true;
  card.next.onclick = async () => {
    if (generation !== photoGeneration) return;
    photoChoices.set(album.class_id, choice + 1);
    card.next.disabled = true;
    await showReference(card, album, classIndex, generation, signal);
    if(typeof scheduleMapThumbnails==='function')scheduleMapThumbnails();
    card.next.disabled = false;
  };
  const loaded = await loadReferenceImage(card.image, photo.image_path, signal);
  if (generation !== photoGeneration) return;
  card.next.disabled = false;
  card.status.textContent = loaded ? `Example ${choice + 1} / ${photos.length}${album.accepted_id !== album.class_id ? ' · GBIF accepted ID ' + album.accepted_id : ''}` : 'Image unavailable or timed out. Try another example.';
}

async function renderPhotos() {
  const generation = ++photoGeneration;
  photoController?.abort();
  photoController = new AbortController();
  const signal = photoController.signal;
  const grid = $('photo-grid');
  grid.replaceChildren();
  const ids = [selected, ...data.neighbours[selected]];
  if (data.metadata.synthetic || !ids.some(i => photoNumber(data.names[i]))) {
    $('photo-status').textContent = 'This case has no GBIF-ID candidates. No online lookup was made.';
    return;
  }
  if (!gbifEnabled() || !$('photo-enabled').checked) {
    $('photo-status').textContent = 'Select GBIF taxon IDs in Settings and enable reference photos to browse examples.';
    return;
  }
  $('photo-status').textContent = 'Loading class examples sequentially; class IDs and numerical neighbourhoods stay unchanged.';
  const cards = ids.map((classIndex, rank) => {
    const card = photoText('article', '', 'photo-card' + (rank === 0 ? ' photo-selected' : ''));
    const open = photoText('button', '', 'photo-open');
    open.type = 'button';
    open.setAttribute('aria-label', `Explore class ${data.names[classIndex]}`);
    const image = document.createElement('img');
    image.decoding = 'async';
    open.append(image);
    open.onclick = () => selectClass(classIndex);
    const title = photoText('strong', data.names[classIndex], 'photo-title');
    const identity = photoText('span', `${rank === 0 ? 'Selected' : 'Neighbour ' + rank} · class ${data.names[classIndex]}`, 'caption');
    const status = photoText('p', 'Waiting…', 'caption');
    const next = photoText('button', 'Another example');
    next.type = 'button'; next.hidden = true;
    const credit = photoText('p', '', 'photo-credit');
    card.append(open, title, identity, status, next, credit);
    grid.append(card);
    return {image, title, status, next, credit};
  });
  // Only fetch the visible neighbourhood. Cache metadata in-session and images
  // on the local server; do not download one image for every checkpoint class.
  for (let rank = 0; rank < ids.length; rank++) {
    if (signal.aborted || generation !== photoGeneration) return;
    const classIndex = ids[rank], id = data.names[classIndex], card = cards[rank];
    if (!photoNumber(id)) {card.status.textContent = 'Not a GBIF ID.'; continue;}
    try {
      const album = await referenceAlbum(id, signal);
      if (signal.aborted || generation !== photoGeneration) return;
      await showReference(card, album, classIndex, generation, signal);
    } catch (error) {
      if (!signal.aborted) card.status.textContent = 'Example lookup unavailable. Toggle photos to retry.';
    }
  }
  if (generation === photoGeneration) $('photo-status').textContent = 'Click a photo to explore that class’s neighbours. Images label classes at their learned prototype directions.';
}
$('photo-enabled').onchange = renderPhotos;
renderPhotos();
