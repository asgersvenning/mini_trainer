# The review sheet of species_screening.R (an HTML page with its own script). Depends on web.R, io.R.

# The review sheet: per species its best photos (best first). Click a photo to select it (click again
# to unselect), drag the selected photo to rotate it, or give another iNaturalist photo (its URL),
# or omit the species; add a comment if wanted. The sheet starts from the saved picks of the study
# (`saved`, see read_picks()); the selection is kept in the browser (localStorage). The floating
# button exports the selection as the picks table (CSV): copy or download it as
# data/species_picks/<study>.csv. The same text can be pasted back and imported.
write_selection_sheet <- function(ranked, study, saved = NULL) {
  card <- \(photo_id, image_url, observation_url, rank, detail) {
    str_c(
      "<div class='card' data-rank='", rank, "' data-photo='", key_string(photo_id),
      "'><div class='wrap'><img src='", html_escape(image_url), "'></div><div class='meta'><b>#",
      rank, "</b> ", html_escape(detail), " <a href='", html_escape(observation_url),
      "' target='_blank'>iNaturalist</a></div></div>"
    )
  }
  rows <- ranked |>
    mutate(
      rank = row_number(),
      detail = str_c(
        "min ", score, ", mean ", round(mean_score, 2), ", quality ", quality, ", ", description,
        "; verif. ", verifications, ", faves ", faves, ", ", licence
      ),
      .by = speciesKey
    ) |>
    slice_head(n = 6, by = speciesKey) |>
    arrange(name, rank) |>
    nest(.by = c(speciesKey, name)) |>
    pmap_chr(\(speciesKey, name, data) {
      str_c(
        "<section class='sp' data-key='", key_string(speciesKey), "' data-name='", html_escape(name),
        "'><h3>", html_escape(name), "</h3><div class='line'><div class='row'>",
        str_c(
          pmap_chr(data[c("photo_id", "image_url", "observation_url", "rank", "detail")], card),
          collapse = ""
        ),
        "</div><div class='side'>",
        "<label>rotation (&deg; clockwise; drag the photo): <input class='rot' type='number' step='1' value='0'></label>",
        "<div class='btns'><button class='l90'>&#8634; 90&deg;</button> <button class='r90'>&#8635; 90&deg;</button> <button class='zero'>reset</button></div>",
        "<label>other photo (iNaturalist URL) <input class='other' type='text' placeholder='https://www.inaturalist.org/photos/...'></label>",
        "<div class='wrap other-wrap' hidden><img class='other-img'></div>",
        "<label><input class='omit' type='checkbox'> omit species</label>",
        "<textarea class='comment' placeholder='comment'></textarea></div></div></section>"
      )
    })
  style <- r"---(
body{font:14px sans-serif;margin:1em 2em 5em}
.line{display:flex;gap:16px;align-items:flex-start}
.row{display:flex;gap:10px;flex-wrap:wrap}
.card{width:240px;border:3px solid transparent;cursor:pointer;box-sizing:content-box}
.card:hover{border-color:#bbb}.card.sel{border-color:#d33}
.wrap{overflow:hidden;width:240px;height:240px;background:#eee}
.wrap img{width:240px;height:240px;object-fit:cover;display:block;transform:rotate(var(--rot,0deg))}
.card.sel img,.other-wrap img{cursor:grab;touch-action:none}
.meta{font-size:12px;color:#444}h3{margin:1.2em 0 .3em}
.side{display:flex;flex-direction:column;gap:6px;width:240px;font-size:12px}
.side input[type=text],.side input[type=number],.side textarea{width:100%;box-sizing:border-box}
.side textarea{height:70px}.sp.done h3::after{content:" \2713";color:#2a2}
#bar{position:fixed;right:16px;bottom:16px;z-index:5}
#bar button{font-size:15px;padding:10px 18px;cursor:pointer}
#out{position:fixed;right:16px;bottom:64px;width:min(760px,90vw);height:60vh;z-index:5;
background:#fff;border:1px solid #888;box-shadow:0 2px 12px #0006;padding:10px;display:none;flex-direction:column;gap:8px}
#out textarea{flex:1;font:12px monospace}
)---"
  script <- r"---(
const KEY = 'picks:' + document.title;
let state = {}, stored = false;
try { const t = localStorage.getItem(KEY); if (t) { state = JSON.parse(t); stored = true; } } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const sections = [...document.querySelectorAll('.sp')];
const urlId = u => { const m = (u || '').match(/inaturalist\.org\/photos\/(\d+)/); return m ? m[1] : null; };
const chosenId = s => urlId(s.other) || s.photo || '';
const done = s => chosenId(s) || s.omit;
const norm = d => ((Math.round(+d || 0) % 360) + 360) % 360;
const views = {};
const count = () => { document.getElementById('open').textContent =
  'Export picks (' + sections.filter(sp => done(state[sp.dataset.key])).length + '/' + sections.length + ')'; };

// the state of every species exists before any is rendered (count() reads them all)
sections.forEach(sp => {
  state[sp.dataset.key] = Object.assign({photo: '', other: '', rotate: 0, omit: false, comment: ''}, state[sp.dataset.key]);
});
sections.forEach(sp => {
  const k = sp.dataset.key;
  const s = state[k];
  const cards = [...sp.querySelectorAll('.card')];
  // a selected photo that is no longer among the cards (the candidates changed) is shown as another photo
  if (s.photo && !cards.some(c => c.dataset.photo === s.photo)) {
    s.other = 'https://www.inaturalist.org/photos/' + s.photo; s.photo = '';
  }
  const rot = sp.querySelector('.rot'), other = sp.querySelector('.other');
  const omit = sp.querySelector('.omit'), comment = sp.querySelector('.comment');
  const otherWrap = sp.querySelector('.other-wrap'), otherImg = sp.querySelector('.other-img');
  const render = () => {
    const oid = urlId(s.other);
    cards.forEach(c => {
      const on = !oid && c.dataset.photo === s.photo;
      c.classList.toggle('sel', on);
      c.style.setProperty('--rot', (on ? s.rotate : 0) + 'deg');
    });
    otherWrap.hidden = !oid;
    if (oid && otherImg.dataset.id !== oid) {
      otherImg.dataset.id = oid; otherImg.dataset.tried = '';
      otherImg.src = 'https://inaturalist-open-data.s3.amazonaws.com/photos/' + oid + '/medium.jpg';
    }
    otherWrap.style.setProperty('--rot', s.rotate + 'deg');
    rot.value = s.rotate; other.value = s.other; omit.checked = s.omit; comment.value = s.comment;
    sp.classList.toggle('done', !!done(s)); count();
  };
  views[k] = render;
  otherImg.addEventListener('error', () => {  // the file is .jpg or .jpeg
    if (!otherImg.dataset.tried) { otherImg.dataset.tried = '1';
      otherImg.src = 'https://inaturalist-open-data.s3.amazonaws.com/photos/' + otherImg.dataset.id + '/medium.jpeg'; }
  });
  const setRotation = deg => { s.rotate = norm(deg); save(); render(); };
  sp.querySelector('.l90').addEventListener('click', () => setRotation(s.rotate - 90));
  sp.querySelector('.r90').addEventListener('click', () => setRotation(s.rotate + 90));
  sp.querySelector('.zero').addEventListener('click', () => setRotation(0));
  rot.addEventListener('change', () => setRotation(rot.value));
  other.addEventListener('input', () => { s.other = other.value.trim(); save(); render(); });
  omit.addEventListener('change', () => { s.omit = omit.checked; save(); render(); });
  comment.addEventListener('input', () => { s.comment = comment.value; save(); });

  // drag the shown photo (the selected card or the other photo) around its centre to rotate it
  // (shift: in steps of 15 degrees)
  let drag = null, moved = false;
  [...cards.map(c => c.querySelector('img')), otherImg].forEach(img => {
    const active = () => img === otherImg ? !!urlId(s.other) :
      !urlId(s.other) && img.closest('.card').dataset.photo === s.photo;
    img.addEventListener('pointerdown', e => {
      if (!active()) return;
      const r = img.parentElement.getBoundingClientRect();
      const centre = [r.left + r.width / 2, r.top + r.height / 2];
      drag = {centre, from: s.rotate, x: e.clientX, y: e.clientY,
              start: Math.atan2(e.clientY - centre[1], e.clientX - centre[0])};
      moved = false; img.setPointerCapture(e.pointerId); e.preventDefault();
    });
    img.addEventListener('pointermove', e => {
      if (!drag) return;
      if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 3) moved = true;
      if (!moved) return;
      let deg = drag.from + (Math.atan2(e.clientY - drag.centre[1], e.clientX - drag.centre[0]) - drag.start) * 180 / Math.PI;
      if (e.shiftKey) deg = Math.round(deg / 15) * 15;
      s.rotate = norm(deg); render();
    });
    img.addEventListener('pointerup', () => { drag = null; save(); });
  });
  cards.forEach(c => c.addEventListener('click', e => {
    if (e.target.closest('a')) return;
    if (moved) { moved = false; return; }
    s.other = ''; s.photo = s.photo === c.dataset.photo ? '' : c.dataset.photo; save(); render();
  }));
  render();
});

// ---- picks as CSV: speciesKey,name,photo_id,rotate,omit,comment ----
const quote = v => '"' + String(v).replace(/"/g, '""') + '"';
const table = () => 'speciesKey,name,photo_id,rotate,omit,comment\n' + sections.map(sp => {
  const s = state[sp.dataset.key];
  return [sp.dataset.key, quote(sp.dataset.name), chosenId(s), s.rotate, s.omit ? 'TRUE' : 'FALSE',
          quote(s.comment.replace(/\s+/g, ' '))].join(',');
}).join('\n') + '\n';
const parseCsv = t => {
  const rows = []; let row = [], f = '', q = false;
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (q) { if (c === '"') { if (t[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && t[i + 1] === '\n') i++; row.push(f); f = ''; rows.push(row); row = []; }
    else f += c;
  }
  if (f || row.length) { row.push(f); rows.push(row); }
  const head = rows.shift().map(h => h.trim());
  return rows.filter(r => r.length > 1).map(r => Object.fromEntries(head.map((h, i) => [h, r[i]])));
};
// Applies picks (rows with speciesKey, photo_id, rotate, omit, comment) to the species they name
const applyRows = rows => {
  let n = 0;
  rows.forEach(r => {
    const sp = sections.find(x => x.dataset.key === String(r.speciesKey).trim());
    if (!sp) return;
    const s = state[sp.dataset.key], id = urlId(r.photo_id) || String(r.photo_id || '').trim();
    const isCard = id && sp.querySelector('.card[data-photo="' + id + '"]');
    s.photo = isCard ? id : '';
    s.other = id && !isCard ? 'https://www.inaturalist.org/photos/' + id : '';
    s.rotate = norm(r.rotate);
    s.omit = String(r.omit).trim().toUpperCase() === 'TRUE';
    s.comment = r.comment || '';
    n++;
  });
  save(); Object.values(views).forEach(render => render());
  return n;
};
if (!stored && INITIAL.length) applyRows(INITIAL);

const out = document.getElementById('out'), text = document.getElementById('text');
const msg = t => { document.getElementById('msg').textContent = t; };
document.getElementById('open').addEventListener('click', () => {
  text.value = table(); msg(''); out.style.display = out.style.display === 'flex' ? 'none' : 'flex';
});
document.getElementById('copy').addEventListener('click', () => {
  text.select(); try { navigator.clipboard.writeText(text.value); } catch (e) { document.execCommand('copy'); }
  msg('copied');
});
document.getElementById('download').addEventListener('click', () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text.value], {type: 'text/csv'}));
  a.download = 'picks.csv'; a.click();
});
document.getElementById('import').addEventListener('click', () => {
  try { msg(applyRows(parseCsv(text.value)) + ' picks imported'); } catch (e) { msg('could not read the text'); }
});
document.getElementById('saved').addEventListener('click', () => {
  if (confirm('Replace the current selection with the saved picks?')) msg(applyRows(INITIAL) + ' picks loaded');
});
)---"
  initial <- if (is.null(saved)) {
    list()
  } else {
    saved |>
      transmute(
        speciesKey = key_string(speciesKey),
        photo_id = if_else(is.na(photo_id), "", key_string(photo_id)),
        rotate = coalesce(rotate, 0),
        omit = coalesce(omit, FALSE),
        comment = coalesce(comment, "")
      )
  }
  path <- derived_path("species_images", study, "selected.html")
  writeLines(
    c(
      str_c("<!doctype html><meta charset='utf-8'><title>Selected pictures ", study, "</title>"),
      str_c("<style>", style, "</style>"),
      "<p>Click a photo to select it (again to unselect), drag it to rotate it, or give another iNaturalist photo or omit the species. ",
      "The selection is kept in this browser. <b>Export picks</b> (bottom right) gives the picks table (CSV).</p>",
      rows,
      "<div id='bar'><button id='open'>Export picks</button></div>",
      "<div id='out'><textarea id='text' spellcheck='false'></textarea><div>",
      "<button id='copy'>Copy</button> <button id='download'>Download picks.csv</button> ",
      "<button id='import'>Import the text above</button> ",
      "<button id='saved'>Load the saved picks</button> <span id='msg'></span></div></div>",
      str_c("<script>const INITIAL = ", toJSON(initial, auto_unbox = TRUE), ";", script, "</script>")
    ),
    path
  )
  path
}
