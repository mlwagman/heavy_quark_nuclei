'use strict';
/* engine.js: the beat API the capture shim expects (bare top-level identifiers),
   deterministic state application, the camera rig, focus tiers, the overview, keys, deep links, presenter sync, fps. */
let si = 0, done = 0, run = null, auto = false, notesOn = false;
const $ = id => document.getElementById(id);
const notesEl = $('notes'), stage = $('stage'), fpsEl = $('fps'), beatEl = $('beatid');
const QS = new URLSearchParams(location.search);
const SPEED = 2;                                              /* a beat's tween lasts d / SPEED seconds (the Deck engine used 3; 2 leaves every reveal readable) */

/* ---------- camera keyframes: one path per beat, previous beat's end as its start ---------- */
const CAM0 = W.KF(TP(0, -150, 88, 95), TP(0, -150, 88, 0));   /* the World opens on the title's frame (S01 b1's own), so no beat starts on an empty stage */
scenes.forEach((sc, i) => { if ((sc.notes || []).length !== sc.beats.length) throw new Error(`S${i + 1}: ${sc.beats.length} beats but ${(sc.notes || []).length} notes (js/notes.js is made from the narration: audio3d.py notes)`); });
const beatList = [];                                          /* flat [ {s, k, b} ] in talk order */
scenes.forEach((sc, s) => sc.beats.forEach((b, k) => beatList.push({ s, k, b })));
/* camAt: a piece of a cut beat (util.js cut) flies its share of the whole beat's path */
function endCam(idx) { for (let i = idx; i >= 0; i--) { const b = beatList[i].b; if (b.camAt) return b.camAt(1, i); if (b.cam) return b.cam[b.cam.length - 1]; } return CAM0; }
function pathCam(b, p, prev) {
  if (!b.cam) return prev;
  const w = b.win || [0, .6];
  return W.samplePath([prev, ...b.cam], b.camU ? b.camU(p) : eio(seg(p, w[0], w[1])), !!b.camU);   /* camU: a beat's own progress along its path; it halts at every keyframe */
}
function camAt(idx, p) { const b = beatList[idx].b; return b.camAt ? b.camAt(p, idx) : pathCam(b, p, idx > 0 ? endCam(idx - 1) : CAM0); }
const beatCam = { pos: new THREE.Vector3(), look: new THREE.Vector3() };
function setCam(c) { beatCam.pos.copy(c.pos); beatCam.look.copy(c.look); }

/* ---------- deterministic state: scene s with n settled beats, beat n at p (p == null: not started) ---------- */
let cur = { s: 0, n: 0, p: null, idx: -1, active: 0 };
function applyState(s, n, p) {
  let first = beatList.findIndex(e => e.s === s && e.k === n);
  if (first < 0) first = beatList.findIndex(e => e.s > s);
  if (first < 0) first = beatList.length;
  const running = p != null && first < beatList.length && beatList[first].s === s;
  OV.beat = 0; OV.trace = 1;                                  /* beat-driven overview: S14 b4 / S15 b1 set it */
  BOARDS.setApply(-1);
  for (let i = beatList.length - 1; i >= first; i--) { if (i === first && running) continue; beatList[i].b.f(0); }
  for (let i = 0; i < first; i++) { BOARDS.setApply(i); beatList[i].b.f(1); }
  if (running) { BOARDS.setApply(first); beatList[first].b.f(p); }
  if (running) setCam(camAt(first, p)); else setCam(endCam(first - 1));
  const idx = running ? first : first - 1;
  cur = { s, n, p: running ? p : null, idx, active: idx >= 0 ? beatList[idx].s : 0 };
  BOARDS.setApply(idx);
}
/* board tiers and focus: what the current beat touched is full; the rest of its scene dims; the previous scene stays in the distance */
const _zero = new THREE.Vector3();
function boardState(b) {
  const { idx, p, active } = cur;
  const tier = b.scene == null ? 'pers' : b.scene === active ? 'cur' : b.scene === active - 1 ? 'prev' : b.scene === active - 2 ? 'old2' : b.scene < active ? 'old' : 'future';
  const now = b.touched === idx ? 1 : 0;
  /* was: its focus when the previous beat ended. A board that beat showed and this one shows again keeps it (no dip on the click) */
  const was = b.keep || b.touched === idx - 1 || (b.touched === idx && b.prev === idx - 1) ? 1 : 0;
  const focus = b.keep ? 1 : p != null ? lerp(was, now, eio(seg(p, 0, .45))) : now;   /* keep: the scene's main board never dims */
  const enter = p != null && idx >= 0 && beatList[idx].k === 0 ? eio(seg(p, 0, .45)) : 1;   /* a scene's first beat: the scene before turns into its ghost over this */
  return { tier, focus, was, enter, ov: OV.kk, gk: b.scene != null ? (scenes[b.scene].ghost ?? 1) : 1 };   /* gk: a beat may fade the previous scene's ghosts (scene.ghost) */
}

/* ---------- overview: a plain pull-back over the real World. Nothing is moved: every scene has its own place
   (tools/contact.mjs --layout checks that no two scenes share one). The line is the track the camera really flew: the
   look-at point at the end of every beat, in talk order. Numbered badges sit on the track where each scene opens; the two
   legends at the sides of the stage name them. k = max(manual toggle, beat): S14 b4 pulls back, S15 b1 flies on to the references. ---------- */
const OV = { on: QS.has('overview'), k: QS.has('overview') ? 1 : 0, beat: 0, trace: 1, kk: 0 };   /* trace: how much of the track the opening beat has retraced */
const SEC = i => i < 3 ? 0 : i < 7 ? 1 : i < 13 ? 2 : 3;
const SECC = ['s1', 's2', 's3', 's4'].map(TV), SECH = ['s1', 's2', 's3', 's4'].map(TC);
const OVCAM = W.KF([-2.6, 64.7, 1051], [-2.6, 64.7, 0]);   /* fitted to the World's box, x -313..308, y -270..400, between the legends (tools/check-map.mjs prints it) */
const track = (() => {
  /* one point per scene, where its badge stands (the look-at point at the end of its first beat, or scene.home), in talk order
     (round 18: through every beat's look-at point the line drew loops and a V over the map). at[beat] = its scene's point */
  const pts = scenes.map((sc, s) => { if (sc.home) return W.V3(sc.home[0], sc.home[1], 0); const L = endCam(beatList.findIndex(e => e.s === s)).look; return W.V3(L.x, L.y, 0); });
  const own = scenes.map((_, s) => s), at = beatList.map(e => e.s);
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.5), TUB = 900, RAD = 8, g = new THREE.TubeGeometry(curve, TUB, 1.3, RAD, false);
  /* the tube is cut by arc length, the points are not evenly spaced: frac[j] = arc-length fraction at point j */
  const LN = curve.getLengths(4000), frac = pts.map((_, j) => LN[Math.round(j / (pts.length - 1) * 4000)] / LN[4000]);
  const col = new Float32Array(g.attributes.position.count * 3), c = new THREE.Color(), c2 = new THREE.Color();
  for (let j = 0, i = 0; j < g.attributes.position.count; j++) { const a = Math.floor(j / (RAD + 1)) / TUB; while (i < pts.length - 2 && a > frac[i + 1]) i++;
    c.setHex(SECH[SEC(own[i])]).lerp(c2.setHex(SECH[SEC(own[i + 1])]), clamp((a - frac[i]) / Math.max(1e-6, frac[i + 1] - frac[i]))); col[3 * j] = c.r; col[3 * j + 1] = c.g; col[3 * j + 2] = c.b; }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, depthWrite: false })); m.frustumCulled = false; W.crisp.add(m);
  const first = scenes.map((sc, s) => frac[at[beatList.findIndex(e => e.s === s)]]);   /* arc-length fraction at which each scene opens */
  const upTo = (idx, r = 1) => r * frac[at[Math.max(0, idx)]];   /* the line is drawn up to the current beat; r < 1 while the overview opens and retraces it */
  const count = a => Math.round(TUB * a) * RAD * 6;
  /* where each scene opens: the look-at point of its first beat, or scene.home when that point is already another scene's */
  const home = pts;
  return { m, upTo, count, first, home };
})();
scenes.forEach((sc, i) => {
  sc.el.style.setProperty('--sec', SECC[SEC(i)]);
  const h = track.home[i];
  sc.badge = BOARDS.board({ px: 64, w: 24,   /* ~36 px from the overview camera */ at: [h.x, h.y, 3], a: 0, cls: 'badge', z: 900, html: String(i + 1) });
  sc.badge.el.style.background = SECC[SEC(i)]; sc.badge.noLayout = true;
});
/* the legend: two lists fixed to the stage, not objects in the World */
const legend = [D('div', 'legend left', stage), D('div', 'legend right', stage)];
scenes.forEach((sc, i) => { sc.lg = D('div', 'lg', legend[i < 7 ? 0 : 1], `<span class="n" style="background:${SECC[SEC(i)]}">${i + 1}</span><span>${sc.title}</span>`); });
function setOverview(on) { OV.on = !!on; ui(); }
/* the overview is also a menu: a click on a badge or on a legend line, or a scene's number typed, opens that scene complete.
   OV.k takes over from a beat-driven overview (S14 b4), so the camera comes down from where it is instead of jumping */
function openScene(i) { OV.k = Math.max(OV.k, OV.kk); OV.on = false; show(i, true); }
scenes.forEach((sc, i) => [sc.badge.el, sc.lg].forEach(e => e.addEventListener('click', ev => { if (OV.kk <= 0.5) return; ev.stopPropagation(); openScene(i); })));
let digits = '', digitT = 0;
function typed(d) { digits += d; clearTimeout(digitT); const go = () => { const n = +digits; digits = ''; if (n >= 1 && n <= scenes.length) openScene(n - 1); };
  if (+digits * 10 > scenes.length) go(); else digitT = setTimeout(go, 700); }   /* "1" waits 0.7 s for a second digit; "2" … "9" and "10" … "15" go at once */

/* ---------- the Deck-compatible API ---------- */
function ui() {
  beatEl.textContent = QS.has('fps') ? `S${String(si + 1).padStart(2, '0')} b${run ? run.i + 1 : done}` : '';
  if (notesOn) { const N = scenes[si].notes || []; const i = run ? run.i : Math.max(0, done - 1); notesEl.textContent = N[i] || ''; }
  post();
}
function applyUpTo(n) { run = null; const B = scenes[si].beats; n = clamp(n, 0, B.length); applyState(si, n, null); done = n; ui(); }
function show(i, end) { i = clamp(i, 0, scenes.length - 1); si = i; applyUpTo(end ? scenes[si].beats.length : 0); }
function next() {
  const B = scenes[si].beats;
  if (run) { const i = run.i; run = null; applyState(si, i + 1, null); done = i + 1; ui(); }
  else if (done < B.length) { run = { i: done, t: 0 }; applyState(si, done, 0); ui(); }
  else if (si < scenes.length - 1) { show(si + 1); next(); }   /* no dead click at a scene boundary: the camera move is the next beat */
}
function prev() {
  if (run) { const i = run.i; run = null; applyUpTo(i); }
  else if (done > 0) applyUpTo(done - 1);
  else if (si > 0) show(si - 1, true);
}
function setAuto(a) { auto = !!a; }

/* ---------- presenter sync ---------- */
function post() {
  const msg = { type: 'state', si, done, run: run ? run.i : null, overview: OV.on, title: scenes[si].title, notes: scenes[si].notes, nextNotes: scenes[si + 1] ? scenes[si + 1].notes : [], nBeats: scenes[si].beats.length, nScenes: scenes.length };
  try { if (window.opener) window.opener.postMessage(msg, '*'); } catch (e) { }
}
addEventListener('message', ev => { const m = ev.data || {}; if (m.type === 'next') next(); else if (m.type === 'prev') prev(); else if (m.type === 'show') show(m.si, m.end); else if (m.type === 'overview') setOverview(!OV.on); else if (m.type === 'hello') post(); });

/* ---------- loop ---------- */
let last = null, tAmb = 0, fpsAcc = 0, fpsN = 0;
window.__fps = 0;
function loop(ts) {
  if (last === null) last = ts;
  const dt = Math.min(0.05, (ts - last) / 1000); last = ts; tAmb += dt;
  if (run) {
    const B = scenes[si].beats, b = B[run.i];
    run.t += dt * SPEED / (b.slow || 1);                     /* slow: this beat's tween lasts d / SPEED * slow seconds */
    if (run.t >= b.d) { const i = run.i; run = null; applyState(si, i + 1, null); done = i + 1; ui(); }
    else { const p = run.t / b.d; BOARDS.setApply(cur.idx); b.f(p); setCam(camAt(cur.idx, p)); cur.p = p; }
  }
  /* overview blend, a real-time tween (it is not a beat) */
  OV.k = clamp(OV.k + (OV.on ? dt / 2.2 : -dt / 1.6));
  const k = Math.max(eio(OV.k), OV.beat); OV.kk = k;
  BOARDS.layer.classList.toggle('thumbs', k > 0.5); stage.classList.toggle('ov', k > 0.5);
  W.cam.pos.lerpVectors(beatCam.pos, OVCAM.pos, k); W.cam.look.lerpVectors(beatCam.look, OVCAM.look, k);
  /* the track is retraced while the overview opens: it grows from the title to the current beat, and each scene's badge and
     legend line come on as it passes them */
  const reach = track.upTo(cur.idx, OV.beat > 0 ? OV.trace : OV.on ? seg(k, .3, .95) : 1);
  track.m.material.opacity = 0.85 * seg(k, 0, .3); track.m.visible = k > 0.003; track.m.geometry.setDrawRange(0, track.count(reach));
  legend.forEach(l => l.style.opacity = seg(k, .93, 1));   /* round 18: only once the map has formed, so no legend line sits on a board on the way up or down */
  scenes.forEach((sc, i) => { const seen = i <= cur.active ? clamp(1 + (reach - track.first[i]) / 0.012) : 0; sc.badge.show(seen * k); sc.lg.style.opacity = lerp(0.22, 1, seen); });
  BOARDS.setApply(cur.idx);
  tickObjects(tAmb);
  for (const f of TICKS) f(tAmb);
  W.render(tAmb);
  BOARDS.update(boardState);
  fpsN++; fpsAcc += dt; if (fpsAcc >= 0.5) { window.__fps = fpsN / fpsAcc; if (QS.has('fps')) fpsEl.textContent = window.__fps.toFixed(0) + ' fps'; fpsN = 0; fpsAcc = 0; }
  requestAnimationFrame(loop);
}
/* keys: Right, Space, PageDown forward; Left, PageUp back; o overview; in the overview, a scene's number opens it */
addEventListener('keydown', ev => {
  const k = ev.key;
  if (k === ' ' || k === 'ArrowRight' || k === 'PageDown') { ev.preventDefault(); next(); }
  else if (k === 'ArrowLeft' || k === 'PageUp') { ev.preventDefault(); prev(); }
  else if (k === 'ArrowDown') { ev.preventDefault(); show(si + 1); }
  else if (k === 'ArrowUp') { ev.preventDefault(); show(si - 1); }
  else if (k === 'o' || k === 'O') { setOverview(!OV.on); }
  else if (/^\d$/.test(k) && OV.kk > 0.5) { typed(k); }
  else if (k === 'n' || k === 'N') { notesOn = !notesOn; notesEl.classList.toggle('show', notesOn); stage.classList.toggle('notes-on', notesOn); ui(); }
  else if (k === 'f' || k === 'F') { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
});
addEventListener('click', ev => { if (ev.button === 0) next(); });
addEventListener('contextmenu', ev => { ev.preventDefault(); prev(); });
function fitStage() { const sc = Math.min(innerWidth / 1920, innerHeight / 1080); stage.style.transform = `translate(-50%,-50%) scale(${sc})`; }
addEventListener('resize', fitStage); fitStage();

/* deep link ?beat=S07b2 opens the World settled on that beat; ?overview starts in the overview */
(function start() {
  const m = /S(\d+)b(\d+)/i.exec(QS.get('beat') || '');
  if (m) { si = clamp(+m[1] - 1, 0, scenes.length - 1); applyUpTo(+m[2]); }
  else applyUpTo(0);
  requestAnimationFrame(loop);
})();
