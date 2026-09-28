/* rebuild-space.js — drives the World's 3D layer for the rebuilt scenes (ticket 06), on top of the Deck engine's bare
   globals (si, glTimer, tAmb), which it only reads. Every animation frame it
     · weighs each rebuilt section by how far it is on screen (1 − |translateX| / width: the Deck glide),
     · blends their cameras (RebuildCore.blend: S04 → S05 trucks, leaving to a hosted scene fades the 3D out),
     · starts / stops WorldSpace (RebuildCore.spacePlan: stop only once the glide has ended; nothing under a clip),
     · draws the crisp Borel-floor overlay (#w-floor-overlay: u-axis, contour, branch point, cut; never bloomed),
     · lets each rebuilt scene glue its DOM labels to 3D points (place()).
   Rebuilt scenes register themselves in window.W_REBUILT = [{si, sc, el, prog, floor?(prog), place?(ctx)}]. */
(function () {
  'use strict';
  const R = window.W_REBUILT || [];
  if (!R.length) return;
  const RC = RebuildCore, L = SeriesLevel, NS = 'http://www.w3.org/2000/svg';
  const space = document.getElementById('space'), backup = document.getElementById('backup'), stage = document.getElementById('stage');
  const ov = document.createElementNS(NS, 'svg');
  ov.id = 'w-floor-overlay'; space.after(ov); // above the canvas, below the stage (DOM order, no z-index)
  const mk = (t, a) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); ov.appendChild(e); return e; };
  const axis = mk('line', { class: 'axis' }), tick = mk('line', { class: 'axis' }), cut = mk('path', { class: 'cut' });
  const contour = mk('line', { class: 'contour' }), ring = mk('circle', { class: 'bp' });
  const ulab = mk('text', { class: 'ulab' }); ulab.textContent = 'u';
  let on = false, failed = false;
  // light scheme: the bloom would whiten the whole background (its threshold is below the light --bg), so the 3D
  // stays off there; the Boards and the crisp floor overlay still show (ticket 06: the stage runs dark)
  const light = () => getComputedStyle(document.body).backgroundColor.match(/\d+/g).slice(0, 3).reduce((a, v) => a + +v, 0) > 384;
  const view = { scene: L.scene, camera: L.camera, tick: () => update() };

  const txOf = el => { const t = getComputedStyle(el).transform; return t && t !== 'none' ? new DOMMatrixReadOnly(t).m41 : 0; };
  function weight(el) {
    if (!(el.classList.contains('cur') || el.style.visibility === 'visible')) return 0;
    return el.offsetWidth ? Math.max(0, 1 - Math.abs(txOf(el)) / el.offsetWidth) : 0; // 0 = not rendered (display:none)
  }
  function drawOverlay(cam, presence, fl) {
    const w = innerWidth, h = innerHeight, k = stage.getBoundingClientRect().width / 1600;
    const P = u => RC.project([RC.U * u, 0, 0], cam, w, h);
    const a = P(-0.1), o = P(0), b = P(0.5), e = P(2.2);
    ov.style.opacity = presence;
    for (const [el, p, q] of [[axis, a, e], [contour, o, b]]) { el.setAttribute('x1', p.x); el.setAttribute('y1', p.y); el.setAttribute('x2', q.x); el.setAttribute('y2', q.y); }
    tick.setAttribute('x1', o.x); tick.setAttribute('x2', o.x); tick.setAttribute('y1', o.y - 7 * k); tick.setAttribute('y2', o.y + 7 * k);
    let d = `M${b.x + 12 * k} ${b.y}`; // the cut: a wavy line along the axis from the branch point to the edge
    for (let x = b.x + 12 * k; x < Math.min(e.x, w + 20); x += 2) d += ` L${x.toFixed(1)} ${(b.y + 4.5 * k * Math.sin((x - b.x) / (3 * k))).toFixed(1)}`;
    cut.setAttribute('d', d);
    ring.setAttribute('cx', b.x); ring.setAttribute('cy', b.y); ring.setAttribute('r', 11 * k);
    ring.style.opacity = 1 - 0.55 * fl.dim;
    contour.style.opacity = fl.contour;
    ulab.setAttribute('x', Math.min(P(1.3).x, w - 30 * k)); ulab.setAttribute('y', b.y - 14 * k);
    ov.style.setProperty('--k', k);
  }
  function update() {
    const all = R.map(r => ({ r, w: weight(r.el), cam: RC.cameraAt(r.sc, r.prog) })), parts = all.filter(p => p.w > 0);
    const { cam, presence } = RC.blend(parts);
    // floor state blended by on-screen weight, so the S04 → S05 glide dims the branch point smoothly
    const fl = { contour: 0, dim: 0 }, ws = parts.reduce((a, p) => a + p.w, 0) || 1;
    for (const p of parts) { const f = p.r.floor(p.r.prog); fl.contour += p.w * f.contour / ws; fl.dim += p.w * f.dim / ws; }
    const pairAlpha = parts.filter(p => p.r.sc === 4).reduce((a, p) => a + p.w, 0);
    const t = typeof tAmb === 'number' ? tAmb : 0;
    if (on) L.frame({ cam, pairAlpha, floorAlpha: presence, t });
    ov.style.display = presence > 0 ? 'block' : 'none';
    if (presence > 0) drawOverlay(cam, presence, fl);
    const rect = stage.getBoundingClientRect(), k = rect.width / 1600;
    for (const p of all) if (p.r.place) { // w = 0 too: an off-screen section's labels must go out, not stay stale
      const tx = txOf(p.r.el);
      p.r.place({ cam, w: p.w, t, toSection: (x, y) => [(x - rect.left) / k - tx, (y - rect.top) / k],
        project: q => RC.project(q, cam, innerWidth, innerHeight) });
    }
  }
  (function poll() {
    const act = RC.spacePlan({ want: !failed && !light() && R.some(r => r.si === si), on, gliding: glTimer !== null, held: backup.classList.contains('on') });
    if (act === 'start') {
      try { L.setBackground(getComputedStyle(document.body).backgroundColor); WorldSpace.start(view); on = true; }
      catch (e) { failed = true; } // no WebGL: the Boards and the crisp overlay still work (test R7)
    } else if (act === 'stop') { WorldSpace.stop(); on = false; }
    update();
    requestAnimationFrame(poll);
  })();
  window.__rebuiltSpace = { update, get on() { return on; }, get failed() { return failed; } };
})();
