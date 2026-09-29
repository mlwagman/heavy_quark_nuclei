/* rebuild-core.js — pure logic of the rebuilt World scenes (ticket 06: S04 renormalon problem, S05 MRS). No DOM.
   Classic script in the page (global RebuildCore), CommonJS in Node for tools/test_rebuild_core.mjs. */
(function (root, factory) {
  const m = factory();
  if (typeof module === 'object' && module.exports) module.exports = m; else root.RebuildCore = m;
})(this, function () {
  'use strict';
  /* the Deck's own easing helpers (js/00-helpers.js), repeated here so this module stays pure */
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const seg = (p, a, b) => clamp((p - a) / (b - a));
  const eio = t => { t = clamp(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  /* S04 b2 |term| vs order, schematic (catalogue S04-I4): h_l ∝ Γ(l+1+b)/Γ(1+b) q^l (the Eq. A.6 shape, b ≈ 0.37),
     q picked so the minimum sits at l = 2 (storyboard #15); heights in units of the plot frame, the last one > 1.
     used = the orders the paper truncates at (NNLO, L = 3) */
  function termDots(n = 7, b = 0.37, q = 0.33, h0 = 0.62) {
    const h = [h0];
    for (let l = 1; l < n; l++) h.push(h[l - 1] * (l + b) * q);
    return { h, used: [0, 1, 2] };
  }
  /* S05 b1: v_l − V_l as a signed fraction of v_l, for the L = 3 orders the paper uses (catalogue S05-F4). To scale:
     +23, −14, +1 % from the paper's V_0 = 0.771 (n_f = 4), computed by the ticket 06 physics review (#21) */
  const REM = [0.23, -0.14, 0.01];
  const remainder = l => l < REM.length ? REM[l] : null;
  /* S04 b3 fixed-order bands, binding energy vs m_Q (Figs. 1-2 left, schematic): x = log m_Q in [0,1], y in frame
     units. Centres flat and widths constant (no trend claimed); LO apart below, NLO and NNLO overlap */
  // centre, half-width at x=0 and x=1: constant, since how the widths run with m_Q depends on the axes (physics review #16)
  const BANDS = { LO: [0.18, 0.05, 0.05], NLO: [0.46, 0.09, 0.09], NNLO: [0.56, 0.13, 0.13] };
  function bandCurves(xs) {
    const out = {};
    for (const [k, [c, w0, w1]] of Object.entries(BANDS)) {
      const w = xs.map(x => w0 + (w1 - w0) * x);
      out[k] = { lo: w.map(v => c - v), hi: w.map(v => c + v) };
    }
    return out;
  }
  /* White hazes (storyboard O1, settled by its physics review #1). prog = per-beat progress of the scene (0..1).
     lambdaR: Λ_R's own haze, from S04 b1 on. band: the haze strip that peels off the Borel cut in S05 b2;
     bandPos 0 = over the floor, 1 = inside the Λ_R box (S05 b3). The two cancel only once the band sits in the box,
     so the band never simply fades (that would read "renormalon-free"). 𝒥 is never hazy. */
  function haze(sc, prog) {
    if (sc === 4) return { lambdaR: seg(prog[0], 0.55, 0.85), band: 0, bandPos: 0, J: 0 };
    const cancel = seg(prog[2], 0.5, 0.75);
    return { lambdaR: 1 - cancel, band: seg(prog[1], 0.72, 0.9) * (1 - cancel), bandPos: eio(seg(prog[2], 0.05, 0.45)), J: 0 };
  }
  /* S05 b3: where the band is on its flight a (over the cut) → b (Λ_R), t = bandPos; scale of its 170 px width.
     Along the floor, a rounded turn, straight up in column xc (right of the "} +"), then the last step over to Λ_R,
     shrinking to the size of Λ_R's own haze: a straight line clipped the 𝒥 box, and the band must never enter 𝒥
     (physics review round 2, R2-1) */
  function bandFlight(a, b, t, xc = b[0]) {
    const sx = Math.sign(xc - a[0]), sy = Math.sign(b[1] - a[1]), h = Math.abs(xc - a[0]), v = Math.abs(b[1] - a[1]), e = Math.abs(xc - b[0]);
    const R = Math.min(h, v, 120), q = Math.PI * R / 2, L = h - R + q + v - R + e, k = t * L; // straight, quarter turn, up, over
    const scale = 1 - 0.65 * t;
    if (k <= h - R) return { x: a[0] + sx * k, y: a[1], scale };
    if (k <= h - R + q) { const th = (k - h + R) / R; return { x: xc - sx * R * (1 - Math.sin(th)), y: a[1] + sy * R * (1 - Math.cos(th)), scale }; }
    if (k <= L - e) return { x: xc, y: b[1] - sy * (L - e - k), scale };
    return { x: b[0] + Math.sign(xc - b[0]) * (L - k), y: b[1], scale };
  }
  /* S04 b4: the haze around the perturbative sum V_R(r) has one width at every r (catalogue S04-P4, never S04-F2) */
  const hazeBand = (ys, w) => ({ lo: ys.map(y => y - w / 2), hi: ys.map(y => y + w / 2) });
  /* ---- the series level of the Zoom loop (storyboard A). World units: the Borel floor is the plane y = 0, its
     u-axis runs along x with x = 8u (u = ½, the branch point, at x = 4); the Boards stand far behind, facing the
     camera. The camera is the same three.js PerspectiveCamera (vertical fov, lookAt with up = +y) in every file:
     rebuild-space.js copies these numbers into THREE, and the DOM labels use project() below, so they agree. */
  const U = 8, PAIR = [4.35, 9.67, 0], BOARD = [2, 6, -60]; // pair: top band, right of the title (x ≈ 0.65 W)
  const VIEW = {
    level: { pos: [2, 6, 12], target: [2, 6, 0], fov: 40 },        // S04 b1-b3: Boards, the pair small top right
    tilt04: { pos: [1.5, 4, 8], target: [1.5, 0.8, -6], fov: 40 },  // S04 b4 end: tilted down onto the Borel floor
    s05: { pos: [4.36, 4, 8], target: [4.36, 0.8, -6], fov: 40 },   // S05: trucked right, branch point below the Board
  };
  const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
  const lerpCam = (a, b, t) => ({ pos: lerp(a.pos, b.pos, t), target: lerp(a.target, b.target, t), fov: a.fov + (b.fov - a.fov) * t });
  /* camera of a rebuilt scene as a pure function of its beats' progress (the tilt runs in the second half of S04 b4) */
  const cameraAt = (sc, prog) => sc === 4 ? lerpCam(VIEW.level, VIEW.tilt04, eio(seg(prog[3], 0.55, 1))) : VIEW.s05;
  /* parts = [{w, cam}], w = how far each rebuilt section is on screen (1 − |translateX| / width, 0 when hidden).
     S04 → S05 glide: weights 1→0 and 0→1, so the camera trucks with the Boards; to a hosted scene the 3D fades out */
  function blend(parts) {
    const s = parts.reduce((a, p) => a + p.w, 0);
    if (!s) return { cam: parts[0] ? parts[0].cam : VIEW.level, presence: 0 };
    let cam = parts[0].cam, acc = parts[0].w;
    for (const p of parts.slice(1)) { acc += p.w; if (p.w) cam = lerpCam(cam, p.cam, p.w / acc); }
    return { cam, presence: Math.min(1, s) };
  }
  const sub = (a, b) => a.map((v, i) => v - b[i]), dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = a => { const l = Math.hypot(...a); return a.map(v => v / l); };
  /* world point → window pixels (x right, y down), as THREE.PerspectiveCamera + lookAt would draw it */
  function project(p, cam, w, h) {
    const z = unit(sub(cam.pos, cam.target)), x = unit(cross([0, 1, 0], z)), y = cross(z, x), d = sub(p, cam.pos);
    const c = [dot(d, x), dot(d, y), dot(d, z)], t = Math.tan(cam.fov * Math.PI / 360);
    return { x: (1 + c[0] / -c[2] / (t * w / h)) * w / 2, y: (1 - c[1] / -c[2] / t) * h / 2, front: c[2] < 0 };
  }
  /* how far the S04 Board slides up (stage px, 900 tall) while the camera tilts: it stands in the same space */
  const boardShift = cam => (project(BOARD, cam, 16, 9).y / 9 - 0.5) * 900;
  /* Story edits E1 + E2 (storyboard/story-edits.md, approved 2026-09-28), for the rebuilt S04 only: the symptom beat
     moves before the origin beat (bodies swap, the <b>bN.</b> prefixes stay in slot order) and the origin beat gains
     the E1 sentence. Throws if the hosted notes no longer read as expected, so a changed Deck is never re-ordered blind. */
  const E1 = 'In the Borel plane, it shows up as a branch point at u equals one half.';
  function e12(notes) {
    const split = n => { const m = /^(<b>b\d\.<\/b>)\s*(.*)$/s.exec(n); if (!m) throw new Error('e12: no <b>bN.</b> prefix'); return m.slice(1); };
    const [p3, origin] = split(notes[2]), [p4, symptom] = split(notes[3]);
    if (!origin.startsWith('The origin is infrared') || !symptom.startsWith('And here is the symptom'))
      throw new Error('e12: hosted S04 b3/b4 are not the origin/symptom beats (already applied, or the Deck changed)');
    return [notes[0], notes[1], `${p3} ${symptom}`, `${p4} ${origin} ${E1}`];
  }
  /* 3D lifecycle (the ticket-04 WorldSpace): want = the current scene is a rebuilt one with 3D; on = started;
     gliding = the Deck glide (glTimer) runs; held = a backup clip covers the stage */
  const spacePlan = ({ want, on, gliding, held }) =>
    want && !on && !held ? 'start' : !want && on && !gliding ? 'stop' : 'none';
  /* rv() in js/00-helpers.js deals reveal variants from a global counter in call order: a rebuilt scene replays its
     hosted scene's beat-zero functions once, on first use, so every later hosted scene keeps its Deck look */
  const once = f => { let done = false; return () => { if (!done) { done = true; f(); } }; };
  /* backup-clip marks (clips/manifest.js contract): log = the recorder's frames [{t, done, run}]; mark k = the first
     frame where beat k is settled (k = 0: the opening); each settled state must stay on screen ≥ 0.1 s after its mark,
     because the World stops the clip at the mark and may overshoot by a frame */
  function marksFrom(log, n) {
    const marks = [];
    for (let k = 0; k <= n; k++) {
      const i = log.findIndex(f => f.done === k && f.run === null);
      if (i < 0) throw new Error(`marksFrom: beat ${k} never settles in the recording`);
      const j = log.findIndex((f, x) => x > i && !(f.done === k && f.run === null));
      const until = j < 0 ? log[log.length - 1].t : log[j - 1].t;
      if (until - log[i].t < 0.1 - 1e-9) throw new Error(`marksFrom: beat ${k} hold ${(until - log[i].t).toFixed(3)} s < 0.1 s`);
      marks.push(log[i].t);
    }
    return marks;
  }
  return { _ease: { clamp, seg, eio }, marksFrom, once, spacePlan, E1, e12, termDots, remainder, bandCurves, haze, bandFlight, hazeBand, cameraAt, project, blend, boardShift, PAIR, U, VIEW };
});
