/* w05-mrs.js — the rebuilt S05 "Minimal renormalon subtraction (MRS)" (ticket 06), storyboard v3 lines 82–89, wording
   from ticket 09's New column + the strings approved at ticket 06 step 1e (evidence/06/screen-text.md, decisions.md).
   Same contract as w04-renormalon.js: takes over scenes[4] (5 beats, the hosted d values), sits in the hosted DOM slot,
   draws everything from prog[] and the camera, replays the hosted reveals once (shadow), notes = the hosted notes.
   The staging of b3 follows the storyboard's physics review (#1, O1): the haze band that peels off the Borel cut goes
   into the Λ_R box and cancels against Λ_R's own haze there; it never enters 𝒥 and never just fades. */
(function () {
  'use strict';
  const I = 4, hosted = scenes[I];
  if (!(scenes.length === 15 && hosted && (hosted.el.querySelector('h1') || {}).textContent === 'Minimal renormalon subtraction (MRS)')) return;
  const RC = RebuildCore;
  const prog = [0, 0, 0, 0, 0];
  const { s, b } = scene('Scene 5 / 15 — Background III', 'Minimal renormalon subtraction (MRS)');
  s.classList.add('w-rebuilt', 'w-s05');
  hosted.el.replaceWith(s);
  const shadow = RC.once(() => hosted.beats.forEach(bt => (bt.z ? bt.z() : bt.f(0))));
  const fade = (el, t) => { t = clamp(t); el.style.opacity = eo(t); el.style.transform = `translateY(${8 * (1 - eo(t))}px)`; };
  const span = (p, html, cls) => D('span', cls || '', p, html);
  const ml = (g, x, y, html, o) => mlab(g, x, y, `<span>${html}</span>`, o); // see w04: keeps the spaces
  const rel = el => { const r = el.getBoundingClientRect(), q = s.getBoundingClientRect(), k = q.width / s.offsetWidth;
    return [(r.left + r.width / 2 - q.left) / k, (r.top + r.height / 2 - q.top) / k, (r.right - q.left) / k]; }; // centre, right edge; section coords

  /* ---- b1: headline (B1) and the |term| dots splitting into V_l and v_l − V_l ---- */
  const th = D('div', 'w-head', b, 'Isolate the leading renormalon into an explicit power correction, and resum the growth <span class="small">(§II B)</span>.');
  const zone = D('div', 'w-zone', b);
  const sp = E('svg', { viewBox: '0 0 700 180', width: 700, height: 180, class: 'w-svg w-split' }, zone);
  const { h: TH } = RC.termDots(), BX = l => 60 + 88 * l, BY = 170, SC = 110;
  const base = E('line', { x1: 30, y1: BY, x2: 660, y2: BY, stroke: 'var(--muted)', 'stroke-width': 1.5 }, sp);
  const gV = E('g', {}, sp), gR = E('g', {}, sp);
  // each dot (height |v_l α^{l+1}|) splits into V_l (white bar) and v_l − V_l (thin grey bar beside it, signed, to scale)
  const cols = TH.map((hh, l) => {
    const f = RC.remainder(l);
    return { l, hh, v: hh * (1 - (f || 0)), r: f === null ? null : hh * f, dot: E('circle', { cx: BX(l), cy: BY - SC * hh, r: 0, fill: 'var(--ink)' }, sp),
      V: E('rect', { x: BX(l) - 14, width: 20, y: BY, height: 0, fill: 'var(--ink)' }, gV),
      R: f === null ? null : E('rect', { x: BX(l) + 9, width: 9, y: BY, height: 0, fill: 'var(--muted)', opacity: 0 }, gR) };
  });
  const legend = E('g', { opacity: 0 }, sp);
  E('rect', { x: 470, y: 18, width: 16, height: 16, fill: 'var(--ink)' }, legend); ml(legend, 494, 26, '<i>V<sub>l</sub></i>', { size: 18, color: 'var(--ink)' });
  E('rect', { x: 470, y: 46, width: 16, height: 16, fill: 'var(--muted)' }, legend); ml(legend, 494, 54, '<i>v<sub>l</sub></i> − <i>V<sub>l</sub></i>', { size: 18 });
  ml(legend, 470, 82, 'schematic', { size: 15 });

  /* ---- b2: the three-box equation, units right (#7), α_s(μ′(r)) in both boxes (#8) ---- */
  const eq = D('div', 'm w-eqrow', zone);
  const lhs = span(eq, '𝒱<sub>R</sub>(<i>r</i>) = −<span class="frac"><span><i>C<sub>R</sub></i></span><span class="den"><i>r</i></span></span> {', 'w-t');
  const box = (cls, html, cap) => { const c = D('div', 'card w-box ' + cls, eq); D('div', 'w-bm', c, html); return { c, cap: D('div', 'small', c, cap) }; };
  const B1 = box('w-b1', '<span class="sum"><span class="ulim">2</span><span class="sg">Σ</span><span class="lim"><i>l</i>=0</span></span>(<i>v<sub>l</sub></i> − <i>V<sub>l</sub></i>) <i>α<sub>s</sub></i><sup>l+1</sup>(<i>μ′</i>(<i>r</i>))',
    'leading renormalon subtracted — no factorial growth at the leading rate. <i>V<sub>l</sub></i> = the renormalon part of <i>v<sub>l</sub></i>; the <i>v<sub>l</sub></i> are known to <i>l</i><sub>max</sub> = 2 (NNLO)');
  const plus1 = span(eq, '+', 'w-t');
  const B2 = box('w-b2', '<span class="frac"><span><i>V</i><sub>0</sub></span><span class="den">2<i>β</i><sub>0</sub></span></span>𝒥(<span class="w-arg w-arg-b"><i>b</i></span>,&thinsp;<span class="w-arg w-arg-y"><span class="frac"><span>1</span><span class="den">2<i>β</i><sub>0</sub><i>α<sub>s</sub></i>(<i>μ′</i>(<i>r</i>))</span></span></span>)',
    'the full renormalon series, Borel-summed in closed form (§II B)');
  const close = span(eq, '} +', 'w-t');
  const B3 = box('w-b3', '<span class="w-lr">Λ<sub>R</sub></span>', 'one power correction (Eq. A.15)');
  const free = D('div', 'small', B3.c, 'free of the leading-renormalon ambiguity');
  const lamR = B3.c.querySelector('.w-lr'), argB = B2.c.querySelector('.w-arg-b'), argY = B2.c.querySelector('.w-arg-y');
  const terms = [lhs, B1.c, plus1, B2.c, close, B3.c];
  const jn = D('div', 'small', b, '𝒥(<i>c</i>,<i>y</i>) = e<sup>−y</sup> Γ(−<i>c</i>) γ*(−<i>c</i>,−<i>y</i>) — an incomplete-gamma closed form');

  /* ---- b3: where the renormalon goes (T11) + the O1 label; b4: V_0 (T13); b5: Minimal (07) + What is new (T14) ---- */
  const row3 = D('div', 'w-row', b);
  const lr = D('div', 'card w-grow1', row3, '<h2>Where does the renormalon go?</h2><div class="big"><span class="mx">Λ<sub>R</sub></span> comes out <b>free of the leading-renormalon ambiguity</b>. For the <b>singlet</b> it is absorbed into the quark mass, tuned so that <span class="mx">2m<sub>Q</sub> + E</span> <b>matches the spin-averaged 1S states</b>. For <b>non-singlet</b> channels (baryons, tetraquarks) it is neglected — extending MRS there is a scheme choice, left to future work.</div>');
  const o1 = D('div', 'w-o1', row3, 'combined with <span class="mx">Λ<sub>R</sub></span>: <span class="mx">Λ<sub>R</sub></span> now free of the leading-renormalon ambiguity <span class="small">(§II B step 3)</span>');
  const row2 = D('div', 'w-row', b);
  const v0 = D('div', 'card', row2, '<h2>Normalization has converged</h2><div class="m num" style="font-size:22px"><i>V</i><sub>0</sub> = 0.771 <span class="small">(NNLO)</span> → 0.772 <span class="small">(N³LO)</span> ✓ <span class="small">(singlet, <i>n<sub>f</sub></i> = 4; §II B)</span></div>');
  const mn = D('div', 'card w-grow1', row2, '<h2>“Minimal” matters</h2><div class="big">Same <span class="ov">MS</span> inputs — the series is only reorganized: the renormalon tail is isolated and resummed (Komijani ’17; Kronfeld ’23).</div>');
  const nw = D('div', 'card w-grow1', row2, '<h2>What is new</h2><div class="big">Used for quark masses — all flavours except top (Fermilab/MILC/TUMQCD). First use in a pNRQCD bound-state problem.</div>');

  /* ---- on the Borel floor (glued to 3D points by place()) ---- */
  const borelLab = D('div', 'w-abs w-label', s, 'Borel plane (textbook) — leading IR singularity at <i class="mx">u</i> = ½, a branch point');
  const sumLab = D('div', 'w-abs w-label w-up', s, '<i class="mx">V<sub>l</sub></i> series (all <i class="mx">l</i> ≥ 0) Borel-summed up to the branch point → 𝒥 <span class="w-soft">(Eq. 4, A.11–A.13)</span>');
  const tagB = D('div', 'w-abs w-tag', s, 'exponent 1 + <i class="mx">b</i>'), tagY = D('div', 'w-abs w-tag', s, 'at <i class="mx">t</i> = 1/(2<i class="mx">β</i>₀) (<i class="mx">u</i> = ½)');
  const band = D('div', 'w-abs w-band', s);

  let at = {}; // positions of Board targets, section coords, refreshed on each draw
  function draw() {
    const [p1, p2, p3, p4, p5] = prog, hz = RC.haze(5, prog);
    fade(th, seg(p1, 0, 0.35));
    const fold = eio(seg(p2, 0, 0.3));
    cols.forEach(c => {
      c.dot.setAttribute('r', 6 * back(seg(p1, 0.1 + 0.025 * c.l, 0.25 + 0.025 * c.l)) * (1 - seg(p1, 0.35, 0.6)));
      const g = eo(seg(p1, 0.35 + 0.04 * c.l, 0.6 + 0.04 * c.l)), hv = SC * c.v * g;
      c.V.setAttribute('height', hv); c.V.setAttribute('y', BY - hv);
      if (c.R) { const hr = SC * c.r * g; c.R.setAttribute('height', Math.abs(hr)); c.R.setAttribute('y', hr > 0 ? BY - hr : BY); c.R.setAttribute('opacity', eo(seg(p1, 0.5, 0.8))); }
    });
    legend.setAttribute('opacity', eo(seg(p1, 0.7, 0.9)) * (1 - fold)); base.setAttribute('opacity', eo(seg(p1, 0.3, 0.45)) * (1 - seg(p2, 0.05, 0.25)));
    terms.forEach((e, i) => fade(e, seg(p2, 0.1 + 0.08 * i, 0.22 + 0.08 * i)));
    fade(B1.cap, seg(p2, 0.3, 0.45)); fade(B2.cap, seg(p2, 0.4, 0.55)); fade(B3.cap, seg(p2, 0.5, 0.62));
    // off screen the section is display:none (rebuild.css) and has no layout: keep the last measured targets
    if (s.offsetWidth) at = { b1: rel(B1.c), b2: rel(B2.c), lam: rel(lamR), close: rel(close), argB: rel(argB), argY: rel(argY), zone: rel(sp) };
    // the V_l column folds into the 𝒥 box, the remainders into box 1 (storyboard S05 b2: the whole column, from l = 0)
    if (at.zone) for (const [g, to] of [[gV, at.b2], [gR, at.b1]]) {
      const dx = (to[0] - at.zone[0]) * fold, dy = (to[1] - at.zone[1]) * fold;
      g.setAttribute('transform', `translate(${dx} ${dy}) translate(350 100) scale(${1 - 0.8 * fold}) translate(-350 -100)`);
      g.setAttribute('opacity', 1 - seg(p2, 0.22, 0.32));
    }
    fade(jn, seg(p2, 0.85, 1));
    lamR.style.setProperty('--hz', hz.lambdaR);
    fade(lr, seg(p3, 0.45, 0.65)); fade(free, seg(p3, 0.6, 0.8)); fade(o1, seg(p3, 0.7, 0.95)); // the card waits for the band's flight
    fade(v0, seg(p4, 0, 0.5)); fade(mn, seg(p5, 0, 0.45)); fade(nw, seg(p5, 0.5, 0.95));
    // b5: the derivation steps back while the closing cards come in (animation review #9)
    // (jn, lr, o1 got their opacity from fade() above; eq itself is never faded, only its terms are)
    const back5 = 1 - 0.5 * seg(p5, 0, 0.3);
    for (const e of [jn, lr, o1]) e.style.opacity = +e.style.opacity * back5;
    eq.style.opacity = back5;
  }
  function place({ w, project, toSection }) {
    const p2 = prog[1], hz = RC.haze(5, prog), P = u => { const q = project([RC.U * u, 0, 0]); return toSection(q.x, q.y); };
    const bp = P(0.5), mid = P(0.2);
    Object.assign(borelLab.style, { left: bp[0] + 'px', top: bp[1] + 26 + 'px', opacity: w });
    Object.assign(sumLab.style, { left: mid[0] + 'px', top: mid[1] - 30 + 'px', opacity: w * eo(seg(p2, 0.4, 0.55)) });
    if (!at.zone) return; // never measured yet: the tags and the band are not on screen before S05 has been shown
    // the two tags leave the branch point one after the other and dissolve into 𝒥's arguments as they land
    [[tagB, [bp[0] + 60, bp[1] - 52], at.argB, 0.5], [tagY, [bp[0] + 250, bp[1] - 52], at.argY, 0.56]].forEach(([el, from, to, a]) => {
      const t = eio(seg(p2, a, a + 0.16)), x = from[0] + (to[0] - from[0]) * t, y = from[1] + (to[1] - from[1]) * t;
      Object.assign(el.style, { left: x + 'px', top: y + 'px', opacity: w * eo(seg(p2, a - 0.06, a)) * (1 - seg(p2, a + 0.12, a + 0.17)),
        transform: `translate(-50%,-50%) scale(${1 - 0.45 * t})` });
    });
    const wait = [bp[0] + 100, bp[1] - 30 * eo(seg(p2, 0.72, 0.9))]; // starts at u = ½, on the cut (physics review #3)
    // it rises clear of the "} +": 45 px = the band's half width near the top (0.4 × 85 px) + its blur + a margin
    const f = RC.bandFlight(wait, at.lam, hz.bandPos, Math.max(at.lam[0], at.close[2] + 45));
    Object.assign(band.style, { left: f.x + 'px', top: f.y + 'px', opacity: w * hz.band, transform: `translate(-50%,-50%) scale(${f.scale})` });
  }
  scenes[I] = { el: s, beats: hosted.beats.map((bt, j) => ({ d: bt.d, f(p) { shadow(); prog[j] = p; draw(); } })), notes: hosted.notes.slice() };
  (window.W_REBUILT = window.W_REBUILT || []).push({ si: I, sc: 5, el: s, prog, place,
    // storyboard S04 out / S05 b2: the branch point stays on the floor, dimmed, below the S05 Board (#2)
    floor: q => ({ contour: eo(seg(q[1], 0.35, 0.55)), dim: 0.5 + 0.5 * seg(q[1], 0.35, 0.6) }) });
  draw();
})();
