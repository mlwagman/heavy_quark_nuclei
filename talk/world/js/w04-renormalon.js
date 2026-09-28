/* w04-renormalon.js — the rebuilt S04 "The renormalon problem" (ticket 06), storyboard v3 lines 72–80, wording from
   ticket 09's New column + the strings approved at ticket 06 step 1e (evidence/06/screen-text.md, decisions.md).
   Loaded after the hosted js/s04-*.js and before js/99-engine.js: it takes over scenes[3] (same beat count, same
   beat durations d in slot order, so the narrated schedule, the presenter and the capture shim are unchanged) and
   puts its own section in the hosted section's DOM slot. Beat order follows story edit E2 (symptom b3, origin b4);
   the notes follow E1/E2 only once E12 is set (ticket 06 step 7c, after ticket 11 is resolved).
   Every visual is a pure function of prog[] (per-beat progress) and the camera (RebuildCore), so a frame is the same
   however it was reached (click, back, applyUpTo, the recorder's virtual clock). No global rv() here: see shadow. */
(function () {
  'use strict';
  const I = 3, hosted = scenes[I];
  if (!(scenes.length === 15 && hosted && (hosted.el.querySelector('h1') || {}).textContent === 'The renormalon problem')) return;
  const RC = RebuildCore, E12 = true; // story edits E1 + E2 in the notes (step 7c: scene_04.md follows the rebuilt order)
  const prog = [0, 0, 0, 0];
  const { s, b } = scene('Scene 4 / 15 — Background II', 'The renormalon problem');
  s.classList.add('w-rebuilt', 'w-s04');
  hosted.el.replaceWith(s);
  const head = [s.querySelector('.eyebrow'), s.querySelector('h1')];
  /* the hosted scene's reveals take reveal variants from rv()'s global counter; replay them once, on first use */
  const shadow = RC.once(() => hosted.beats.forEach(bt => (bt.z ? bt.z() : bt.f(0))));

  const fade = (el, t) => { t = clamp(t); el.style.opacity = eo(t); el.style.transform = `translateY(${8 * (1 - eo(t))}px)`; };
  const op = (el, t) => el.setAttribute('opacity', eo(clamp(t)));
  const span = (p, html, cls) => D('span', cls || '', p, html);
  // the Deck's mlab() lays its html out as flex items, which drops the spaces at their edges: one span keeps them
  const ml = (g, x, y, html, o) => mlab(g, x, y, `<span>${html}</span>`, o);
  const L = D('div', 'col w-col', b), Rc = D('div', 'col w-col', b);
  L.style.width = '610px'; Rc.style.flex = '1';

  /* ---- b1: the series, factorial growth ---- */
  const cA = D('div', 'card', L, '<h2>The series</h2>');
  const eqA = D('div', 'm w-eq', cA);
  const tA = [
    span(eqA, '<i>V<sub>R</sub></i>(<i>r</i>) =&nbsp;'),
    span(eqA, '−<span class="frac"><span><i>C<sub>R</sub></i></span><span class="den"><i>r</i></span></span>'),
    span(eqA, '<span class="sum"><span class="sg">Σ</span><span class="lim"><i>l</i></span></span>'),
    span(eqA, '<i>v<sub>l</sub></i> <i>α<sub>s</sub></i><sup>l+1</sup>(<i>μ</i>)'),
    span(eqA, '&nbsp;+ <span class="w-lr">Λ<sub>R</sub></span> → 𝒱<sub>R</sub> <span class="small">(Eq. A.1)</span>', 'w-dim'),
  ];
  const lamR = eqA.querySelector('.w-lr');
  const cB = D('div', 'card', L, '<h2>Factorial growth</h2>');
  const eqB = D('div', 'm w-eq', cB, '<span><i>v<sub>l</sub></i> ~ <i>V</i><sub>0</sub> (2<i>β</i><sub>0</sub>)<sup>l</sup></span><span class="frac"><span>Γ(<i>l</i>+1+<i>b</i>)</span><span class="den">Γ(1+<i>b</i>)</span></span>');
  const asy = D('div', 'small', cB, 'the series is <b>asymptotic</b>, not convergent');

  /* ---- b2: |term| vs order (schematic, S04-I4) ---- */
  const cC = D('div', 'card w-over', Rc, '<h2>|term| vs order</h2>');
  const sC = E('svg', { viewBox: '0 0 812 200', width: 812, height: 200, class: 'w-svg' }, cC);
  const axC = E('g', {}, sC); axes(axC, 60, 160, 790, 8);
  ml(axC, 74, 16, '|<i>v<sub>ℓ</sub></i> <i>α<sub>s</sub></i><sup>ℓ+1</sup>| (schematic)', { size: 18 });
  ml(axC, 790, 176, '<i>ℓ</i> (order)', { anchor: 'end' });
  const { h: TH, used: USED } = RC.termDots();
  const X = l => 110 + 95 * l, Yc = hh => 160 - 150 * hh;
  const dots = TH.map((hh, l) => E('circle', { cx: X(l), cy: Yc(hh), r: 0, fill: 'var(--ink)' }, sC));
  const hmin = Math.min(...TH);
  const minLine = E('line', { x1: 60, y1: Yc(hmin), x2: 790, y2: Yc(hmin), stroke: 'var(--muted)', 'stroke-width': 1.5, 'stroke-dasharray': '7 5' }, sC); dl(minLine);
  const minLab = ml(sC, 470, Yc(hmin) + 18, 'minimal term ⇒ <i>δV</i> ~ Λ<sub>QCD</sub>', { size: 17 });
  const used = E('g', {}, sC);
  E('path', { d: `M${X(0) - 12} 168 v6 H${X(USED.at(-1)) + 12} v-6`, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.4 }, used);
  ml(used, (X(0) + X(USED.at(-1))) / 2, 188, 'used (<i>L</i> = 3)', { anchor: 'middle', size: 16 });
  const capC = ml(sC, 790, 195, 'schematic — terms shrink, reach a minimum, grow', { anchor: 'end', size: 15 });

  /* ---- b3: the symptom (M3) + the band gauge flying back from S02 + the fixed-order bands ---- */
  const cD = D('div', 'card', L, '<h2>Symptom</h2>');
  const m3 = D('div', 'big', cD, 'Varying <i class="m">μ</i> by 2× moves Ω<sub>bbb</sub> by up to <span class="num">~700 MeV</span> at NNLO <span class="small">(Tab. III)</span>; in binding energy at fixed <i class="m">m<sub>Q</sub></i>, LO does not even overlap NLO/NNLO <span class="small">(Figs. 1–2)</span>');
  const gauge = D('div', 'w-gauge', cD);
  const gRow = (label, value, cls) => {
    const r = D('div', 'w-grow', gauge); D('div', 'small', r, label);
    const line = D('div', 'w-gline', r), bar = D('div', 'w-gbar ' + cls, line), val = D('div', 'small num', line, value);
    return { bar, val };
  };
  const g1 = gRow('target precision', '~10 MeV', 'w-target'), g2 = gRow('fixed-order scale dependence', '~700 MeV (Ω<sub>bbb</sub> mass, NNLO, Tab. III)', 'w-hatch-nnlo');
  const GW = 260; // px for 700 MeV; the 10 MeV tick is to the same scale
  const cE = D('div', 'card', Rc, '<h2>Bands at successive orders — fixed order</h2>');
  const sE = E('svg', { viewBox: '0 0 812 170', width: 812, height: 170, class: 'w-svg' }, cE);
  const defs = E('defs', {}, sE);
  const BCOL = { LO: '#5E81B5', NLO: '#E19C24', NNLO: '#8FB032' };
  for (const [k, c] of Object.entries(BCOL)) {
    const pt = E('pattern', { id: `w04-h-${k}`, width: 7, height: 7, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    E('line', { x1: 0, y1: 0, x2: 0, y2: 7, stroke: c, 'stroke-width': 2.2 }, pt);
  }
  const axE = E('g', {}, sE); axes(axE, 60, 125, 790, 8);
  ml(axE, 74, 14, '|Δ<i>E</i>| (schematic)', { size: 17 });
  const XE = x => 90 + 610 * x, YE = y => 125 - 112 * y, XS = Array.from({ length: 41 }, (_, i) => i / 40);
  const BC = RC.bandCurves(XS);
  const bands = Object.keys(BCOL).map(k => {
    const path = E('path', { fill: `url(#w04-h-${k})`, stroke: BCOL[k], 'stroke-width': 1.5 }, sE);
    const ly = { LO: (BC.LO.lo.at(-1) + BC.LO.hi.at(-1)) / 2, NLO: BC.NLO.lo.at(-1), NNLO: BC.NNLO.hi.at(-1) }[k]; // NLO/NNLO overlap: labels at their outer edges
    const lab = E('text', { x: XE(1) + 10, y: YE(ly) + 6, 'font-size': 17, fill: BCOL[k], opacity: 0 }, sE);
    lab.textContent = k;
    return { k, path, lab };
  });
  const capE = ml(sE, 425, 144, 'binding energy vs <i>m<sub>Q</sub></i> (Figs. 1–2 left, schematic)', { anchor: 'middle', size: 15 });
  const loApart = ml(sE, 425, 163, 'LO apart — no meaningful uncertainty', { anchor: 'middle', size: 15, color: 'var(--ink)' });

  /* ---- b4: the origin (T5), V_R(r) with a constant-width haze, the force F(r) = +dV/dr as a hairline ---- */
  const cF = D('div', 'card', L, '<h2>Origin — infrared</h2>');
  const t5 = D('div', 'big', cF, 'Infrared sensitivity (momenta <i class="m">~ Λ</i><sub>QCD</sub>) ⇒ leading ambiguity <i class="m">δV ~ Λ</i><sub>QCD</sub> — a constant, same at every <i class="m">r</i>');
  const pole = D('div', 'small', cF, 'singlet: the same renormalon as the pole mass');
  const cG = D('div', 'card w-rel', Rc);
  const sG = E('svg', { viewBox: '0 0 812 150', width: 812, height: 150, class: 'w-svg' }, cG);
  const hzF = E('filter', { id: 'w04-blur', x: '-10%', y: '-50%', width: '120%', height: '200%' }, E('defs', {}, sG));
  E('feGaussianBlur', { stdDeviation: 3.5 }, hzF);
  const gV = E('g', {}, sG), gF = E('g', { transform: 'translate(420 0)' }, sG);
  axes(gV, 30, 128, 380, 30); axes(gF, 30, 128, 380, 30);
  ml(gV, 380, 142, '<i>r</i>', { anchor: 'end', size: 17 }); ml(gF, 380, 142, '<i>r</i>', { anchor: 'end', size: 17 });
  const vLab = ml(gV, 34, 12, 'perturbative sum <i>V<sub>R</sub></i>(<i>r</i>) <span class="small">(Eq. A.1)</span>', { size: 17, color: 'var(--ink)' });
  const fLab = ml(gF, 34, 12, '<i>F</i>(<i>r</i>) = <i>dV</i>/<i>dr</i> <span class="small">(Eq. A.3)</span>', { size: 17, color: 'var(--ink)' });
  const fLab2 = ml(gF, 40, 36, 'no constant (leading-renormalon) ambiguity', { size: 15 }); // under the title, clear of the curve (animation review #2)
  const RR = Array.from({ length: 60 }, (_, i) => 0.2 + 1.4 * i / 59), RF = RR.filter(r => r >= 0.36);
  // schematic Coulomb-like sum, rising toward 0 with r; F = +dV/dr (Eq. A.3): positive, falling
  const Vr = RR.map(r => -0.12 / r), FV = RF.map(r => 0.12 / (r * r));
  const xr = r => 36 + 335 * (r - 0.2) / 1.4, yV = v => 122 - 80 * (v + 0.6) / 0.54, yF = f => 122 - 64 * f / 0.93; // F tops out at y 58
  const HZ = RC.hazeBand(Vr, 0.13);
  const haze = E('path', { d: bandPath(xr, yV, RR, HZ.lo, HZ.hi), fill: 'var(--ink)', 'fill-opacity': 0.28, filter: 'url(#w04-blur)' }, gV);
  const vCurve = E('path', { d: 'M' + RR.map((r, i) => `${xr(r).toFixed(1)} ${yV(Vr[i]).toFixed(1)}`).join(' L'), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2.2 }, gV); dl(vCurve);
  const fCurve = E('path', { d: 'M' + RF.map((r, i) => `${xr(r).toFixed(1)} ${yF(FV[i]).toFixed(1)}`).join(' L'), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.1 }, gF); dl(fCurve);
  const deriv = D('div', 'small w-center', cG, '<i>taking a derivative removes the constant</i>');
  D('div', 'small w-note', cG, 'schematic'); // physics review #9: both curves are cartoons

  /* ---- labels glued to 3D points (placed by rebuild-space.js every frame) ---- */
  const pairLab = D('div', 'w-abs w-label w-left', s, 'colour singlet — schematic');
  const borelLab = D('div', 'w-abs w-label', s, 'Borel plane (textbook) — leading IR singularity at <i class="mx">u</i> = ½, a branch point');

  const olds = [cA, cB, cC, cD, cE];
  function draw() {
    const [p1, p2, p3, p4] = prog, hz = RC.haze(4, prog), cam = RC.cameraAt(4, prog), tilt = eio(seg(p4, 0.55, 1));
    // b1
    fade(cA, seg(p1, 0, 0.08)); tA.forEach((e, i) => fade(e, seg(p1, 0.04 + 0.1 * i, 0.16 + 0.1 * i)));
    tA[4].style.opacity = 0.55 * eo(seg(p1, 0.5, 0.7));
    lamR.style.setProperty('--hz', hz.lambdaR);
    fade(cB, seg(p1, 0.55, 0.62)); fade(eqB, seg(p1, 0.6, 0.85)); fade(asy, seg(p1, 0.85, 1));
    // b2
    fade(cC, seg(p2, 0, 0.1)); op(axC, seg(p2, 0.05, 0.2));
    dots.forEach((c, l) => c.setAttribute('r', Math.max(0, (l === dots.length - 1 ? 9 : 7) * back(seg(p2, 0.15 + 0.08 * l, 0.27 + 0.08 * l)))));
    ds(minLine, seg(p2, 0.75, 0.85)); op(minLab, seg(p2, 0.8, 0.9)); op(used, seg(p2, 0.85, 0.95)); op(capC, seg(p2, 0.9, 1));
    // b3
    fade(cD, seg(p3, 0, 0.1)); fade(m3, seg(p3, 0.03, 0.3));
    const gin = eo(seg(p3, 0.25, 0.55)); gauge.style.opacity = gin; gauge.style.transform = `translateY(${-140 * (1 - gin)}px)`;
    g1.bar.style.width = (GW * 10 / 700) + 'px'; g2.bar.style.width = (GW * eo(seg(p3, 0.45, 0.7))) + 'px';
    g1.val.style.opacity = eo(seg(p3, 0.45, 0.55)); g2.val.style.opacity = eo(seg(p3, 0.6, 0.75));
    fade(cE, seg(p3, 0.3, 0.42)); op(axE, seg(p3, 0.32, 0.45));
    bands.forEach(({ k, path, lab }, i) => {
      const t = eo(seg(p3, 0.45 + 0.1 * i, 0.65 + 0.1 * i)), n = Math.max(2, Math.round(t * XS.length));
      path.setAttribute('d', t > 0 ? bandPath(XE, YE, XS.slice(0, n), BC[k].lo.slice(0, n), BC[k].hi.slice(0, n)) : '');
      op(lab, seg(p3, 0.6 + 0.1 * i, 0.72 + 0.1 * i));
    });
    op(capE, seg(p3, 0.8, 0.9)); op(loApart, seg(p3, 0.88, 1));
    // b4
    fade(cF, seg(p4, 0, 0.08)); fade(t5, seg(p4, 0.02, 0.2)); fade(pole, seg(p4, 0.15, 0.25));
    fade(cG, seg(p4, 0.15, 0.25)); ds(vCurve, seg(p4, 0.2, 0.35)); op(vLab, seg(p4, 0.2, 0.3));
    haze.setAttribute('opacity', eo(seg(p4, 0.32, 0.45)));
    ds(fCurve, seg(p4, 0.38, 0.5)); op(fLab, seg(p4, 0.38, 0.48)); op(fLab2, seg(p4, 0.45, 0.55)); fade(deriv, seg(p4, 0.5, 0.58));
    // the camera tilts down to the Borel floor: the Board slides up with it, the earlier cards step back
    const sh = RC.boardShift(cam);
    for (const e of [...head, b]) e.style.transform = `translateY(${sh.toFixed(1)}px)`;
    olds.forEach(c => { c.style.opacity = +c.style.opacity * (1 - 0.6 * tilt); });
    draw.labels = { pair: 1 - seg(p4, 0.55, 0.8), borel: eo(seg(p4, 0.85, 1)) };
  }
  function place({ w, project, toSection }) {
    const a = project(RC.PAIR.map((v, i) => v + [-0.75, 0, 0][i])), q = toSection(a.x, a.y); // label's right edge: just left of the left quark (x −0.4, r 0.15)
    Object.assign(pairLab.style, { left: q[0] + 'px', top: q[1] + 'px', opacity: w * draw.labels.pair });
    const bp = project([RC.U * 0.5, 0, 0]), r = toSection(bp.x, bp.y); // same offset as w05's copy, so the two coincide in the glide
    Object.assign(borelLab.style, { left: r[0] + 'px', top: r[1] + 26 + 'px', opacity: w * draw.labels.borel });
  }
  scenes[I] = { el: s, beats: hosted.beats.map((bt, j) => ({ d: bt.d, f(p) { shadow(); prog[j] = p; draw(); } })),
    notes: E12 ? RC.e12(hosted.notes) : hosted.notes.slice() };
  (window.W_REBUILT = window.W_REBUILT || []).push({ si: I, sc: 4, el: s, prog, place, floor: () => ({ contour: 0, dim: 0 }), E12 });
  draw();
})();
