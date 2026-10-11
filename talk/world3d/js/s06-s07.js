'use strict';
/* S06 Method I: μ′ inside a bound-state solver · S07 Method II: quantum Monte Carlo.
   The climb back from the bottom to the pair. */

/* ======================= S06 ======================= */
(function () {
  const SC = 5, DIST = 60, { P, V, KS } = FR(SC);
  const CORNER = TW(3, -48, -57, 0), HOME = V(LAY.towerX, LAY.pairY, LAY.pairZ);   /* from S04's corner to its S06 place */
  const k1 = BOARDS.board({ scene: SC, cls: 'panel', px: 640, w: U(640, DIST), at: P(-2.5, -6, 0), a: 0, html: `<h2>Why not ${I('μ')} = 1/${I('r')}?</h2><div class="big">A wavefunction samples <i>all</i> separations: at large ${I('r')} you walk straight out of the perturbative regime.</div>` });
  /* the μ′ plot: schematic curves (never Fig. 6, S06-F1) */
  const pl = BOARDS.board({ scene: SC, cls: 'panel', px: 700, w: U(700, DIST), at: P(-4, -30, 0), a: 0, html: `<h2>The interpolating scale</h2><svg viewBox="0 0 640 300" width="640" height="300" id="ps"></svg>
    <div class="small t" id="k2"><b>Smooth interpolation:</b> ${H.mup} ≈ ${I('λ')}/${I('r')} at short distance → saturates at <span class="num">1 GeV</span>. No kinks in the force: stable Monte Carlo.</div>
    <div class="small t" id="k3"><b>The cutoff matters:</b> <span class="num">1 GeV</span> keeps MRS <i>active</i> where bound states live. At <span class="num">2 GeV</span> MRS is rarely used and results fall back to fixed order.</div>
    <div class="small t" id="k4"><b>What we vary:</b> <span class="rc">${I('λ')} ∈ {½, 1, 2}</span>: the analogue of the usual factor-of-two variation. ${I('n')}${sub('f')} = 3 / 4 / 5, set by the lightest valence quark (${I('c')}/${I('b')}/${I('t')}).</div>` });
  const ps = pl.q('#ps');
  const X = r => 60 + (r / 1.2) * 560, Y = m => 260 - (m / 4.4) * 240;
  S('path', { d: 'M60 260 H630 M60 260 V20', class: 'ax' }, ps);
  const xl = S('text', { x: 626, y: 284, 'text-anchor': 'end', class: 'mt' }, ps); xl.innerHTML = `<tspan class="it">r</tspan> [fm]`;
  const yl = S('text', { x: 64, y: 16, class: 'mt' }, ps); yl.innerHTML = `<tspan class="it">μ</tspan> [GeV]`;
  S('rect', { x: 60, y: Y(0.35), width: 570, height: 260 - Y(0.35), fill: 'var(--lat)', 'fill-opacity': .1 }, ps);
  const inv = r => 0.1973 / r, mup = (r, l = 1, cut = 1) => Math.sqrt(cut * cut + (l * 0.1973 / r) ** 2);   /* Eq. 5, ħc = 0.1973 GeV fm */
  const path = (f, r0 = 0.05, r1 = 1.2) => { let d = ''; for (let r = r0; r <= r1 + 1e-9; r += 0.01) { const m = f(r); if (m > 4.3) continue;   /* a curve above the frame starts where it enters it: clamped, it drew a flat top with a corner */ d += (d ? 'L' : 'M') + X(r).toFixed(1) + ' ' + Y(m).toFixed(1) + ' '; } return d; };
  const c1 = S('path', { d: path(inv), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2.4 }, ps);
  const c1lab = S('text', { x: X(0.5) + 6, y: Y(inv(0.5)) - 8, class: 'it' }, ps); c1lab.innerHTML = `<tspan class="it">μ</tspan> = 1/<tspan class="it">r</tspan>`;
  const cutL = S('line', { x1: 60, y1: Y(1), x2: 630, y2: Y(1), stroke: 'var(--muted)', 'stroke-width': 1.2, 'stroke-dasharray': '7 5', opacity: 0 }, ps);
  const cutT = S('text', { x: 626, y: Y(1) - 17, 'text-anchor': 'end', class: 'mt', opacity: 0 }, ps);   /* above the curves that come down onto the line */ cutT.innerHTML = `<tspan class="it">μ</tspan><tspan baseline-shift="sub" font-size="12">cut</tspan> = 1 GeV: saturates`;
  const frm = S('text', { x: 250, y: 50, 'font-size': 22, opacity: 0 }, ps);   /* the plot's three lines of words share one left edge, in the free corner */ frm.innerHTML = `<tspan class="it">μ′</tspan>(<tspan class="it">r</tspan>; <tspan class="it">λ</tspan>) = √(<tspan class="it">μ</tspan><tspan baseline-shift="sub" font-size="14">cut</tspan>² + <tspan class="it">λ</tspan>²/<tspan class="it">r</tspan>²)`;
  /* the smooth cutoff only (round 19): the hard cutoff and the frozen α_s of App. A.2 are not shown, their dashed and dotted lines
     overloaded the plot. The 2 GeV comparison is the same smooth μ′ with μ_cut = 2 GeV */
  const g2 = S('path', { id: 'g2', d: path(r => mup(r, 1, 2)), fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.6, opacity: 0 }, ps);
  const g2T = S('text', { x: 626, y: Y(2) - 8, 'text-anchor': 'end',   /* at its curve's right end, above it: the left is the n_f line's */ class: 'mt', opacity: 0, 'font-size': 15 }, ps); g2T.innerHTML = `<tspan class="it">μ</tspan><tspan baseline-shift="sub" font-size="11">cut</tspan> = 2 GeV → fixed order`;
  /* the scale variation λ ∈ {½, 1, 2}: one half-transparent band from the λ = ½ to the λ = 2 curve, the systematic uncertainty
     (round 24; two separate lines before): the residual scale dependence, a diagnostic of scheme stability, not a truncation
     error (the paper's words). Its top is cut at the frame, like the curves; it lies under the λ = 1 line */
  const bandD = (() => { const R = []; for (let r = 0.05; r <= 1.2 + 1e-9; r += 0.01) R.push(r); const P = (r, l) => X(r).toFixed(1) + ' ' + Y(Math.min(mup(r, l), 4.3)).toFixed(1);
    return 'M' + R.map(r => P(r, 2)).join(' L') + ' L' + R.slice().reverse().map(r => P(r, 0.5)).join(' L') + ' Z'; })();
  const gh = [S('path', { id: 'lamband', d: bandD, fill: 'var(--ink2)', 'fill-opacity': 0.22, stroke: 'none', opacity: 0 })]; ps.insertBefore(gh[0], c1);
  const lamT = S('text', { x: 250, y: 84, class: 'it', opacity: 0 }, ps); lamT.innerHTML = `<tspan class="it">λ</tspan> ∈ {½, 1, 2}: the band probes<tspan x="250" dy="21">the residual scale dependence</tspan>`;   /* two lines: on one it ran out of the panel */
  const nfT = S('text', { x: 250, y: 130, class: 'mt', opacity: 0 }, ps); nfT.innerHTML = `<tspan class="it">n</tspan><tspan baseline-shift="sub" font-size="12">f</tspan> = 3 / 4 / 5 ← <tspan class="it">c</tspan> / <tspan class="it">b</tspan> / <tspan class="it">t</tspan>`;
  pl.measure();
  const nfChips = BOARDS.board({ scene: SC, px: 440, w: U(440, DIST), at: P(-38, -29.6, 4), a: 0,   /* one row, between the |ψ|² caption and the place the S07 cloud's caption takes */ html: `<span class="chip">${I('n')}${sub('f')} = 3 ← ${I('c')}</span> <span class="chip">${I('n')}${sub('f')} = 4 ← ${I('b')}</span> <span class="chip">${I('n')}${sub('f')} = 5 ← ${I('t')}</span>` });
  scenes.push({
    id: 'S06', title: 'Making MRS work inside a bound-state solver', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: climb back to the pair. A ruler runs out from one quark along the axis; the other quark walks out along it, past the
         point where μ = 1/r would fall below Λ_QCD; the ruler takes the gauge's zones, the gauge reads α_s (objects.js PSI); the r²|ψ|² hill sits on it */
      { d: 18, cam: [KS([-18, -3, 56], [-18, -21, 0])], win: [0, .6], f(p) {   /* one smooth zoom-pan up from S05: the two hand-placed legs made the fastest pan of the talk */
        window.__S05.lam1.show(1 - seg(p, 0, .2));
        const u = eio(seg(p, 0, .45)); PAIR.g.position.lerpVectors(CORNER, HOME, u); PAIR.st.scale = lerp(0.45, 1, u); PAIR.st.bOp = 1;
        PSI.st.rule = seg(p, .25, .45); PSI.st.clip = 0;
        /* B comes in to 0.2 fm, then walks out with μ = 1/r: past 0.56 fm α_s blows up (the S06 version shows it, objects.js PSI) */
        PAIR.st.sep = p < .55 ? lerp(7, 2.4, eio(seg(p, .45, .55))) : lerp(2.4, 15, eio(seg(p, .58, .88)));
        PSI.st.red = seg(p, .55, .58);   /* the gauge comes up once B is close (review 1.6: earlier it swung red, then teal) */ PSI.st.psi = seg(p, .7, .95);
        k1.show(seg(p, .5, .7));
      } },
      ...cut({ d: 18, f(p) {                                     /* a stop on μ = 1/r with its name, then the morph into μ′ */
        pl.show(seg(p, 0, .2)); const m = eo(seg(p, .22, .5)); c1.setAttribute('d', path(r => lerp(inv(r), mup(r), m)));
        c1lab.setAttribute('opacity', 1 - m); cutL.setAttribute('opacity', seg(p, .45, .6)); cutT.setAttribute('opacity', seg(p, .5, .65)); frm.setAttribute('opacity', seg(p, .55, .7));
        rv(pl.q('#k2'), seg(p, .8, 1));
        PSI.st.clip = eo(seg(p, .22, .5));                  /* in space: μ becomes μ′ at the pair, so α_s heals at once */
        PAIR.st.sep = p < .58 ? lerp(15, 2.4, eio(seg(p, .5, .58))) : lerp(2.4, 15, eio(seg(p, .6, .85)));   /* with μ′, B walks out again: nothing blows up */
      } }, [4, 14]),
      { d: 20, f(p) { pl.touch(); g2.setAttribute('opacity', seg(p, .1, .3)); g2T.setAttribute('opacity', seg(p, .25, .4)); rv(pl.q('#k3'), seg(p, .4, .7)); } },
      { d: 19, f(p) {
        pl.touch(); gh.forEach((g, i) => g.setAttribute('opacity', seg(p, .05 + i * .1, .25 + i * .1))); lamT.setAttribute('opacity', seg(p, .25, .4));
        nfT.setAttribute('opacity', seg(p, .55, .7)); nfChips.show(seg(p, .55, .7)); rv(pl.q('#k4'), seg(p, .5, .8));
      } },
    ],
  });
})();

/* ======================= S07 ======================= */
(function () {
  const SC = 6, DIST = 62, { P, V, KS } = FR(SC);
  const HOME6 = TW(5, LAY.towerX, LAY.pairY, LAY.pairZ), HOME7 = V(LAY.towerX + 20, LAY.pairY, LAY.pairZ);   /* as far right as the S06 boards' place allows: the whole cloud is on the stage */
  const cloudLab = BOARDS.board({ scene: SC, px: 540, w: 20, a: 0, cls: 'panel', html: `<span class="lbl">samples of |Ψ${sub('T')}|²: configurations, not particles</span><br><span class="tiny">one point = one sampled ${I('r')}${sub('12')}</span>` });
  const gfmcLab = BOARDS.board({ scene: SC, px: 540, w: 20, a: 0, cls: 'panel', html: `<span class="lbl">GFMC walkers = sampled configurations, imaginary time ${I('τ')}</span><br><span class="tiny">imaginary time τ = <span id="tau" class="num">0.00</span></span>` });
  TICKS.push(() => { const A = PAIR.posA(); if (WALK.follow !== false) WALK.pts.position.copy(A); cloudLab.set(A.x - 4, A.y - 16, A.z + 4); gfmcLab.set(A.x - 4, A.y - 16, A.z + 4); });   /* under the cloud: between the stage edge and the S06 plot */
  const vc = BOARDS.board({ scene: SC, cls: 'panel', px: 620, w: U(620, DIST), at: P(26, -19, 0), a: 0, html: `<h2>VMC: variational minimum</h2>
    <div class="m" style="font-size:26px">Δ${I('E')}${sub('VMC')} = min${sub(I('ω'))} ${frac('⟨Ψ' + sub('T') + '|' + I('H') + '|Ψ' + sub('T') + '⟩', '⟨Ψ' + sub('T') + '|Ψ' + sub('T') + '⟩')}</div>
    <svg viewBox="0 0 620 232" width="620" height="232" id="vs"></svg>` });
  const vs = vc.q('#vs');
  S('path', { d: 'M50 160 H600 M50 160 V14', class: 'ax' }, vs);
  const vX = u => 50 + 540 * u, vE = u => 150 - 300 * Math.pow(u - 0.62, 2);
  const sl = S('line', { x1: 50, y1: 204, x2: 600, y2: 204, stroke: 'var(--muted)', 'stroke-width': 1.5 }, vs);   /* room between the axis and the slider for the line of words */
  const knob = S('circle', { cx: vX(0.06), cy: 204, r: 7, fill: 'var(--ink)', opacity: 0 }, vs);
  const slT = S('text', { x: 600, y: 227, 'text-anchor': 'end', class: 'mt', 'font-size': 15 }, vs);   /* under the slider's end: beyond it, it left the panel and the stage */ slT.innerHTML = `<tspan class="it">ω</tspan> slider: Eq. (7), generic`;
  const hl = S('text', { x: 58, y: 26, class: 'mt' }, vs); hl.innerHTML = `⟨<tspan class="it">H</tspan>⟩`;
  const vcv = S('path', { d: (() => { let d = ''; for (let u = 0.02; u <= 1; u += 0.02) d += (d ? 'L' : 'M') + vX(u).toFixed(1) + ' ' + vE(u).toFixed(1) + ' '; return d; })(), fill: 'none', stroke: 'var(--ink2)', 'stroke-width': 2 }, vs);
  const ball = S('circle', { cx: vX(0.06), cy: vE(0.06), r: 8, fill: 'var(--strong)', opacity: 0 }, vs);
  const vmin = S('text', { x: vX(0.62), y: 182, 'text-anchor': 'middle', class: 'it', opacity: 0 }, vs); vmin.textContent = 'upper bound + starting point';
  vc.measure();
  const gc = BOARDS.board({ scene: SC, cls: 'panel', px: 620, w: U(620, DIST), at: P(26, -34, 0), a: 0, html: `<h2>GFMC: imaginary-time projection</h2>
    <div class="m" style="font-size:24px">e${sup('−' + I('H') + I('τ'))} |Ψ${sub('T')}⟩ → |Ψ${sub('0')}⟩ &nbsp; as ${I('τ')} → ∞</div>
    <svg viewBox="0 0 620 190" width="620" height="190" id="gs"></svg>` });
  const gs = gc.q('#gs');
  S('path', { d: 'M50 160 H600 M50 160 V14', class: 'ax' }, gs);
  const gxl = S('text', { x: 596, y: 182, 'text-anchor': 'end', class: 'mt it' }, gs); gxl.textContent = 'τ';
  const gyl = S('text', { x: 58, y: 26, class: 'mt' }, gs); gyl.innerHTML = `Δ<tspan class="it">E</tspan>(<tspan class="it">τ</tspan>)`;
  const E0 = 122;
  const gas = S('line', { x1: 50, y1: E0, x2: 600, y2: E0, stroke: 'var(--ink2)', 'stroke-width': 1.3, 'stroke-dasharray': '7 5' }, gs);
  const gX = u => 50 + 540 * u, gY = u => E0 - (E0 - 40) * Math.exp(-4.2 * u);           /* monotone from above (Eq. 10, S07-F3) */
  const gcv = S('path', { d: (() => { let d = ''; for (let u = 0; u <= 1; u += 0.02) d += (d ? 'L' : 'M') + gX(u).toFixed(1) + ' ' + gY(u).toFixed(1) + ' '; return d; })(), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2.4 }, gs);
  const gT = S('text', { x: 330, y: E0 + 24, 'text-anchor': 'middle', class: 'it', opacity: 0 }, gs); gT.textContent = 'descends monotonically from above onto ΔE₀';
  const gE = S('text', { x: 596, y: E0 - 6, 'text-anchor': 'end', class: 'mt' }, gs); gE.innerHTML = `Δ<tspan class="it">E</tspan>₀`;
  gc.measure();
  /* b3: trial states per channel, small 3D insets (S07-P5, S07-P6) */
  const insets = new THREE.Group(); W.scene.add(insets);
  const mkInset = (pts, links, x, o = {}) => {
    const g = new THREE.Group(); g.position.set(...P(x, -48, 6)); insets.add(g);
    const qs = pts.map((p, i) => { const q = W.quark(0.7, { glow: 1, c: W.CHARGE[pts.length][i] }); q.position.set(...p); g.add(q); return q; });
    const ls = links.map(([a, b, sh]) => { const l = W.link({ th: 0.1, opacity: 0.8, color: sh === 'a' ? TC('link-a') : sh === 'b' ? TC('link-b') : TC('link-c') }); l.set(qs[a].position, qs[b].position); g.add(l); return l; });
    g.setOpacity = a => { qs.forEach(q => q.material.opacity = a); ls.forEach(l => l.material.opacity = 0.8 * a); g.visible = a > 0.003; };
    g.setOpacity(0); return g;
  };
  const th = Math.sqrt(2 / 3), tet = [[1.6, 0, -1.6 / Math.SQRT2], [-1.6, 0, -1.6 / Math.SQRT2], [0, 1.6, 1.6 / Math.SQRT2], [0, -1.6, 1.6 / Math.SQRT2]];  /* regular tetrahedron: six equal links */
  const inM = mkInset([[-1.6, 0, 0], [1.6, 0, 0]], [[0, 1, 'a']], -32);
  const inB = mkInset([[-1.7, -1, 0], [1.7, -1, 0], [0, 1.9, 0]], [[0, 1, 'a'], [1, 2, 'a'], [2, 0, 'a']], -20);
  const inT = mkInset(tet, [[0, 2, 'a'], [1, 3, 'a'], [0, 3, 'b'], [1, 2, 'b'], [0, 1, 'c'], [2, 3, 'c']], -8);
  TICKS.push(t => { inM.rotation.y = t * 0.3; inB.rotation.y = t * 0.25; inT.rotation.y = t * 0.2; inT.rotation.x = 0.3; });
  const lM = BOARDS.board({ scene: SC, px: 260, w: U(260, DIST), at: P(-32, -53.5, 6), a: 0, html: `<div style="text-align:center"><span class="lbl">mesons</span><br>e${sup('−' + I('r') + '/' + I('a') + '₀')}: Coulombic</div>` });
  const lB = BOARDS.board({ scene: SC, px: 280, w: U(280, DIST), at: P(-20, -53.5, 6), a: 0, html: `<div style="text-align:center"><span class="lbl">baryons</span><br><span class="tc">product of pair Coulomb factors</span></div>` });
  const lT = BOARDS.board({ scene: SC, px: 330, w: U(330, DIST), at: P(-8, -54.3, 6), a: 0, html: `<div style="text-align:center"><span class="lbl">tetraquarks</span><br><span class="tc">Hylleraas forms</span><br><span class="tiny">a, b, c variational (Eq. 20)</span><br><span class="chip">3⊗${ov('3')}</span> <span class="chip">6⊗${ov('6')}</span></div>` });
  const w1 = BOARDS.board({ scene: SC, px: 640, w: U(640, DIST), at: P(-20, -43.6, 6), a: 0,   /* a heading stands over what it heads */ html: `<h2 style="text-align:center;margin:0">Trial wavefunctions: per channel</h2>` });
  BOARDS.topAlign(lT, lM, lB);                                  /* the three captions start on one line, under the lowest inset */
  const w2 = BOARDS.board({ scene: SC, cls: 'panel', px: 620, w: U(620, DIST), at: P(26, -50, 0), a: 0, html: `<h2>Errors</h2><div class="big" style="font-size:24px">Bootstrap errors: <span class="num">≤ 0.4 MeV</span> (most states) · <span class="num">6–11 MeV</span> (tetraquarks) · <span class="num">40–80 MeV</span> (η${sub('t')}, Ω${sub('ttt')}). Systematics: τ extent, step δτ.</div>` });   /* Tabs. I–V, storyboard S07 b4 */
  scenes.push({
    id: 'S07', title: 'Solving the few-body problem: quantum Monte Carlo', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: the |ψ|² bump dissolves into the cloud of samples in the relative coordinate; VMC breathes into its minimum */
      { d: 16, cam: [KS([0, -9, DIST], [0, -23, 0])], f(p) {
        PSI.st.psi = 1 - seg(p, 0, .3); PSI.st.red = 1 - seg(p, 0, .3); PSI.st.rule = 1 - seg(p, 0, .3); PSI.st.clip = 1;
        PAIR.g.position.lerpVectors(HOME6, HOME7, eio(seg(p, .1, .5)));
        scenes[SC - 1].ghost = 1 - seg(p, .5, .75);              /* the S06 boards carry the pan, then leave: they sat between the cloud and the S07 boards */
        PAIR.st.bOp = 1 - seg(p, .1, .35); window.__S03.singlet.show(1 - seg(p, .1, .35));
        const q = seg(p, .35, 1), w = 0.62 - 0.56 * Math.exp(-3.2 * q) * Math.cos(7.85 * q);   /* ω rolls into the minimum */
        WALK.set({ uOp: seg(p, .1, .4), uSpread: 0.75 + 0.5 * w, uTau: 0, uWeight: 0, uJit: 1, uRelax: 0 });
        cloudLab.show(seg(p, .3, .45)); gfmcLab.show(0);
        vc.show(seg(p, .2, .35)); knob.setAttribute('opacity', seg(p, .35, .45)); knob.setAttribute('cx', vX(w)); ball.setAttribute('opacity', seg(p, .35, .45)); ball.setAttribute('cx', vX(w)); ball.setAttribute('cy', vE(w) - 9);
        vmin.setAttribute('opacity', seg(p, .9, 1));
      } },
      /* b2: GFMC. Walkers take Gaussian steps and carry weights; as ΔE(τ) comes down from above the far walkers die out and the
         cloud tightens onto the ground state, in step with the curve */
      { d: 20, f(p) {
        const tau = seg(p, .15, .9), relax = (1 - Math.exp(-3.2 * tau)) / (1 - Math.exp(-3.2));
        WALK.set({ uTau: tau, uWeight: seg(p, .15, .5), uSpread: lerp(1.06, 0.8, relax), uRelax: relax });
        cloudLab.show(1 - seg(p, 0, .15)); gfmcLab.show(seg(p, .1, .25)); gfmcLab.q('#tau').textContent = (tau * 6).toFixed(2);
        gc.show(seg(p, 0, .2)); dl(gcv, seg(p, .15, .9)); gT.setAttribute('opacity', seg(p, .85, 1));
      } },
      { d: 18, cam: [KS([0, -19, 64], [0, -39, 0])], f(p) {
        WALK.set({ uOp: 1 - seg(p, 0, .3) }); PAIR.st.opacity = 1 - seg(p, 0, .3);   /* the quark and its cloud have done their part: they go as the camera leaves them (the new frame cut them at its top edge) */
        w1.show(seg(p, 0, .2)); [inM, inB, inT].forEach((g, i) => g.setOpacity(seg(p, .15 + i * .2, .35 + i * .2)));
        [lM, lB, lT].forEach((b, i) => b.show(seg(p, .2 + i * .2, .4 + i * .2)));
      } },
      { d: 12, f(p) { w2.show(seg(p, 0, .4)); } },
    ],
  });
  window.__S07 = { insets: [inM, inB, inT], labels: [lM, lB, lT, w1, w2, gfmcLab, cloudLab] };
})();
