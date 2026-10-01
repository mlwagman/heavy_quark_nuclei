'use strict';
/* S04 The renormalon problem · S05 Minimal renormalon subtraction. The bottom of the World: the series lives here,
   the Borel plane is the floor. */

/* world offset (x,y) of an element inside a board, so a chip can fly to a term */
function boardPoint(b, sel) {
  const e = b.q(sel); let x = e.offsetWidth / 2, y = e.offsetHeight / 2, n = e;
  while (n && n !== b.el) { x += n.offsetLeft; y += n.offsetTop; n = n.offsetParent; }
  const s = b.w / b.px; return W.V3(b.anchor.x + (x - b.px / 2) * s, b.anchor.y - (y - b.h / 2) * s, b.anchor.z + 0.5);
}
/* the Borel inset's two places: beside the V(r) plot (S04 b4), under the term dots (S05). [x, y, z, scale, label offset, contour-label offset] */
const BPA = [...TP(3, -32, -113, 0), 0.44, [31, -2.5], [31, -6.5]], BPB = [...TP(4, 86, -99, 0), 0.44, [0, -12], [0, -8.5]];   /* S04's place, S05's place */   /* S05: right under the closed-form box */
const borelAt = u => BOREL.place(lerp(BPA[0], BPB[0], u), lerp(BPA[1], BPB[1], u), 0, 0.44, [lerp(BPA[4][0], BPB[4][0], u), lerp(BPA[4][1], BPB[4][1], u)], [lerp(BPA[5][0], BPB[5][0], u), lerp(BPA[5][1], BPB[5][1], u)]);
/* svg hatch pattern in an order colour */
function hatchDef(svg, id, col) { const p = S('pattern', { id, width: 7, height: 7, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(-45)' }, S('defs', {}, svg)); S('line', { x1: 0, y1: 0, x2: 0, y2: 7, stroke: col, 'stroke-width': 2 }, p); return `url(#${id})`; }

/* ======================= S04 ======================= */
(function () {
  const SC = 3, DIST = 80, { P, V: VW, KS } = FR(SC);   /* V is this scene's potential V(r) */
  const CORNER = VW(-48, -57, 0), HOME = TW(2, LAY.towerX, LAY.pairY, LAY.pairZ);   /* the pair leaves its S03 place for S04's corner */
  const ser = BOARDS.board({ scene: SC, cls: 'panel', px: 700, w: U(700, DIST), at: P(-32, -70, 0), a: 0, html: `<h2>The series</h2>
    <div class="m" style="font-size:27px;display:flex;align-items:center;flex-wrap:wrap;column-gap:6px">
      <span class="ti">${H.VR}(${I('r')}) = −${frac(H.CR, I('r'))}</span><span class="ti">${sum(I('l'))}</span><span class="ti">${I('v')}${sub(I('l'))} ${H.as}${sup(I('l') + '+1')}(${I('μ')})</span>
      <span class="ti muted" style="font-size:22px">&nbsp;+ <span class="box haze" id="lr0" style="padding:2px 8px">${H.LR}</span> → 𝒱${sub('R')} <span class="tiny">(Eq. A.1)</span></span></div>
    <h2 class="t rc" style="margin-top:14px">Factorial growth</h2>
    <div class="m t" style="font-size:26px">${I('v')}${sub(I('l'))} ~ ${I('V')}${sub('0')} (2${I('β')}${sub('0')})${sup(I('l'))} ${frac('Γ(' + I('l') + '+1+' + I('b') + ')', 'Γ(1+' + I('b') + ')')}</div>
    <div class="small t" style="margin-top:6px">the series is <b>asymptotic</b>, not convergent</div>` });
  /* V(r) and F(r): the constant-width ambiguity band (S04-P4), the force free of it (Eq. A.3) */
  const vp = BOARDS.board({ scene: SC, cls: 'panel', px: 700, w: U(700, DIST), at: P(-32, -92, 0), a: 0, html: `<h2 id="vt">${H.VR}(${I('r')}) from the static pair</h2><svg viewBox="0 0 660 250" width="652" height="247" id="vs"></svg>
    <div class="t" id="orig"><div class="big" style="font-size:23px">Infrared sensitivity (momenta ~ ${H.LQCD}) ⇒ leading ambiguity ${I('δV')} ~ ${H.LQCD}: a constant, same at every ${I('r')}</div><div class="small" style="margin-top:4px">singlet: the same renormalon as the pole mass</div></div>` });
  const vs = vp.q('#vs');
  const XR = r => 60 + (r - 0.2) / 3.0 * 580, YV = v => 120 - v * 50, YF = f => 128 - f * 12;
  S('path', { d: 'M60 240 H640 M60 240 V20', class: 'ax' }, vs);
  const xt = S('text', { x: 636, y: 236, 'text-anchor': 'end', class: 'mt it' }, vs); xt.textContent = 'r';
  const V = r => -1 / r + 0.2, F = r => 1 / (r * r);
  const RR = []; for (let r = 0.42; r <= 3.2; r += 0.04) RR.push(r);
  const blur = S('filter', { id: 'w04blur' }, S('defs', {}, vs)); S('feGaussianBlur', { stdDeviation: 5 }, blur);
  const band = S('path', { d: RR.map((r, i) => `${i ? 'L' : 'M'}${XR(r).toFixed(1)} ${(YV(V(r)) - 13).toFixed(1)}`).join(' ') + ' ' + RR.slice().reverse().map(r => `L${XR(r).toFixed(1)} ${(YV(V(r)) + 13).toFixed(1)}`).join(' ') + 'Z', fill: 'var(--hazeband)', filter: 'url(#w04blur)', opacity: 0 }, vs);   /* the haze colour of S05: white was invisible on the light theme */
  const curve = S('path', { d: RR.map((r, i) => `${i ? 'L' : 'M'}${XR(r).toFixed(1)} ${YV(V(r)).toFixed(1)}`).join(' '), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2.4 }, vs);
  const fcurve = S('path', { d: RR.map((r, i) => `${i ? 'L' : 'M'}${XR(r).toFixed(1)} ${YF(F(r)).toFixed(1)}`).join(' '), fill: 'none', stroke: 'var(--ink2)', 'stroke-width': 1, opacity: 0 }, vs);
  const cl = S('text', { x: 330, y: 222, 'text-anchor': 'middle', opacity: 0 }, vs); cl.innerHTML = `perturbative sum <tspan class="it">V</tspan><tspan baseline-shift="sub" font-size="12">R</tspan>(<tspan class="it">r</tspan>) (Eq. A.1)`;
  const fl = S('text', { x: 636, y: 40, 'text-anchor': 'end', opacity: 0, fill: 'var(--ink2)' }, vs);   /* from the right end: started at mid-plot it ran out of the panel */ fl.innerHTML = `<tspan class="it">F</tspan>(<tspan class="it">r</tspan>) = <tspan class="it">dV</tspan>/<tspan class="it">dr</tspan> (Eq. A.3): no constant (leading-renormalon) ambiguity`;
  const fl2 = S('text', { x: 636, y: 62, 'text-anchor': 'end', opacity: 0, class: 'it mt' }, vs); fl2.textContent = 'taking a derivative removes the constant';
  const bl = S('text', { x: 440, y: 186, 'text-anchor': 'middle', opacity: 0, class: 'it', fill: 'var(--rose)' }, vs);   /* under the curve and its haze, not across them */ bl.innerHTML = `constant-width haze: <tspan class="it">δV</tspan> ~ Λ<tspan baseline-shift="sub" font-size="12">QCD</tspan>`;
  vp.measure();
  const bands = BOARDS.board({ scene: SC, cls: 'panel', px: 700, w: U(700, DIST), at: P(20, -104, 0), a: 0, html: `<h2>Bands at successive orders: fixed order</h2><svg viewBox="0 0 660 170" width="652" height="168" id="bs"></svg>
    <div class="tiny" style="margin-top:6px">binding energy vs ${H.mQ} (Figs. 1–2 left, schematic)</div>` });
  const bs = bands.q('#bs');
  S('path', { d: 'M50 160 H640 M50 160 V14', class: 'ax' }, bs);
  const byl = S('text', { x: 58, y: 24, class: 'mt' }, bs); byl.innerHTML = `|Δ<tspan class="it">E</tspan>| (schematic)`;
  const BC = { LO: 'var(--lo)', NLO: 'var(--nlo)', NNLO: 'var(--nnlo)' };
  /* LO sits apart; NLO and NNLO overlap (S04-P7, S04-F7) */
  const brects = [['LO', 80, 112, 200, 34], ['NLO', 330, 58, 220, 44], ['NNLO', 440, 40, 200, 40]].map(([k, x, y, w, h]) => {
    const r = S('rect', { x, y, width: 0, height: h, fill: hatchDef(bs, 'w04h' + k, BC[k]), stroke: BC[k], 'stroke-width': 1.5 }, bs);
    const t = S('text', { x: x - 8, y: y + h / 2 + 6, 'text-anchor': 'end', fill: BC[k], opacity: 0 }, bs); t.textContent = k;
    if (k === 'NNLO') { t.setAttribute('x', x + w); t.setAttribute('y', y - 7); }   /* over the right end of its band: beside it, it left the panel */
    return { r, t, w }; });
  const gapT = S('text', { x: 300, y: 135, 'text-anchor': 'start', class: 'it', opacity: 0 }, bs); gapT.textContent = 'LO apart: no meaningful uncertainty';
  bands.measure();
  const sparts = ser.qa('.ti');
  const GB = VW(20, -87.5, 0);
  scenes.push({
    id: 'S04', title: 'The renormalon problem', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: the pair turns side-on and shrinks into the corner; its V_R link unrolls into the r axis and the V(r) curve */
      { d: 15, cam: [KS([-4, -82, DIST], [-4, -82, 0])], f(p) {
        const u = eio(seg(p, 0, .5));
        TOWER.rest(seg(p, .45, .75));                          /* once the tower is out of the frame it goes to rest, for S06 and S07 */
        PAIR.g.position.lerpVectors(HOME, CORNER, u); PAIR.st.scale = lerp(1, 0.45, u); PAIR.st.sep = 7;
        window.__S03.vr.show(0); window.__S03.cf.show(0);
        const g3 = 1 - seg(p, 0, .25), g4 = 1 - seg(p, 0, .1); PAIR.st.third = g3; PAIR.st.tri = g4; window.__S03.cb1.show(g4); window.__S03.cb2.show(g4); window.__S03.cb3.show(g4); window.__S03.tri.show(g4);   /* the S03 baryon leaves here: its labels first, then the two quarks merge back into the antiquark */
        vp.show(seg(p, .2, .4)); dl(curve, seg(p, .3, .6)); cl.setAttribute('opacity', seg(p, .55, .7));
        band.setAttribute('opacity', 0); fcurve.setAttribute('opacity', 0); fl.setAttribute('opacity', 0); fl2.setAttribute('opacity', 0); bl.setAttribute('opacity', 0); rv(vp.q('#orig'), 0);
        ser.show(seg(p, .35, .5)); sparts.forEach((e, i) => rv(e, seg(p, .45 + i * .1, .6 + i * .1)));
        BOARDS.T(ser, '.t', seg(p, .8, 1));
        TERMS.b.set(...P(20, -70, 0)); TERMS.b.w = U(760, DIST); TERMS.b.show(0);
      } },
      { d: 18, f(p) { TERMS.b.show(seg(p, 0, .15)); TERMS.b.touch(); TERMS.reveal(p); } },
      /* b3: the S02 band gauge comes back; then the fixed-order band stack (the camera eases back so the stack's frame is whole) */
      { d: 13, cam: [KS([-4, -84, 84], [-4, -84, 0])], f(p) {
        GAUGE.set({ title: 'Symptom', tcls: 'rc', rows: [
          { lab: 'target precision', style: 'tick', val: '~10 MeV' },
          { lab: 'fixed-order scale dependence', cls: 'nnlo', style: 'hatch', w: 1.0, val: `~700 MeV (${H.Obbb} mass, NNLO, Tab. III)` },   /* Tab. III: 14.282(1)(706) */
        ], note: `Varying ${I('μ')} by 2× moves ${H.Obbb} by up to <span class="num rc">~700 MeV</span> at NNLO <span class="small">(Tab. III)</span>; in binding energy at fixed ${H.mQ}, LO does not even overlap NLO/NNLO <span class="small">(Figs. 1–2)</span>` });
        GAUGE.b.setV(GB); GAUGE.b.w = U(760, DIST);   /* in its place from the start: flown in from the S02 place it crossed the |term| board */ GAUGE.b.show(seg(p, 0, .15)); GAUGE.b.touch(); rv(GAUGE.b.q('#gnote'), seg(p, .35, .5));
        bands.show(seg(p, .45, .6)); brects.forEach((b, i) => { b.r.setAttribute('width', b.w * eo(seg(p, .5 + i * .12, .7 + i * .12))); b.t.setAttribute('opacity', seg(p, .55 + i * .12, .7 + i * .12)); });
        gapT.setAttribute('opacity', seg(p, .88, 1));
      } },
      /* b4: origin. The constant-width haze on V(r), the force free of it; the Borel plane appears as a small inset under the plot */
      { d: 20, win: [.45, .85], cam: [KS([-4, -87, 90], [-4, -87, 0])], f(p) {
        vp.touch(); rv(vp.q('#orig'), seg(p, 0, .2)); vp.q('#vt').innerHTML = p > 0.05 ? 'Origin: infrared' : `${H.VR}(${I('r')}) from the static pair`;
        band.setAttribute('opacity', seg(p, .15, .35)); bl.setAttribute('opacity', seg(p, .25, .4));
        fcurve.setAttribute('opacity', seg(p, .35, .5)); dl(fcurve, seg(p, .35, .55)); fl.setAttribute('opacity', seg(p, .45, .6)); fl2.setAttribute('opacity', seg(p, .5, .65));
        borelAt(0); const k = seg(p, .6, .85); BOREL.show(k); BOREL.label.show(seg(p, .8, 1)); BOREL.cLabel.show(0); BOREL.contourOp(0);
      } },
    ],
  });
})();

/* ======================= S05 ======================= */
(function () {
  const SC = 4, DIST = 80, { P, V, KS } = FR(SC);
  const head = BOARDS.board({ scene: SC, px: 1300, w: U(1300, DIST), at: P(76, -63, 0), a: 0, html: `<div style="font:italic 30px/1.35 var(--serif);text-align:center">Isolate the leading renormalon into <span class="ti rc">an explicit power correction</span>, and <span class="ti ac">resum the growth</span> <span class="small">(§II B)</span>.</div>` });
  const eq = BOARDS.board({ scene: SC, cls: 'panel', keep: true, px: 1040, w: U(1040, DIST), at: P(92, -79, 0), a: 0, html: `
    <div class="m" style="font-size:25px;display:flex;align-items:center;flex-wrap:wrap;column-gap:4px;row-gap:10px">
      <span class="ti">𝒱${sub('R')}(${I('r')}) = −${frac(H.CR, I('r'))} {</span>
      <span class="ti box" id="b1">${sum(I('l') + '=0', '2')} (${I('v')}${sub(I('l'))} − ${I('V')}${sub(I('l'))}) ${H.as}${sup(I('l') + '+1')}(${H.mup}(${I('r')}))</span>
      <span class="ti">+</span>
      <span class="ti box" id="b2">${frac(I('V') + sub('0'), '2' + I('β') + sub('0'))} 𝒥(<span id="argb">${I('b')}</span>, <span id="argy">${frac('1', '2' + I('β') + sub('0') + H.as + '(' + H.mup + '(' + I('r') + '))')}</span>)</span>
      <span class="ti">}</span>
      <span class="ti">+</span>
      <span class="ti box haze" id="b3" style="padding:8px 16px">${H.LR}</span>
    </div>
    <div style="display:flex;gap:18px;margin-top:12px">
      <div class="small t tc" id="c1" style="flex:1.3">leading renormalon subtracted: no factorial growth at the leading rate. ${I('V')}${sub(I('l'))} = the renormalon part of ${I('v')}${sub(I('l'))}; the ${I('v')}${sub(I('l'))} are known to ${I('l')}${sub('max')} = 2 (NNLO)</div>
      <div class="small t ac" id="c2" style="flex:1">the full renormalon series, Borel-summed in closed form (§II B)</div>
      <div class="small t rc" id="c3" style="flex:.8">one power correction (Eq. A.15)<br><span id="c3b" style="opacity:0">free of the leading-renormalon ambiguity</span></div>
    </div>
    <div class="small t" id="jl" style="margin-top:10px">𝒥(${I('c')},${I('y')}) = e${sup('−' + I('y'))} Γ(−${I('c')}) γ*(−${I('c')},−${I('y')}): an incomplete-gamma closed form</div>
    <div class="t lbl" id="cancel" style="margin-top:8px;text-align:right">same form as ${H.LR} (a constant) → combined: ${H.LR} now free of the leading-renormalon ambiguity (§II B step 3)</div>` });
  const hazeB = BOARDS.board({ px: 200, w: U(200, DIST), a: 0, html: `<div class="hazeband" style="position:relative;width:200px;height:26px"></div>` });
  const cancel = { show: a => rv(eq.q('#cancel'), a), set() { } };
  const where = BOARDS.board({ scene: SC, cls: 'panel', px: 800, w: U(800, DIST), at: P(76, -98, 0), a: 0, html: `<h2 class="rc">Where does the renormalon go?</h2>
    <div class="big" style="font-size:24px">${H.LR} comes out <b>free of the leading-renormalon ambiguity</b>. For the <b>singlet</b> it is absorbed into the quark mass, tuned so that 2${H.mQ} + ${I('E')} <b class="rc">matches the spin-averaged 1S states</b>. For <b>non-singlet</b> channels (baryons, tetraquarks) it is neglected: extending MRS there is a scheme choice, left to future work.</div>` });
  const lam1 = BOARDS.board({ px: 300, w: U(300, DIST), a: 0, html: `<span class="chip pin">Λ₁ absorbed into ${H.mQ}</span>` }); lam1.el.style.textAlign = 'center';
  const LAM1 = P(49, -97.4, 0);                                    /* where the chip waits for S08: at the panel's edge, beside "absorbed into the quark mass" */
  const v0 = BOARDS.board({ scene: SC, cls: 'panel', px: 520, w: U(520, DIST), at: P(114, -98, 0), a: 0, html: `<h2>Normalization has converged</h2>
    <div class="m num" style="font-size:24px">${I('V')}${sub('0')} = 0.771 <span class="small">(NNLO)</span> → 0.772 <span class="small">(N³LO)</span> <span class="tc">✓</span> <span class="small">(singlet, ${I('n')}${sub('f')} = 4; §II B)</span></div>` });   /* §II B prose, S05-P5 */
  const mn = BOARDS.board({ scene: SC, cls: 'panel', px: 600, w: U(600, DIST), at: P(70, -111, 0), a: 0, html: `<h2>“Minimal” matters</h2><div class="big" style="font-size:23px">Same ${ov('MS')} inputs: the series is only reorganized: the renormalon tail is isolated and resummed (Komijani ’17; Kronfeld ’23).</div>` });
  const nw = BOARDS.board({ scene: SC, cls: 'panel', px: 600, w: U(600, DIST), at: P(104, -111, 0), a: 0, html: `<h2 class="rc">What is new</h2><div class="big" style="font-size:23px">Used for quark masses: all flavours except top (Fermilab/MILC/TUMQCD). <span class="rc">First use in a pNRQCD bound-state problem.</span></div>` });
  const eparts = eq.qa('.ti'), b3 = eq.q('#b3'), b1e = eq.q('#b1'), b2e = eq.q('#b2');
  /* the dots that leave the |term| plot: the remainder (teal) into the first box, the renormalon column (amber) into the closed form */
  const flyR = BOARDS.board({ px: 120, w: U(120, DIST), a: 0, z: 300, html: `<div class="tc" style="font-size:30px;letter-spacing:3px;text-align:center">●●●</div>` });
  const flyV = BOARDS.board({ px: 260, w: U(260, DIST), a: 0, z: 300, html: `<div class="ac" style="font-size:30px;letter-spacing:3px;text-align:center">●●●●●●●</div>` });
  let PL = null, PB1 = null, PB2 = null, S1 = null, S2 = null;   /* targets inside the equation board (measured once it is laid out) */
  /* S1, S2: where the dots stop before they go in, outside the panel (on it they were written on the equation): above the first box and above the second */
  const targets = () => { if (!PL) { PL = boardPoint(eq, '#b3'); PB1 = boardPoint(eq, '#b1'); PB2 = boardPoint(eq, '#b2');
    const hw = eq.w / 2, hh = eq.h * eq.w / eq.px / 2; S1 = W.V3(PB1.x, eq.anchor.y + hh + 1.6, PB1.z); S2 = W.V3(PB2.x, eq.anchor.y + hh + 1.6, PB2.z); } };
  /* Λ_R singled out (S05 b2d, undone in b3): a rose ring on its box; the other two boxes and their captions at .45, still read */
  const lamFocus = k => { [b1e, b2e, eq.q('#c1'), eq.q('#c2'), eq.q('#jl')].forEach(e => { e.style.opacity = lerp(1, .45, k); });
    b3.style.outline = k > 0 ? `${(3 * k).toFixed(2)}px solid var(--rose)` : ''; b3.style.outlineOffset = '5px'; };
  scenes.push({
    id: 'S05', title: 'Minimal renormalon subtraction', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: headline; the |term| dots split into V_l and the remainder (l ≤ 2 only) */
      { d: 12, cam: [KS([73, -83, 90], [73, -84, 0])], f(p) {
        head.show(seg(p, 0, .3)); BOARDS.T(head, '.ti', seg(p, .2, .6));
        GAUGE.b.show(1 - seg(p, 0, .3));
        const u = eio(seg(p, 0, .5)); TERMS.b.lerpTo(TW(3, 20, -70, 0), V(42, -79, 0), u); TERMS.b.w = U(760, DIST) * lerp(1, 1.18, u); TERMS.b.show(1); TERMS.b.touch();   /* the dots grow into the main picture */
        TERMS.fold(0); TERMS.foldR(0); TERMS.split(seg(p, .3, 1));   /* split last: fold(0) alone would show every amber dot, split or not (they leaked into S04) */
        borelAt(u); BOREL.dimPoint(lerp(1, 0.45, seg(p, 0, .3)));
      } },
      /* b2, four stops. b2a: the three-box equation is written; the remainder dots (teal) leave the |term| plot and stop at the first box */
      { d: 10, cam: [KS([73, -82, 88], [73, -83, 0])], f(p) {
        eq.show(seg(p, 0, .2)); eparts.forEach((e, i) => rv(e, seg(p, .1 + i * .06, .3 + i * .06)));
        targets(); TERMS.b.show(1); TERMS.b.touch(); b3.classList.add('haze');
        TERMS.foldR(seg(p, .6, .72)); flyR.lerpTo(TERMS.at(TERMS.X(1), 190), S1, eio(seg(p, .66, .95))); flyR.show(seg(p, .64, .72));
      } },
      /* b2b: the teal dots go into the box (the series with the renormalon part subtracted); the renormalon column (amber, every
         order) gathers and stops at the closed form */
      { d: 7, f(p) {
        targets(); eq.touch(); TERMS.b.show(1); TERMS.b.touch();
        flyR.lerpTo(S1, PB1, eio(seg(p, 0, .2))); flyR.show(1 - seg(p, .05, .2)); b1e.style.borderColor = p > .1 ? 'var(--teal)' : '';
        rv(eq.q('#c1'), seg(p, .15, .4));
        TERMS.fold(seg(p, .35, .55)); flyV.lerpTo(TERMS.at(700, 110), S2, eio(seg(p, .5, .95))); flyV.show(seg(p, .45, .55));
      } },
      /* b2c: the amber dots go into the closed form; on the Borel inset the contour lights up to the branch point: Borel-summed */
      { d: 7, f(p) {
        targets(); eq.touch(); TERMS.b.show(1); TERMS.b.touch();
        flyV.lerpTo(S2, PB2, eio(seg(p, 0, .2))); flyV.show(1 - seg(p, .05, .2)); b2e.style.borderColor = p > .1 ? 'var(--amber)' : '';
        BOREL.contourOp(seg(p, .15, .45) * 0.9); BOREL.cLabel.show(seg(p, .25, .5));
        rv(eq.q('#c2'), seg(p, .3, .55)); rv(eq.q('#jl'), seg(p, .6, .85));
      } },
      /* b2d: the third term on its own. A rose ring on the Λ_R box while the other two boxes and their captions step back; its
         caption; and on the |term| plot, what the Borel sum cannot fix rises as a rose haze, the thing Λ_R will take in */
      { d: 6, f(p) {
        targets(); eq.touch(); TERMS.b.show(1); TERMS.b.touch(); b3.classList.add('haze');
        lamFocus(seg(p, 0, .35)); rv(eq.q('#c3'), seg(p, .25, .55));
        hazeB.setV(TERMS.at(600, 70)); hazeB.show(seg(p, .55, .9)); eq.q('#c3b').style.opacity = 0; cancel.show(0);
      } },
      /* b3: the haze band slides into the Λ_R box and cancels its own haze; Λ_R turns sharp. Then the chip parks for S08 */
      { d: 18, cam: [KS([73, -87, 90], [73, -89, 0])], f(p) {
        targets(); eq.touch();
        const u = eio(seg(p, 0, .4)); hazeB.lerpTo(TERMS.at(600, 70), PL, u); hazeB.show(1 - seg(p, .4, .55));
        const k = seg(p, .4, .6); b3.style.boxShadow = k < 1 ? `0 0 ${18 * (1 - k)}px ${8 * (1 - k)}px var(--haze)` : ''; b3.style.borderColor = k > 0.5 ? 'var(--ink)' : '';
        eq.q('#c3b').style.opacity = seg(p, .55, .7);
        cancel.show(seg(p, .5, .65));
        lamFocus(1 - seg(p, .55, .75));                         /* the other two terms come back once Λ_R is sharp */
        where.show(seg(p, .6, .8));
        lam1.set(...LAM1); lam1.show(seg(p, .85, 1));
        const bo = 1 - seg(p, .45, .62); TERMS.b.show(bo); BOREL.show(bo); BOREL.dimPoint(0.45 * bo); BOREL.label.show(bo); BOREL.cLabel.show(bo); BOREL.contourOp(0.9 * bo);   /* the inset has done its job once Λ_R is sharp */
      } },
      { d: 15, f(p) { v0.show(seg(p, 0, .4)); } },
      { d: 24, f(p) { mn.show(seg(p, 0, .4)); nw.show(seg(p, .5, .9)); } },
    ],
  });
  window.__S05 = { lam1, eq, LAM1 };
})();
