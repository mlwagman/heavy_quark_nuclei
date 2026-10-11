'use strict';
/* S04 The renormalon problem · S05 Minimal renormalon subtraction. The bottom of the World: the series lives here,
   the Borel plane is the floor. */

/* world offset (x,y) of an element inside a board, so a chip can fly to a term */
function boardPoint(b, sel) {
  const e = b.q(sel); let x = e.offsetWidth / 2, y = e.offsetHeight / 2, n = e;
  while (n && n !== b.el) { x += n.offsetLeft; y += n.offsetTop; n = n.offsetParent; }
  const s = b.w / b.px; return W.V3(b.anchor.x + (x - b.px / 2) * s, b.anchor.y - (y - b.h / 2) * s, b.anchor.z + 0.5);
}
/* a World point on board b, moved to where the board draws it: a board is sized by its anchor's distance from the camera, not by
   its depth, so off the camera's axis it is a few percent smaller than its World points say (11 px at the Λ_R box of S05) */
const _dd = W.V3(0, 0, 0), _da = W.V3(0, 0, 0);
function asDrawn(b, p) { const c = W.cam, De = c.pos.distanceTo(b.anchor), Dz = _da.subVectors(b.anchor, c.pos).dot(_dd.subVectors(c.look, c.pos).normalize()); return p.sub(b.anchor).multiplyScalar(Dz / De).add(b.anchor); }
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
      <span class="ti muted" style="font-size:22px;flex-basis:100%;margin-top:6px">𝒱${sub('R')}(${I('r')}) = ${H.VR}(${I('r')}) + <span class="box haze" id="lr0" style="padding:2px 8px">${H.LR}</span> <span class="tiny">(Eq. A.1)</span></span></div>
    <h2 class="t rc" style="margin-top:14px">Factorial growth</h2>
    <div class="m t" style="font-size:26px">${I('v')}${sub(I('l'))} ~ ${I('V')}${sub('0')} (2${I('β')}${sub('0')})${sup(I('l'))} ${frac('Γ(' + I('l') + '+1+' + I('b') + ')', 'Γ(1+' + I('b') + ')')}</div>
    <div class="small t" style="margin-top:6px">the series is <b>asymptotic</b>, not convergent</div>` });
  /* V(r) and F(r): the constant-width ambiguity band (S04-P4), the force free of it (Eq. A.3) */
  const vp = BOARDS.board({ scene: SC, cls: 'panel', px: 700, w: U(700, DIST), at: P(-32, -92, 0), a: 0, html: `<h2 id="vt">${H.VR}(${I('r')}) from the static pair</h2><svg viewBox="0 0 660 250" width="652" height="247" id="vs"></svg>
    <div class="t" id="orig"><div class="big" style="font-size:23px">Infrared sensitivity (momenta ~ ${H.LQCD}) ⇒ leading ambiguity ${I('δV')} ~ ${H.LQCD}: a constant, same at every ${I('r')}</div><div class="small" style="margin-top:4px">singlet: the same renormalon as the pole mass</div></div>` });
  const vs = vp.q('#vs');
  const HW = 23;                                              /* half the band: the band is as wide as the smallest term is high on the |term| plot (the same δV) */
  const SV = 96, XR = r => 60 + (r - 0.2) / 3.0 * 580, YV = v => 56 - v * SV;   /* V = 0 at y 56, and the scale keeps the upper copy of the band under it: the curve is −1/r, steep at short distance, flattening under the zero line (round 21: from r = 0.5 with no zero line it did not read as 1/r) */
  S('path', { d: 'M60 240 H640 M60 240 V20', class: 'ax' }, vs);
  const xt = S('text', { x: 636, y: 236, 'text-anchor': 'end', class: 'mt it' }, vs); xt.textContent = 'r';
  const V = r => -1 / r, dV = r => 1 / (r * r);
  const RR = []; for (let r = 0.55; r <= 3.2; r += 0.02) RR.push(r);
  S('line', { id: 'vzero', x1: 60, y1: YV(0), x2: 640, y2: YV(0), stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': '3 5' }, vs);
  S('text', { x: 50, y: YV(0) + 6, 'text-anchor': 'end', class: 'mt' }, vs).textContent = '0';
  S('rect', { x: 60, y: 0, width: 600, height: 239 }, S('clipPath', { id: 'w04clip' }, S('defs', {}, vs)));   /* the band and the lower copy stop at the r axis */
  const dC = dy => RR.map((r, i) => `${i ? 'L' : 'M'}${XR(r).toFixed(1)} ${(YV(V(r)) + dy).toFixed(1)}`).join(' ');
  const blur = S('filter', { id: 'w04blur' }, S('defs', {}, vs)); S('feGaussianBlur', { stdDeviation: 5 }, blur);
  const band = S('path', { id: 'vband', d: dC(-HW) + ' ' + RR.slice().reverse().map(r => `L${XR(r).toFixed(1)} ${(YV(V(r)) + HW).toFixed(1)}`).join(' ') + 'Z', fill: 'var(--hazeband)', filter: 'url(#w04blur)', opacity: 0 }, S('g', { 'clip-path': 'url(#w04clip)' }, vs));   /* the haze colour of S05: white was invisible on the light theme */
  /* the ambiguity is a slide: the curve as far up and as far down as it may sit */
  const ghosts = [-HW, HW].map((dy, i) => S('path', { id: 'vg' + (i + 1), d: dC(dy), fill: 'none', stroke: 'var(--rose)', 'stroke-width': 1.3, 'stroke-dasharray': '6 5', 'clip-path': 'url(#w04clip)', opacity: 0 }, vs));
  const curve = S('path', { id: 'vcurve', d: dC(0), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2.4 }, vs);
  const slide = dy => curve.setAttribute('transform', `translate(0 ${dy.toFixed(2)})`);
  /* the width of the band, at two places: the same at every r */
  const RA = 1.3, RB = 2.7, arrow = (id, r) => { const x = +XR(r).toFixed(1), a = +(YV(V(r)) - HW).toFixed(1), b = +(YV(V(r)) + HW).toFixed(1);
    return S('path', { id, d: `M${x} ${a} V${b} M${x - 4} ${a + 6} L${x} ${a} L${x + 4} ${a + 6} M${x - 4} ${b - 6} L${x} ${b} L${x + 4} ${b - 6}`, fill: 'none', stroke: 'var(--rose)', 'stroke-width': 2, opacity: 0 }, vs); };
  const a1 = arrow('va1', RA), a2 = arrow('va2', RB);
  S('text', { x: 48, y: 34, 'text-anchor': 'end', 'font-size': 24, 'font-family': 'var(--serif)' }, vs).textContent = '𝒱';   /* the y axis: the static potential of Eq. A.1, in its script letter */
  const a1T = S('text', { x: XR(RA), y: +(YV(V(RA)) - HW - 7).toFixed(1), 'text-anchor': 'middle', class: 'it', fill: 'var(--rose)', opacity: 0 }, vs); a1T.textContent = 'δV';   /* names the carried mark where it lands */
  TICKS.push(() => a1T.setAttribute('opacity', a1.getAttribute('opacity')));
  /* the slope at one r, on the curve and on both copies: the force does not see the constant */
  const R0 = 0.75, tans = [-HW, 0, HW].map(dy => { const x = XR(R0), y = YV(V(R0)) + dy, m = -SV * dV(R0) * 3 / 580, L = 38;
    return S('line', { class: 'vtan', x1: x - L, y1: +(y - m * L).toFixed(2), x2: x + L, y2: +(y + m * L).toFixed(2), stroke: 'var(--teal)', 'stroke-width': 3, 'stroke-linecap': 'round', opacity: 0 }, vs); });
  const cl = S('text', { x: 330, y: 222, 'text-anchor': 'middle', opacity: 0 }, vs); cl.innerHTML = `perturbative sum <tspan class="it">V</tspan><tspan baseline-shift="sub" font-size="12">R</tspan>(<tspan class="it">r</tspan>) (Eq. A.1)`;
  const fl = S('text', { id: 'vfl', x: 636, y: 18, 'text-anchor': 'end', opacity: 0, fill: 'var(--ink2)' }, vs);   /* from the right end: started at mid-plot it ran out of the panel */ fl.innerHTML = `<tspan class="it">F</tspan>(<tspan class="it">r</tspan>) = <tspan class="it">dV</tspan>/<tspan class="it">dr</tspan> (Eq. A.3): no constant (leading-renormalon) ambiguity`;
  const fl2 = S('text', { id: 'vfl2', x: 636, y: 39, 'text-anchor': 'end', opacity: 0, class: 'it mt' }, vs); fl2.textContent = 'taking a derivative removes the constant';
  const bl = S('text', { id: 'vbl', x: 440, y: 190, 'text-anchor': 'middle', opacity: 0, class: 'it', fill: 'var(--rose)' }, vs);   /* under the curve and its haze, not across them */ bl.innerHTML = `same width at every <tspan class="it">r</tspan>: <tspan class="it">δV</tspan> ~ Λ<tspan baseline-shift="sub" font-size="12">QCD</tspan>`;
  /* δV on its way from the |term| plot to this one: a marker as long as the smallest term is high there and as the band is wide here */
  const DVH = 2 * HW * 652 / 660;
  const dvB = BOARDS.board({ px: 16, w: U(16, DIST), a: 0, z: 300, html: `<svg class="dvfly" viewBox="0 0 16 ${DVH}" width="16" height="${DVH}"><path d="M8 0 V${DVH} M4 6 L8 0 L12 6 M4 ${DVH - 6} L8 ${DVH} L12 ${DVH - 6}" fill="none" stroke="var(--rose)" stroke-width="2"/></svg>` });
  let dvU = -1;
  TICKS.push(() => { if (dvU < 0 || dvB.a <= 0.003) return; dvB.lerpTo(asDrawn(TERMS.b, TERMS.markAt()), asDrawn(vp, svgPoint(vp, vs, 660, XR(RA), YV(V(RA)))), dvU); });
  vp.measure();
  const bands = BOARDS.board({ scene: SC, cls: 'panel', px: 700, w: U(700, DIST), at: P(20, -104, 0), a: 0, html: `<h2>Bands at successive orders: fixed order</h2><svg viewBox="0 0 660 190" width="652" height="188" id="bs"></svg>
    <div class="tiny" style="margin-top:6px">binding energy at fixed ${H.mQ} (Figs. 1–2 left, schematic)</div>` });
  const bs = bands.q('#bs');
  S('path', { d: 'M50 160 H640 M50 160 V14', class: 'ax' }, bs);
  const byl = S('text', { x: 58, y: 24, class: 'mt' }, bs); byl.innerHTML = `|Δ<tspan class="it">E</tspan>| (schematic)`;
  const BC = { LO: 'var(--lo)', NLO: 'var(--nlo)', NNLO: 'var(--nnlo)' };
  /* the x axis is the order: one range of |ΔE| per order, side by side. LO sits apart; NLO and NNLO overlap (S04-P7, S04-F7).
     Until round 19 the three were boxes staggered across an x axis that meant nothing */
  const brects = [['LO', 150, 123, 12],   /* narrow: the paper's LO band collapses at a stationary point (Sec. III A), which is not a small uncertainty */
     ['NLO', 300, 58, 44], ['NNLO', 450, 40, 40]].map(([k, xc, y, h]) => {
    const r = S('rect', { x: xc - 26, y: y + h, width: 52, height: 0, fill: hatchDef(bs, 'w04h' + k, BC[k]), stroke: BC[k], 'stroke-width': 1.5 }, bs);
    const t = S('text', { x: xc, y: 181, 'text-anchor': 'middle', fill: BC[k], opacity: 0 }, bs); t.textContent = k;
    return { r, t, y, h }; });
  const gapT = S('text', { x: 196, y: 137, 'text-anchor': 'start', class: 'it', opacity: 0 }, bs); gapT.textContent = 'LO apart: no meaningful uncertainty';
  bands.measure();
  const sparts = ser.qa('.ti');
  const GB = VW(20, -87.5, 0);
  /* the symptom in words (round 24d: the scale-band board appears once, in S10) */
  const sym = BOARDS.board({ scene: SC, cls: 'panel', px: 760, w: U(760, DIST), a: 0, html: `<h2 class="rc">Symptom</h2><div class="t big" style="font-size:22px">Varying ${I('μ')} by 2× moves ${H.Obbb} by up to <span class="num rc">~700 MeV</span> at NNLO <span class="small">(Tab. III)</span>; in binding energy at fixed ${H.mQ}, LO does not even overlap NLO/NNLO <span class="small">(Figs. 1–2)</span></div>` });
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
        band.setAttribute('opacity', 0); fl.setAttribute('opacity', 0); fl2.setAttribute('opacity', 0); bl.setAttribute('opacity', 0); rv(vp.q('#orig'), 0);
        ser.show(seg(p, .35, .5)); sparts.forEach((e, i) => rv(e, seg(p, .45 + i * .1, .6 + i * .1)));
        BOARDS.T(ser, '.t', seg(p, .8, 1));
        TERMS.b.set(...P(20, -70, 0)); TERMS.b.w = U(760, DIST); TERMS.b.show(0);
      } },
      { d: 18, f(p) { TERMS.b.show(seg(p, 0, .15)); TERMS.b.touch(); TERMS.reveal(p); } },
      /* b3: the symptom in words (round 24d); then the fixed-order band stack (the camera eases back so the stack's frame is whole) */
      { d: 13, cam: [KS([-4, -84, 84], [-4, -84, 0])], f(p) {
        sym.setV(GB); sym.show(seg(p, 0, .15)); rv(sym.q('.t'), seg(p, .1, .35));
        bands.show(seg(p, .45, .6)); brects.forEach((b, i) => { const e = eo(seg(p, .5 + i * .12, .7 + i * .12)); b.r.setAttribute('y', b.y + b.h * (1 - e)); b.r.setAttribute('height', b.h * e); b.t.setAttribute('opacity', seg(p, .55 + i * .12, .7 + i * .12)); });
        gapT.setAttribute('opacity', seg(p, .88, 1));
      } },
      /* b4, four stops. b4a, the link: the size of the smallest term (a mark on the |term| plot) is carried to the V(r) plot, where it
         lands across the curve: the same δV. Both plots stay in the frame. Timed on the narration (slow 4.2, 12.6 s): the mark at
         "its smallest term" (6 s), the flight at "of order lambda QCD" (9.4 s), the landing at "it is this much" (11.9 s) */
      { d: 6, slow: 4.2, f(p) {
        vp.touch(); TERMS.b.touch(); vp.q('#vt').innerHTML = p > 0.05 ? 'Origin: infrared' : `${H.VR}(${I('r')}) from the static pair`; rv(vp.q('#orig'), seg(p, .08, .21));
        TERMS.mark(seg(p, .476, .556));
        dvU = eio(seg(p, .746, .944)); dvB.show(seg(p, .714, .746) * (1 - seg(p, .96, .99))); a1.setAttribute('opacity', seg(p, .937, .968));
        slide(0); ghosts.forEach(g => g.setAttribute('opacity', 0)); band.setAttribute('opacity', 0); a2.setAttribute('opacity', 0); bl.setAttribute('opacity', 0);
      } },
      /* b4b, the ambiguity is a slide: the camera comes to the plot; the whole curve goes up by δV and down by δV, leaving a dashed copy
         at each end and the band between them; a second arrow, far out in r, is as long as the first: the same at every r.
         slow 2.88 (7.2 s): the slide at "slide up or down" (2.8 s), the second arrow at "the same at every distance" (5.6 s) */
      { d: 5, slow: 2.88, cam: [KS([-31, -92, 58], [-31, -93, 0])], win: [0, .3], f(p) {
        vp.touch(); const q = seg(p, .39, .75); slide(-HW * Math.sin(2 * Math.PI * q));
        ghosts[0].setAttribute('opacity', seg(q, .2, .3)); ghosts[1].setAttribute('opacity', seg(q, .7, .8)); band.setAttribute('opacity', seg(p, .69, .78));
        a2.setAttribute('opacity', seg(p, .78, .89)); bl.setAttribute('opacity', seg(p, .87, .97));
        tans.forEach(t => t.setAttribute('opacity', 0)); fl.setAttribute('opacity', 0); fl2.setAttribute('opacity', 0);
      } },
      /* b4c, the slope is safe: one tangent mark on the upper copy, on the curve and on the lower copy, at the same r: they are parallel.
         The force does not see the constant. slow 2.4 (6 s): the marks at "its slope" (0.8 s), the force label at "the force" (2.6 s) */
      { d: 5, slow: 2.4, f(p) {
        vp.touch(); tans.forEach((t, i) => t.setAttribute('opacity', seg(p, .13 + i * .1, .25 + i * .1)));
        fl.setAttribute('opacity', seg(p, .43, .55)); fl2.setAttribute('opacity', seg(p, .63, .78));
      } },
      /* b4d: in the Borel plane the same ambiguity is a branch point: the inset appears under the plot */
      { d: 4, slow: 2, cam: [KS([-24, -100, 82], [-24, -101, 0])], win: [0, .5], f(p) {
        vp.touch(); borelAt(0); BOREL.show(seg(p, .25, .6)); BOREL.label.show(seg(p, .6, .85)); BOREL.cLabel.show(0); BOREL.contourOp(0);
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
      <span class="ti box" id="b2" style="display:inline-grid;overflow:hidden;align-items:center;justify-items:center">
        <span id="b2s" style="grid-area:1/1">${sum(I('l') + '=0', '∞')} ${I('V')}${sub(I('l'))} ${H.as}${sup(I('l') + '+1')}(${H.mup}(${I('r')}))</span>
        <span id="b2j" style="grid-area:1/1;opacity:0">${frac(I('V') + sub('0'), '2' + I('β') + sub('0'))} 𝒥(<span id="argb">${I('b')}</span>, <span id="argy">${frac('1', '2' + I('β') + sub('0') + H.as + '(' + H.mup + '(' + I('r') + '))')}</span>)</span>
      </span>
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
  const hazeB = BOARDS.board({ px: 200, w: U(200, DIST), a: 0, z: 300, html: `<div class="hazeband" style="position:relative;width:200px;height:26px"></div>` });
  const cancel = { show: a => rv(eq.q('#cancel'), a), set() { } };
  const where = BOARDS.board({ scene: SC, cls: 'panel', px: 800, w: U(800, DIST), at: P(76, -98, 0), a: 0, html: `<h2 class="rc">Where does the renormalon go?</h2>
    <div class="big" style="font-size:24px">${H.LR} comes out <b>free of the leading-renormalon ambiguity</b>. It is absorbed into the quark mass, tuned so that 2${H.mQ} + ${I('E')} <b class="rc">matches the spin-averaged 1S states</b>. For baryons and tetraquarks it is neglected, because they use the <b>same tuned masses</b>, which already include it.</div>` });
  const lam1 = BOARDS.board({ px: 300, w: U(300, DIST), a: 0, html: `<span class="chip pin">${H.LR} absorbed into ${H.mQ}</span>` }); lam1.el.style.textAlign = 'center';
  const LAM1 = P(49, -97.4, 0);                                    /* where the chip waits for S08: at the panel's edge, beside "absorbed into the quark mass" */
  const v0 = BOARDS.board({ scene: SC, cls: 'panel', px: 520, w: U(520, DIST), at: P(114, -98, 0), a: 0, html: `<h2>Normalization has converged</h2>
    <div class="m num" style="font-size:24px">${I('V')}${sub('0')} = 0.771 <span class="small">(NNLO)</span> → 0.772 <span class="small">(N³LO)</span> <span class="tc">✓</span> <span class="small">(singlet, ${I('n')}${sub('f')} = 4; §II B)</span></div>` });   /* §II B prose, S05-P5 */
  const mn = BOARDS.board({ scene: SC, cls: 'panel', px: 600, w: U(600, DIST), at: P(70, -111, 0), a: 0, html: `<h2>“Minimal” matters</h2><div class="big" style="font-size:23px">Same ${ov('MS')} inputs: the series is only reorganized: the renormalon tail is isolated and resummed (Komijani ’17; Kronfeld ’23).</div>` });
  const nw = BOARDS.board({ scene: SC, cls: 'panel', px: 600, w: U(600, DIST), at: P(104, -111, 0), a: 0, html: `<h2 class="rc">What is new</h2><div class="big" style="font-size:23px">Used for quark masses: all flavours except top (Fermilab/MILC/TUMQCD). <span class="rc">First use in a pNRQCD bound-state problem.</span></div>` });
  const eparts = eq.qa('.ti'), b3 = eq.q('#b3'), b1e = eq.q('#b1'), b2e = eq.q('#b2');
  /* the dots that leave the |term| plot: the remainder (teal) into the first box, the renormalon column (amber) into the second box, the sum of the V_l */
  const flyR = BOARDS.board({ px: 120, w: U(120, DIST), a: 0, z: 300, html: `<div class="tc" style="font-size:30px;letter-spacing:3px;text-align:center">●●●</div>` });
  const flyV = BOARDS.board({ px: 260, w: U(260, DIST), a: 0, z: 300, html: `<div class="ac" style="font-size:30px;letter-spacing:3px;text-align:center">●●●●●●●</div>` });
  let PB1 = null, PB2 = null, S1 = null, S2 = null;   /* targets inside the equation board (measured once it is laid out) */
  /* S1, S2: where the dots stop before they go in, outside the panel (on it they were written on the equation): above the first box and above the second */
  const targets = () => { if (!PB1) { PB1 = boardPoint(eq, '#b1'); PB2 = boardPoint(eq, '#b2');
    const hw = eq.w / 2, hh = eq.h * eq.w / eq.px / 2; S1 = W.V3(PB1.x, eq.anchor.y + hh + 1.6, PB1.z); S2 = W.V3(PB2.x, eq.anchor.y + hh + 1.6, PB2.z); } };
  /* Λ_R singled out (S05 b2d, undone in b3): a rose ring on its box; the other two boxes and their captions at .45, still read */
  const lamFocus = k => { [b1e, b2e, eq.q('#c1'), eq.q('#c2'), eq.q('#jl')].forEach(e => { e.style.opacity = lerp(1, .45, k); });
    b3.style.outline = k > 0 ? `3px solid color-mix(in srgb, var(--rose) ${(100 * k).toFixed(1)}%, transparent)` : ''; b3.style.outlineOffset = '5px'; };   /* it fades in: a width under 1 px is drawn at 1 px, so a growing ring popped */
  /* the second box: k = 0 the infinite sum of the V_l, k = 1 its Borel sum. Both sit in one grid cell, so the box keeps its size */
  const b2s = eq.q('#b2s'), b2j = eq.q('#b2j');
  const borelSum = k => { const h = b2s.parentElement.clientHeight, u = eio(k);   /* the sum rolls up out of its box as the closed form rolls in from below (the box clips them): never one on the other, the box never empty */
    b2s.style.opacity = k < 1 ? 1 : 0; b2s.style.transform = `translateY(${(-h * u).toFixed(2)}px)`; b2j.style.opacity = k > 0 ? 1 : 0; b2j.style.transform = `translateY(${(h * (1 - u)).toFixed(2)}px)`; };
  const CUTAT = 1;                                                /* where on the u axis the haze rises: on the cut, past the branch point at ½ */
  /* the haze's way from the cut into the Λ_R box (hazeU: 0 on the cut … 1 in the box; -1: it does not travel). A follower, because
     the end of the way is where the box is drawn (asDrawn), and the camera moves in this beat */
  let hazeU = -1;
  TICKS.push(() => { if (hazeU < 0 || hazeB.a <= 0.003) return; hazeB.lerpTo(BOREL.at(CUTAT), asDrawn(eq, boardPoint(eq, '#b3')), hazeU); });
  scenes.push({
    id: 'S05', title: 'Minimal renormalon subtraction', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: headline; the |term| dots split into V_l and the remainder (l ≤ 2 only) */
      { d: 12, cam: [KS([73, -83, 90], [73, -84, 0])], f(p) {
        head.show(seg(p, 0, .3)); BOARDS.T(head, '.ti', seg(p, .2, .6));
        const u = eio(seg(p, 0, .5)); TERMS.b.lerpTo(TW(3, 20, -70, 0), V(42, -79, 0), u); TERMS.b.w = U(760, DIST) * lerp(1, 1.18, u); TERMS.b.show(1); TERMS.b.touch();   /* the dots grow into the main picture */
        TERMS.fold(0); TERMS.foldR(0); TERMS.split(seg(p, .3, 1));   /* split last: fold(0) alone would show every amber dot, split or not (they leaked into S04) */
        borelAt(u); BOREL.dimPoint(lerp(1, 0.45, seg(p, 0, .3)));
      } },
      /* b2, five stops. b2a: the three-box equation is written; the remainder dots (teal) leave the |term| plot and stop at the first box */
      { d: 10, cam: [KS([73, -82, 88], [73, -83, 0])], f(p) {
        eq.show(seg(p, 0, .2)); eparts.forEach((e, i) => rv(e, seg(p, .1 + i * .06, .3 + i * .06)));
        targets(); TERMS.b.show(1); TERMS.b.touch(); b3.classList.add('haze');
        TERMS.foldR(seg(p, .6, .72)); flyR.lerpTo(TERMS.at(TERMS.X(1), 190), S1, eio(seg(p, .66, .95))); flyR.show(seg(p, .64, .72));
      } },
      /* b2b: the teal dots go into the box (the series with the renormalon part subtracted); the renormalon column (amber, every
         order) gathers and stops at the second box: their sum over every order */
      { d: 7, f(p) {
        targets(); eq.touch(); TERMS.b.show(1 - seg(p, .8, 1));   /* the graph leaves once its last dots have left it */
        flyR.lerpTo(S1, PB1, eio(seg(p, 0, .2))); flyR.show(1 - seg(p, .05, .2)); b1e.style.borderColor = p > .1 ? 'var(--teal)' : '';
        rv(eq.q('#c1'), seg(p, .15, .4));
        TERMS.fold(seg(p, .35, .55)); flyV.lerpTo(TERMS.at(700, 110), S2, eio(seg(p, .5, .95))); flyV.show(seg(p, .45, .55));
      } },
      /* b2c, a stop on the infinite sum: the amber dots (every order) go into it */
      { d: 3, slow: 1.5, cam: [KS([86, -85, 86], [86, -86, 0])], win: [0, 1], f(p) {   /* the graph has left: the frame recentres on the equation and the inset */
        targets(); eq.touch();
        flyV.lerpTo(S2, PB2, eio(seg(p, 0, .4))); flyV.show(1 - seg(p, .1, .4)); b2e.style.borderColor = p > .2 ? 'var(--amber)' : '';
      } },
      /* b2c′: the sum is Borel-summed. On the inset the contour lights up to the branch point, then the sum turns into its closed
         form. slow 3.5 (7 s), timed on the sentence: "along this path up to the branch point" (2.3 s), "it turns into this
         closed form" (4.5 s) */
      { d: 4, slow: 3.5, f(p) {
        targets(); eq.touch();
        BOREL.show(1); BOREL.label.touch(); BOREL.dimPoint(lerp(0.45, 1, seg(p, .15, .33)));   /* the inset is the stage now: the branch point and the cut come back to full strength */
        BOREL.contourOp(seg(p, .33, .55) * 0.9); BOREL.cLabel.show(seg(p, .5, .62)); borelSum(seg(p, .64, .86));
        rv(eq.q('#c2'), seg(p, .8, .92)); rv(eq.q('#jl'), seg(p, .9, 1));
      } },
      /* b2d: the third term on its own. A rose ring on the Λ_R box while the other two boxes and their captions step back; its
         caption; then, on the Borel inset, what the sum cannot reach (the cut past the branch point) rises as a rose haze: the
         thing Λ_R will take in. slow 3.6 (10.8 s): the haze rises with its sentence, 9.3 s in */
      { d: 6, slow: 3.6, f(p) {
        targets(); eq.touch(); b3.classList.add('haze');
        lamFocus(seg(p, 0, .1)); rv(eq.q('#c3'), seg(p, .07, .17));
        BOREL.show(1); BOREL.dimPoint(1); BOREL.label.touch(); BOREL.cLabel.touch();
        hazeU = -1; hazeB.setV(BOREL.at(CUTAT)); hazeB.w = U(200, DIST); hazeB.show(seg(p, .85, .96)); eq.q('#c3b').style.opacity = 0; cancel.show(0);
      } },
      /* b3: the haze band leaves the cut, narrows, and dissolves into the Λ_R box as it arrives; the box's own haze swells with it,
         then goes for good: Λ_R is sharp (an inline 'none': with '' the class's shadow came back). Then the chip parks for S08 */
      { d: 18, cam: [KS([73, -87, 90], [73, -89, 0])], f(p) {
        targets(); eq.touch();
        const u = eio(seg(p, 0, .4)); hazeU = u; hazeB.show(1 - seg(p, .255, .305)); hazeB.w = U(200, DIST) * lerp(1, .25, seg(u, .3, .9));   /* it narrows to the box it enters, and dissolves at the box's edge: across the symbol it read as a strike-through */
        const g = 1 + 0.8 * seg(p, .27, .36), k = seg(p, .4, .6); b3.style.boxShadow = k < 1 ? `0 0 ${18 * g * (1 - k)}px ${8 * g * (1 - k)}px var(--haze)` : 'none'; b3.style.borderColor = k > 0 ? `color-mix(in srgb, var(--ink) ${(100 * k).toFixed(1)}%, var(--box-border))` : '';
        eq.q('#c3b').style.opacity = seg(p, .55, .7);
        cancel.show(seg(p, .5, .65));
        lamFocus(1 - seg(p, .55, .75));                         /* the other two terms come back once Λ_R is sharp */
        where.show(seg(p, .6, .8));
        lam1.set(...LAM1); lam1.show(seg(p, .85, 1));
        const bo = 1 - seg(p, .45, .62); BOREL.show(bo); BOREL.dimPoint(bo); BOREL.label.show(bo); BOREL.cLabel.show(bo); BOREL.contourOp(0.9 * bo);   /* the inset has done its job once Λ_R is sharp */
      } },
      { d: 15, f(p) { v0.show(seg(p, 0, .4)); } },
      { d: 24, f(p) { mn.show(seg(p, 0, .4)); nw.show(seg(p, .5, .9)); } },
    ],
  });
  window.__S05 = { lam1, eq, LAM1 };
})();
