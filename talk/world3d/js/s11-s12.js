'use strict';
/* S11 Results IV: the ratio and the Detmold bound · S12 Results V: all-heavy tetraquarks. */

/* the dashed line that carries from Fig. 4 (Detmold bound) to Fig. 5 (R = 0 threshold), relabelled during the morph (#45) */
const DASH = BOARDS.board({ px: 520, w: 30, a: 0, html: `<svg viewBox="0 0 520 40" width="520" height="40"><line x1="0" y1="20" x2="520" y2="20" stroke="var(--ink)" stroke-width="2" stroke-dasharray="10 7"/></svg><div class="lbl" id="dl" style="position:absolute;left:-240px;right:-240px;text-align:center"></div>` });
/* the line is len px long about the board's centre (the board keeps its scale, so the words keep their size); the words sit drop px under it */
DASH.lay = (len, drop) => { const l = DASH.q('line'); l.setAttribute('x1', 260 - len / 2); l.setAttribute('x2', 260 + len / 2); DASH.q('#dl').style.top = (20 + drop) + 'px'; };

/* ======================= S11 ======================= */
(function () {
  const SC = 10, DIST = 118, { P, V, KS } = FR(SC);
  const PF = figBoard(SC, 'Fixed order', 'assets/s11-fixed-order.png', '', P(-100, 232, 0), 560, DIST);   /* Fig. 4 left */
  const PM = figBoard(SC, 'MRS', 'assets/s11-mrs.png', '', P(-52, 232, 0), 560, DIST);                    /* Fig. 4 right */
  const k1 = BOARDS.board({ scene: SC, cls: 'panel', px: 400, w: U(400, DIST), at: P(-122, 200, 0), a: 0, html: `<h2>Why a ratio?</h2><div class="big" style="font-size:22px">The leading renormalon partially cancels in M${sub('QQQ')}/M${sub('Q' + ov('Q'))}. Dashed line: Detmold bound, M${sub('QQQ')} ≥ (3/2) M${sub('Q' + ov('Q'))}.</div>` });
  const k2 = BOARDS.board({ scene: SC, cls: 'panel', px: 400, w: U(400, DIST), at: P(-90, 200, 0), a: 0, html: `<h2 class="tc">MRS</h2><div class="big" style="font-size:22px">All three orders collapse into <span class="num">1.50–1.51</span>: just above the bound, essentially saturating it.</div>` });
  const k3 = BOARDS.board({ scene: SC, cls: 'panel', px: 400, w: U(400, DIST), at: P(-58, 200, 0), a: 0, html: `<h2 class="lat">Lattice</h2><div class="big" style="font-size:22px">Red points 1.52–1.56 (Tab. I + III, lattice [27]), drifting toward us as ${H.mQ} grows: the same 1/${H.mQ} gap, seen in the ratio.</div>` });
  const k4 = BOARDS.board({ scene: SC, cls: 'panel', px: 400, w: U(400, DIST), at: P(-26, 200, 0), a: 0, html: `<h2>Fixed-order “agreement” is bought by enormous bands</h2><div class="banner" style="font-size:22px">Better agreement with a bigger error bar is not better physics.</div>` });
  BOARDS.topAlign(k1, k2, k3, k4);
  PF.keep = PM.keep = DASH.keep = true;                         /* the two figures are the scene: pale or lit, never dimmed and shrunk (the dashed line laid on one must stay on it) */
  const UNLIT = k => `opacity(${0.55 + 0.45 * k})`;             /* a figure waiting for its beat is pale, on either theme (darkened, it was a grey slab on the light one) */
  const DET = V(-50.95, 225.8, 1);                           /* on the figure's own 1.50 line, as long as its frame; the words go under the panel, not over the plot */
  window.__S11 = { DET };
  scenes.push({
    id: 'S11', title: 'The ratio of baryon to meson mass and the Detmold bound', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: the two Fig. 4 panels, waiting pale; the Detmold line on them */
      { d: 14, win: [0, .7], cam: [KS([-76, 216, DIST], [-76, 216, 0])], f(p) {
        GAUGE.b.show(1 - seg(p, 0, .2)); NLINE.b.show(0);
        PF.show(seg(p, .35, .55)); PM.show(seg(p, .35, .55)); PF.el.style.filter = UNLIT(0); PM.el.style.filter = UNLIT(0);
        DASH.setV(DET); DASH.w = U(520, DIST); DASH.lay(378, 128); DASH.q('#dl').innerHTML = `Detmold bound M${sub('QQQ')} ≥ (3/2) M${sub('Q' + ov('Q'))} (dashed)`; DASH.show(seg(p, .6, .8));
        k1.show(seg(p, .5, .75));
      } },
      { d: 20, f(p) { PM.touch(); PM.el.style.filter = UNLIT(seg(p, 0, .3)); k2.show(seg(p, .2, .5)); } },
      { d: 20, f(p) { k3.show(seg(p, 0, .4)); } },
      { d: 19, f(p) { PF.touch(); PF.el.style.filter = UNLIT(seg(p, 0, .3)); k4.show(seg(p, .2, .5)); } },
    ],
  });
})();

/* ======================= S12 ======================= */
(function () {
  const SC = 11, DIST = 120, { P, V, KS } = FR(SC);
  const q = BOARDS.board({ scene: SC, cls: 'panel', px: 520, w: U(520, DIST), at: P(12, 232, 0), a: 0, html: `<h2>A fourth body arrives</h2>
    <div class="m" style="font-size:22px">R = (Δ${I('E')}${sub('4Q')} − 2Δ${I('E')}${sub('Q' + ov('Q′'))}) / 2Δ${I('E')}${sub('Q' + ov('Q′'))}</div>
    <div class="small t" style="margin-top:6px">R &gt; 0 ⇔ bound below the two-quarkonium threshold</div>
    <div class="big t" style="margin-top:6px">binding requires a mass <b>hierarchy</b></div>` });
  const fig = BOARDS.board({ scene: SC, cls: 'panel', px: 1000, w: U(1000, DIST), at: P(82, 233, 0), a: 0, html: `<h2>Binding ratio R vs ${H.mQ}/${I('m')}${sub('Q′')}: R &gt; 0 ⇔ bound below the two-meson threshold · MRS at LO/NLO</h2>
    <div style="background:#fff;border-radius:4px;padding:6px"><img src="assets/s12-figure.png" style="width:100%"></div>
    <div class="small t" style="margin-top:8px">critical <span class="num">${H.mQ}/${I('m')}${sub('Q′')} = 0.085(5)</span> (§V A) at NLO: fixed-order NLO said ≈0.125 (§VI). MRS requires a <b>smaller</b> mass ratio: the binding window <b>tightens</b> by ~30%, with the same verdicts in both schemes.</div>` });   /* Fig. 5; §V A, §VI prose (S12-U1) */
  const sketch = BOARDS.board({ scene: SC, px: 320, w: 24, at: P(68, 79, 0), a: 0,   /* at the right end of the top row, beside the glyphs it is about */
    html: `<span class="lbl">sketch: ${ov('t')}${ov('t')} compact, lighter quarks outside; sizes not computed</span>` });
  /* Tab. V, NLO: B [MeV] with the full range of the scale variation (second parenthesis); drawn as ± half of it about the centre (G5) */
  const BOUND = [['bbtt', H.bbtt, 509, 340, 476, 19], ['bctt', H.bctt, 604, 410, 516, 135], ['cctt', H.cctt, 742, 479, 622, 240]];   /* [id, name, MRS B, MRS range, FO B, FO range] */
  const bound = BOARDS.board({ scene: SC, cls: 'panel', px: 560, w: U(560, 175), at: P(32, 156, 0), a: 0, html: `<h2 class="tc">BOUND · ΔE ≈ <span class="num">509–742 MeV</span> <span class="tiny">(Tab. V, NLO) · top decays</span></h2><div id="rows"></div>
    <div class="tiny" style="margin-top:6px">bars: scale range (full range, Tab. V) · <span class="nlo">hatched = fixed order</span>, <span class="nlo">solid = MRS</span>: MRS is wider; what stays small is the order-by-order shift</div>` });
  const brows = BOUND.map(([id, nm, mb, mr, fb, fr]) => { const r = D('div', 'row t', bound.q('#rows')); r.style.margin = '4px 0';
    D('div', 'm', r, nm).style.cssText = 'width:60px;flex:none';
    const col = D('div', '', r); col.style.cssText = 'flex:1;display:flex;flex-direction:column;gap:3px';
    const f = D('div', 'bar hatch nlo', col); f.style.cssText = `height:10px;width:0`; const m = D('div', 'bar solid nlo', col); m.style.cssText = `height:10px;width:0`;
    D('div', 'small num', r, `${mb}(${mr}) MeV`).style.cssText = 'width:150px;flex:none;text-align:right';
    return { r, f, m, fw: fr / 479 * 230, mw: mr / 479 * 230 }; });
  bound.measure();
  const unb = BOARDS.board({ scene: SC, cls: 'panel', px: 420, w: U(420, 175), at: P(88, 156, 0), a: 0, html: `<h2 class="rc">UNBOUND · ratio ≫ 0.085</h2><div class="big" style="font-size:22px"><span class="chip t">${H.bbbb}</span> <span class="chip t">${H.ccbb}</span><br><span class="t">no binding within our variational space</span><br><span class="small t">mass ratios 1 and 0.324, far above the threshold</span></div>` });   /* Tab. I derived: 1.55489/4.80221 = 0.324 */
  const col = BOARDS.board({ scene: SC, cls: 'panel', px: 560, w: U(560, 175), at: P(142, 156, 0), a: 0, html: `<h2>Colour structure</h2>
    <div class="row"><div class="small m" style="width:56px;flex:none">3⊗${ov('3')}</div><div class="bar solid" id="c1" style="width:0;height:13px;flex:none;color:var(--ink2)"></div></div>
    <div class="row"><div class="small m" style="width:56px;flex:none">6⊗${ov('6')}</div><div class="bar solid" id="c2" style="width:0;height:13px;flex:none;color:var(--dim)"></div></div>
    <div class="tiny">GFMC from 3⊗${ov('3')} vs 6⊗${ov('6')} starting states: slow colour convergence (§V B) · schematic lengths</div>
    <div class="big t" style="font-size:22px;margin-top:6px">3⊗${ov('3')} beats 6⊗${ov('6')} by 50–150 MeV <span class="small">(§V B)</span> ⇒ consistent with a <b>diquark–antidiquark</b> structure (colour convergence under study)</div>` });
  BOARDS.topAlign(bound, unb, col);
  window.__S12 = { sketch };
  scenes.push({
    id: 'S12', title: 'All-heavy tetraquarks: who binds?', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: the dashed Detmold line morphs into the dashed R = 0 threshold, relabelled on the way */
      ...cut({ d: 18, cam: [KS([56, 233, 110], [56, 233, 0])], f(p) {   /* on the two S12 boards, at their own size (the old frame, 150 back, showed them at 0.75 to 0.8); a stop on the R = 0 line, then it goes */
        /* the Detmold line fades where it was; the R = 0 threshold is a different line and appears on its own (no morph) */
        const early = p < .38; DASH.setV(early ? window.__S11.DET : V(82, 232, 1)); DASH.w = U(520, DIST); DASH.lay(early ? 378 : 960, early ? 128 : 22);
        DASH.show(early ? 1 - seg(p, .1, .35) : seg(p, .4, .6) * (1 - seg(p, .72, .9)));
        DASH.q('#dl').innerHTML = early ? `Detmold bound M${sub('QQQ')} ≥ (3/2) M${sub('Q' + ov('Q'))} (dashed)` : `R = 0 threshold (dashed): a different line, a different quantity`;
        q.show(seg(p, .2, .45)); BOARDS.T(q, '.t', seg(p, .45, .68));
      } }, [12.5, 5.5]),
      { d: 22, f(p) { fig.show(seg(p, 0, .3)); rv(fig.q('.t'), seg(p, .6, .9)); } },
      /* b3, in two stages. First close on the shelf: the bound states join the dim top row (t̄t̄ compact); the unbound ones try to
         form beside it and fall apart. Then the camera goes up to the two verdict boards, which give the numbers, and frames them
         alone: one frame over shelf and boards left the boards at the top, the shelf at 0.45 in a corner and the middle empty */
      ...cut({ d: 26, slow: 1.5, camU: p => (eio(seg(p, 0, .18)) + eio(seg(p, .55, .78))) / 2,   /* cut where the camera holds on the shelf */
        cam: [KS([9.7, 77, 116], [9.7, 77, 0]), KS([56, 153, 150], [56, 153, 0])], f(p) {   /* the first frame holds the whole top row, the η_t name at its left end included */
        const u = eio(seg(p, 0, .2)); SHELF.place('s10b', 's12', u); SHELF.travel(9, SC, eio(seg(p, 0, .18)));   /* the shelf comes over from S10 with the camera's first stage */
        scenes[SC - 1].ghost = 1 - seg(p, .5, .65);              /* the S11 boards go before the second frame, which held one of them cut in its corner */
        BOUND.forEach(([id], i) => { const G = SHELF.glyphs[id]; const k = seg(p, .14 + i * .06, .26 + i * .06); G.g.setOpacity(0.55 * k); G.g.setGlow(0.6); G.lab.show(k); G.chip.html(`<span class="chip">top decays</span>`).show(k); });
        sketch.show(seg(p, .2, .3));
        /* unbound: the glyphs try to form beside the row, then split into two mesons and fade (S12-F6) */
        ['bbbb', 'ccbb'].forEach((id, i) => { const G = SHELF.glyphs[id]; G.g.position.set(63 + 12 * i, LAY.topY + 8, 0);
          const form = seg(p, .34, .4), split = eo(seg(p, .42, .49)), fade = seg(p, .5, .56);   /* a stop on the two split pairs with their names; they go as the camera leaves */
          G.g.setOpacity(0.55 * form * (1 - fade)); G.g.userData.split = split; G.lab.show(form * (1 - fade)); });
        bound.show(seg(p, .68, .78)); brows.forEach((r, i) => { const k = eo(seg(p, .74 + i * .04, .88 + i * .04)); rv(r.r, seg(p, .72 + i * .04, .8 + i * .04)); r.f.style.width = (r.fw * k) + 'px'; r.m.style.width = (r.mw * k) + 'px'; });
        unb.show(seg(p, .84, .92)); BOARDS.T(unb, '.t', seg(p, .9, 1));
      } }, [13, 13]),
      { d: 18, cam: [KS([87, 156, 147], [87, 156, 0])], f(p) { col.show(seg(p, .4, .58));   /* the three verdict boards fill the frame; the third comes once the camera holds it whole */
        col.q('#c1').style.width = (240 * eo(seg(p, .5, .7))) + 'px'; col.q('#c2').style.width = (160 * eo(seg(p, .6, .8))) + 'px'; rv(col.q('.t'), seg(p, .8, 1)); } },
    ],
  });
})();
