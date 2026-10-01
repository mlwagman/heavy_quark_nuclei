'use strict';
/* S10 Results III: triply-heavy baryons vs lattice QCD. Numbers: Tab. III (MRS NNLO, lattice), Tab. IV (top baryons). */
(function () {
  const SC = 9, DIST = 112, { P, V, KS } = FR(SC);
  const OM = ['Occc', 'Occb', 'Ocbb', 'Obbb'];
  const MRS = ['4.6186(1)(34)', '7.8511(1)(52)', '11.0701(1)(83)', '14.2311(1)(112)'];                 /* Tab. III, This work MRS, NNLO (rounded as on the Deck) */
  const LAT = ['4.796(8)(18) [27]', '8.005(6)(11) [28]', '11.194(5)(12) [28]', '14.366(7)(9) [31]'];   /* Tab. III, lattice QCD column */
  const GAP = ['~175', '~155', '~125', '~135'];                                                          /* Tab. III derived: 177.5 / 153.9 / 123.9 / 134.9 MeV */
  const PCT = [3.7, 1.9, 1.1, 0.9];                                                                      /* gap ÷ lattice, storyboard S10 b4 */
  /* the two row headers stand under the tetraquark, which has no number: beside the row they name and inside the closest shelf frame */
  const hdr = BOARDS.board({ scene: SC, px: 170, w: U(170, DIST), at: P(44.6, 61.8, 0), a: 0, html: `<span class="lbl">NNLO MRS [GeV]</span>` });
  const hdr2 = BOARDS.board({ scene: SC, px: 170, w: U(170, DIST), at: P(44.6, 57.8, 0), a: 0, html: `<span class="lbl lat">lattice QCD [GeV]</span>` });
  const lchips = OM.map(() => BOARDS.board({ scene: SC, px: 240, w: U(240, DIST) * 0.84, a: 0, cls: 'small', html: '' }));   /* at full size a lattice value with its reference is wider than the glyph spacing */
  lchips.forEach(b => b.el.style.textAlign = 'center');
  /* the lattice chips ride under their glyphs while the shelf is here; once it has moved on they stay, under where the glyphs stood */
  TICKS.push(() => { const here = SHELF.at.distanceTo(SHELF.O(SC)) < 0.5; OM.forEach((id, i) => { const p = here ? SHELF.wp(id) : SHELF.layoutPos('s10b', id).add(SHELF.O(SC)); lchips[i].set(p.x, p.y - 12.2, p.z); }); });
  const nb = BOARDS.board({ scene: SC, cls: 'panel', px: 620, w: U(620, DIST), at: P(114, 88, 0), a: 0, html: `<div class="big">We undershoot the lattice by 125–175 MeV. Fixed order sits closer (80–110 MeV below): the MRS advantage is the band, not the central value.</div>` });   /* §IV B; Tab. III derived */
  const gaps = BOARDS.board({ scene: SC, cls: 'panel', px: 620, w: U(620, DIST), at: P(114, 112, 0), a: 0, html: `<h2>The undershoot is a pattern</h2><svg viewBox="0 0 560 170" width="560" height="170" id="gs"></svg>
    <div class="big" style="font-size:22px">fractional gap ∼ 1/${H.mQ}: not noise; consistent with missing <b>𝒪(1/${H.mQ}) + spin</b> <span class="small">(§VI)</span></div>` });
  const gs = gaps.q('#gs');
  const gl = S('text', { x: 552, y: 20, 'text-anchor': 'end', class: 'mt' }, gs); gl.textContent = 'gap ÷ lattice';
  const gbars = PCT.map((g, i) => { const t0 = S('text', { x: 75 + 130 * i, y: 162, 'text-anchor': 'middle', class: 'mt' }, gs); t0.innerHTML = 'Ω<tspan baseline-shift="sub" font-size="12">' + ['ccc', 'ccb', 'cbb', 'bbb'][i] + '</tspan>';
    const r = S('rect', { x: 49 + 130 * i, y: 140, width: 52, height: 0, fill: 'var(--ink2)', 'fill-opacity': .7 }, gs);
    const t = S('text', { x: 75 + 130 * i, y: 132 - g * 30, 'text-anchor': 'middle', opacity: 0 }, gs);   /* above the bar's final top, not on the bar */ t.textContent = g.toFixed(1) + ' %'; return { r, t, g }; });
  gaps.measure();
  /* the summary of b5 stands over the top row it sums up, in the one free place there (between the S08 drift board and the S09
     benchmark board): the beat can end close on the shelf. Beside the shelf, it forced a frame 162 back and numbers 9 to 11 px high */
  const topN = BOARDS.board({ scene: SC, cls: 'panel', px: 900, w: 38, at: P(9.5, 90.3, 0), a: 0, html: `<div class="big">Top-quark benchmarks up to ${H.Ottt} = <span class="num">514.46(4)(94) GeV</span>: for ${H.Ottt} the band is wide, ±0.94 GeV. <span class="small">cannot form: top decays (Tab. IV)</span></div>` });
  const TOP = [['Otcc', '175.55827(6)(648)'], ['Otcb', '178.76384(5)(995)'], ['Otbb', '181.85756(8)(1444)'], ['Ottc', '346.44124(7)(1273)'], ['Ottb', '349.33347(34)(4681)'], ['Ottt', '514.46(4)(94)']];   /* Tab. IV, NNLO */
  scenes.push({
    id: 'S10', title: 'Triply-heavy baryons against lattice QCD', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    /* the camera frames what each beat is about: the shelf (b1, first half of b2), the boards right of it (b2 to b4),
       the shelf again, closer, with the summary over its top row (b5). One frame wide enough for all of it made every number small. */
    beats: [
      /* b1: a third quark drops into the mesons: the four Ω glyphs light up with their NNLO MRS chips */
      { d: 16, cam: [KS([8.5, 75, 102], [8.5, 75, 0])], f(p) {
        NLINE.b.show(1 - seg(p, 0, .25)); SHELF.travel(8, SC, eio(seg(p, 0, .6)));   /* the shelf moves on with the camera */
        const u = eio(seg(p, 0, .35)); SHELF.place('s09c', 's10a', u);
        OM.forEach((id, i) => { const G = SHELF.glyphs[id]; const k = seg(p, .25 + i * .12, .45 + i * .12);
          if (id !== 'Occc') { G.g.setOpacity(k); G.g.setGlow(1); G.lab.show(k); }
          G.chip.html(`<span class="chip">${MRS[i]}</span>`).show(k); });
        hdr.show(seg(p, .3, .5));
      } },
      /* b2: the faint red lattice wall rises behind the baryons; lattice chips with refs; gap arrows.
         Then the camera travels to the boards: the zoom-pan pulls back over shelf and boards together on the way */
      ...cut({ d: 18, win: [.5, .9], cam: [KS([114, 101, 80], [114, 101, 0])], f(p) {   /* a stop on the shelf with the lattice row, then the boards */
        LATT.show(seg(p, 0, .25)); hdr2.show(seg(p, .1, .25));
        scenes[SC - 1].ghost = 1 - seg(p, .5, .7);              /* the S09 boards hang above the shelf: they leave with it, so no ghost is cut by the closer frames */
        OM.forEach((id, i) => { const k = seg(p, .12 + i * .08, .26 + i * .08); lchips[i].html(`<span class="chip lat">${LAT[i]}</span><br><span class="tiny rc">↕ ${GAP[i]} MeV</span>`).show(k); });
        nb.show(seg(p, .7, .9));
        NLINE.config({ key: 'obbb', title: `${H.Obbb} at NNLO: ours vs lattice`, lo: 14.19, hi: 14.41, unit: 'GeV',
          items: [{ v: 14.2311, dev: 0.01116, cls: 'nnlo', style: 'solid', lab: 'MRS 14.2311(1)(112)', side: 'down', devLab: 'max deviation ±11 MeV' },   /* Tab. III */
                  { v: 14.282, style: 'hatchpt', cls: 'nnlo', lab: 'fixed order 14.282(1)(706)', side: 'up', anchor: 'middle' },                        /* Tab. III (band ±706 MeV not drawn to scale) */
                  { v: 14.366, dev: 0.009, cls: 'lat', style: 'square', lab: 'lattice 14.366(7)(9) [31]', side: 'down', devLab: 'stat ⊕ sys' }],       /* Tab. III */
          gap: { a: 14.2311, b: 14.366, lab: '~135 MeV' } });
        NLINE.b.set(...P(114, 112, 0)); NLINE.b.w = U(720, DIST); NLINE.b.show(seg(p, .55, .7)); NLINE.b.touch(); NLINE.reveal(seg(p, .6, 1), 0.15);
      } }, [9, 9]),
      /* b3: the band gauge returns for Ω_bbb: the hatched ~700 MeV bar collapses into a solid ~10 MeV sliver */
      { d: 18, f(p) {
        GAUGE.set({ title: `The scale band, ${H.Obbb} at NNLO`, rows: [
          { lab: 'fixed order', cls: 'nnlo', style: 'hatch', w: 1.0, val: '~700 MeV', ghost: true },   /* Tab. III: 706; its width stays as an outline, so the ratio is still on the board after the collapse */
          { lab: 'MRS', cls: 'nnlo', style: 'solid', w: 11.16 / 706, val: '~10 MeV' },              /* Tab. III: 11.16 MeV; 706 → 11 */
        ], note: '~700 MeV → ~10 MeV: <b class="tc">nearly two orders of magnitude</b>. One-sided envelope, not a symmetric 1σ bar.' });
        GAUGE.b.set(...P(114, 88, 0)); GAUGE.b.w = U(760, DIST); GAUGE.b.show(seg(p, 0, .2)); GAUGE.b.touch(); nb.show(1 - seg(p, 0, .2));
        const u = eo(seg(p, .3, .75)); GAUGE.width(0, 1 - (1 - 11.16 / 706) * u); GAUGE.width(1, 11.16 / 706 * seg(p, .7, .85));
        GAUGE.rows[1].r.style.opacity = seg(p, .7, .85); rv(GAUGE.b.q('#gnote'), seg(p, .8, 1));
      } },
      /* b4: bars gap ÷ lattice */
      { d: 16, f(p) {
        NLINE.b.show(1 - seg(p, 0, .25)); gaps.show(seg(p, .2, .4));
        gbars.forEach((gb, i) => { const u = eo(seg(p, .35 + i * .12, .6 + i * .12)); gb.r.setAttribute('height', u * gb.g * 30); gb.r.setAttribute('y', 140 - u * gb.g * 30); gb.t.setAttribute('opacity', seg(p, .5 + i * .12, .7 + i * .12)); });
      } },
      /* b5: the dim top row gets Ω_tcc … Ω_ttt. The camera travels back to the shelf and stays close on it: the six masses land,
         then the summary comes over the top row. The frame is as tight as the two rows allow (every number 18 px or more) */
      { d: 12, slow: 1.5, win: [0, .36], cam: [KS([0, 75, 86], [0, 75, 0])], f(p) {
        const u = eio(seg(p, 0, .35)); SHELF.place('s10a', 's10b', u);
        LATT.show(1 - seg(p, 0, .2));                            /* the wall stood behind the baryons for the lattice numbers; from this frame its grid ran through the top row */
        TOP.forEach(([id, v], i) => { const G = SHELF.glyphs[id]; const k = seg(p, .26 + i * .05, .38 + i * .05); G.g.setOpacity(0.55 * k); G.g.setGlow(0.6); G.lab.show(k);
          /* the Ω_ttt note stands beside its chip (it is the last of the row): under it, it ran into the lower level of chips */
          G.chip.html(id === 'Ottt' ? `<span style="position:relative;display:inline-block"><span class="chip">${v}</span><span class="tiny" style="position:absolute;left:100%;top:9px;margin-left:4px;white-space:nowrap">±0.94 GeV band</span></span>` : `<span class="chip">${v}</span>`).show(k); });
        topN.show(seg(p, .75, .95));
      } },
    ],
  });
})();
