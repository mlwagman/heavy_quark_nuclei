'use strict';
/* S13 Synthesis · S14 Conclusions · S15 Key references. Every band gauge lands in one table; the shelf is full. */

/* ======================= S13 ======================= */
(function () {
  const SC = 12, DIST = 112, { P, V, KS } = FR(SC);
  const ROWS = [
    [`scale band, ${H.Obbb} at NNLO`, '~700 MeV', '~10 MeV'],                            /* Tab. III: 706 → 11.16 */
    [`order-by-order shifts (baryons, |NNLO−NLO|)`, '33–55 MeV', '4–18 MeV'],            /* Tab. III derived (#46): FO 44/39/55/33, MRS 3.6/9.0/13.8/18.3 */
    [`tuned ${H.mc}, ${H.mb} drift`, '4–11%', '&lt; 1%'],                                /* Tab. I + caption */
    [`LO / NLO / NNLO bands`, 'LO apart', 'overlap (partial)'],                          /* Figs. 1–2, §IV A */
  ];
  /* under the numbers of rows 1 to 3, a bar each, in the gauge's vocabulary (hatched = fixed order, solid = MRS): MRS over fixed order,
     from the larger number of each range (10/700, 18/55, 1/11). The fourth row is a statement, not a number: no bar. */
  const BW = 120, RATIO = [10 / 700, 18 / 55, 1 / 11, null];
  const bar = (id, kind, i) => RATIO[i] == null ? '' : `<div class="bar ${kind}" id="${id}${i}" style="width:0;height:8px;margin-top:4px;color:var(--ink2)"></div>`;
  const tb = BOARDS.board({ scene: SC, cls: 'panel', keep: true, px: 900, w: U(900, DIST), at: P(-30, 292, 0), a: 0, html: `<table><tr><th>diagnostic (c, b)</th><th>fixed order</th><th></th><th>MRS</th></tr>${ROWS.map((r, i) => `<tr><td>${r[0]}</td><td class="t num" id="a${i}" style="min-width:${BW}px;white-space:nowrap">${r[1]}${bar('fb', 'hatch', i)}</td><td class="t muted" id="r${i}">→</td><td class="t num" id="c${i}" style="min-width:${BW}px;white-space:nowrap">${r[2]}${bar('mb', 'solid', i)}</td></tr>`).join('')}</table>` });
  const h1 = BOARDS.board({ scene: SC, cls: 'panel', px: 640, w: U(640, DIST), at: P(-30, 262, 0), a: 0, html: `<h2>Hallmarks of improved convergence</h2>
    <div class="big" style="font-size:23px">For charm and bottom hadrons, every diagnostic moves the same way. Taken together they say the leading renormalon <i>was</i> the obstruction: not one problem among several.</div>
    <div class="small t" style="margin-top:8px">still asymptotic, only the leading renormalon is gone</div>
    <div class="small t" style="margin-top:4px">λ-bands are a scheme-stability diagnostic, not a truncation-error estimate: order-by-order shifts gauge the truncation.</div>` });
  const h2 = BOARDS.board({ scene: SC, cls: 'panel', px: 640, w: U(640, DIST), at: P(30, 262, 0), a: 0, html: `<h2 class="tc">The payoff</h2>
    <div class="big" style="font-size:23px">Order-by-order shifts of <span class="num">4–18 MeV</span> make the remaining <span class="num">125–175 MeV</span> gap <i>interpretable</i>: consistent with missing 1/${H.mQ} physics, <b>not perturbative noise</b>.</div>` });
  BOARDS.topAlign(h1, h2);
  scenes.push({
    id: 'S13', title: 'What did renormalon subtraction buy us?', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: the band gauge flies in from S10 and becomes row 1; the other rows follow */
      ...cut({ d: 26, win: [0, .48], cam: [KS([0, 275, DIST], [0, 275, 0])], f(q) {   /* a stop on the gauge in row 1, then the table */
        const u = eio(seg(q, 0, .46)); GAUGE.b.lerpTo(TW(9, 114, 88, 0), V(-30, 294, 0), u);   /* from its S10 place */ GAUGE.b.show(seg(q, 0, .1) * (1 - seg(q, .5, .56))); GAUGE.b.touch();
        tb.show(seg(q, .56, .64));                                /* the gauge is gone before the table comes: cross-faded in one place, the two were written on each other */
        const p = q < .6 ? 0 : .55 + (q - .6) * 1.125;             /* the rows keep their old pace over the rest of the beat */
        ROWS.forEach((r, i) => { rv(tb.q('#a' + i), seg(p, .55 + i * .1, .63 + i * .1)); rv(tb.q('#r' + i), seg(p, .59 + i * .1, .67 + i * .1)); rv(tb.q('#c' + i), seg(p, .63 + i * .1, .71 + i * .1));
          if (RATIO[i] == null) return;                          /* the fixed-order bar grows with its number; the MRS bar starts as long and shrinks to its own length: the same move in every row */
          tb.q('#fb' + i).style.width = (BW * eo(seg(p, .55 + i * .1, .66 + i * .1))) + 'px';
          tb.q('#mb' + i).style.width = (BW * seg(p, .62 + i * .1, .64 + i * .1) * lerp(1, RATIO[i], eo(seg(p, .66 + i * .1, .8 + i * .1)))) + 'px'; });
      } }, [13, 13]),
      { d: 20, f(p) { h1.show(seg(p, 0, .3)); BOARDS.T(h1, '.t', seg(p, .5, .9)); } },
      /* b3: the S10 gap arrow comes back on the number line */
      { d: 22, f(p) {
        h2.show(seg(p, 0, .3));
        NLINE.config({ key: 'payoff', title: `ours vs lattice: the gap, now meaningful`, lo: 0, hi: 1, unit: '',
          items: [{ v: 0.3, dev: 0.02, cls: 'nnlo', style: 'solid', lab: 'ours: shifts 4–18 MeV', side: 'up', devLab: '~±10 MeV' },
                  { v: 0.72, style: 'square', cls: 'lat', lab: 'lattice', side: 'up' }],
          gap: { a: 0.3, b: 0.72, lab: '← 125–175 MeV, now meaningful →' }, note: `→ missing 1/${H.mQ} physics (§IV B, §VI)` });
        NLINE.b.set(...P(34, 292, 0)); NLINE.b.w = U(720, DIST); NLINE.b.show(seg(p, .3, .45)); NLINE.b.touch(); NLINE.reveal(seg(p, .35, 1), 0.2);
      } },
    ],
  });
})();

/* ======================= S14 ======================= */
(function () {
  /* three framings, the whole shelf in each: the claims above-left of it (b1, b2), the predictions above it (b3), the outlook
     below-right (b4). Every board sits in a place no other scene uses (tools/contact.mjs --layout). */
  const SC = 13, { P, V, KS } = FR(SC);
  const c1 = BOARDS.board({ scene: SC, cls: 'panel', px: 560, w: 70, at: P(-116, 137.2, 0), a: 0, html: `<div class="big"><b>First application of MRS inside a bound-state pNRQCD calculation</b>: NNLO static potentials + VMC / GFMC.</div>` });
  const c2 = BOARDS.board({ scene: SC, cls: 'panel', px: 560, w: 70, at: P(-116, 103.6, 0), a: 0,   /* clear of the first claim on the map too, where thumbnail borders grow */ html: `<div class="big">Scale sensitivity of charm and bottom hadrons down by <span class="num tc">up to two orders of magnitude</span>; results stable order by order.</div>` });   /* Tab. III: 706 → 11 */
  const chips = ['B' + sub('c'), 'η' + sub('t') + ' (toponium)', 'Ω' + sub('ccc') + ' … Ω' + sub('bbb'), 'top baryons → Ω' + sub('ttt'), `bound ${H.bbtt} · ${H.bctt} · ${H.cctt}`];
  const c3 = BOARDS.board({ scene: SC, cls: 'panel', px: 560, w: 44, at: P(51.5, 125.5, 0), a: 0, html: `<h2>Predictions: 2 tuned masses + ${H.mt} input</h2><div>${chips.map(c => `<span class="chip t">${c}</span>`).join(' ')}</div>` });
  const thesis = BOARDS.board({ scene: SC, px: 1000, w: 68, at: P(51.5, 115.6, 0), a: 0, html: `<div style="font-size:26px;font-style:italic;text-align:center">A decades-old infrared obstruction: the <b>leading</b> renormalon → removed → a whole family of states becomes predictable.</div>` });
  const c4 = BOARDS.board({ scene: SC, cls: 'panel', px: 560, w: 46, at: P(92.5, 44.5, 0), a: 0, html: `<h2>Next</h2><div class="big">𝒪(1/${H.mQ}) + spin potentials: where the gap points · MRS derivation for non-singlet and 3-body channels · NNLO tetraquarks · excited states · the t${ov('t')} threshold vs new LHC data · dark-sector bound states.</div>` });
  const ty = BOARDS.board({ scene: SC, px: 640, w: 49.2, at: P(92.5, 24, 0), a: 0, html: `<div style="display:flex;align-items:center;justify-content:center;gap:40px"><div style="font:italic 36px/1.3 var(--serif)">Thank you <span class="small" style="font-style:normal">: happy to take questions</span></div><div style="text-align:center"><img src="assets/qr.png" style="width:130px;border-radius:6px"><div class="tiny">arXiv 2607.08817</div></div></div>` });
  const FLASH = ['bc', 'et', 'Occc', 'Occb', 'Ocbb', 'Obbb', 'Otcc', 'Otcb', 'Otbb', 'Ottc', 'Ottb', 'Ottt', 'bbtt', 'bctt', 'cctt'];
  scenes.push({
    id: 'S14', title: 'Conclusions and outlook', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: down from the table to the whole shelf, the first claim above-left of it */
      { d: 14, win: [0, .75], cam: [KS([-42.5, 101.5, 184], [-42.5, 101.5, 0])], f(p) { SHELF.travel(11, SC, eio(seg(p, 0, .75))); NLINE.b.show(1 - seg(p, 0, .2)); c1.show(seg(p, .5, .8));
        window.__S12.sketch.a = 1 - seg(p, 0, .25); } },   /* the shelf comes up from S12 with the camera; the S12 caption about its glyphs goes with it (.a: no touch, it stays an S12 ghost) */   /* the shelf comes up from S12 with the camera */
      { d: 12, f(p) { c2.show(seg(p, 0, .4)); } },
      /* b3: closer on the shelf: the glyphs flash one by one as the narration names them; the S01 thesis line comes back above it */
      { d: 18, cam: [KS([11.5, 93.75, 122], [11.5, 93.75, 0])], f(p) {   /* the whole top row, the η_t name left of it included, and the thesis line */
        c3.show(seg(p, 0, .2)); BOARDS.T(c3, '.t', seg(p, .15, .6));
        FLASH.forEach((id, i) => { const G = SHELF.glyphs[id]; const t = seg(p, .2 + i * .045, .32 + i * .045); const base = G.top ? 0.6 : 1; G.g.setGlow(base + 1.2 * Math.sin(Math.PI * t)); });
        thesis.show(seg(p, .8, 1));
      } },
      /* b4: Next, Thank you, then the World lifts off into the journey map (the overview) and stays there for questions */
      ...cut({ d: 20, win: [0, .4], cam: [KS([27.35, 66.5, 147.6], [27.35, 66.5, 0])], f(p) { c4.show(seg(p, .38, .46)); ty.show(seg(p, .42, .5)); OV.beat = eio(seg(p, .55, .95)); OV.trace = seg(p, .62, 1); } }, [10, 10]),   /* a stop on the outlook and the thank-you (round 18: the lift-off shrank them to 4 px in the beat that wrote them), then the lift-off */
    ],
  });
})();

/* ======================= S15 ======================= */
(function () {
  const SC = 14, DIST = 150, { P, V, KS } = FR(SC);
  const refs = BOARDS.board({ scene: SC, cls: 'panel', px: 1600, w: U(1600, DIST), at: P(0, 334, -6), a: 0, html: `<h2>Key references</h2>
    <div style="display:flex;gap:48px;font-size:19px;line-height:1.7;color:var(--ink2)">
    <div><b class="white">This work &amp; predecessors</b><br>Assi, Kronfeld, Vaiva, Wagman: arXiv:2607.08817 · FERMILAB-PUB-26-0061-T (2026)<br>Assi, Wagman: PRD 108, 096004 (2023) · PRD 110, 094001 (2024)<br>Assi, Grebe, Wagman: PRD 113, 016013 (2026)<br><br>
    <b class="white">MRS</b><br>Komijani: JHEP 08 (2017) 062<br>Brambilla, Komijani, Kronfeld, Vairo (TUMQCD): PRD 97, 034503 (2018)<br>Kronfeld: JHEP 12 (2023) 108<br>Bazavov et al. (FNAL/MILC/TUMQCD): PRD 98, 054517 (2018)<br><br>
    <b class="white">pNRQCD / renormalons</b><br>Brambilla, Pineda, Soto, Vairo: RMP 77, 1423 (2005)<br>Beneke: Phys. Rept. 317, 1 (1999)</div>
    <div><b class="white" id="rgl">Lattice QCD comparisons</b><br>Meinel: PRD 82, 114514 (2010)<br>Brown, Detmold, Meinel, Orginos: PRD 90, 094507 (2014)<br>Mathur, Padmanath, Mondal: PRL 121, 202002 (2018)<br>Mathur, Padmanath, Chakraborty: PRL 130, 111901 (2023)<br>Gregory et al.: PRL 104, 022001 (2010)<br><br>
    <b class="white" id="rgb">Bounds, tetraquarks, toponium</b><br>Detmold: PRL 114, 222001 (2015)<br>Czarnecki, Leng, Voloshin: PLB 778, 233 (2018)<br>CMS: Rept. Prog. Phys. 88, 087801 (2025)<br>ATLAS: arXiv:2601.11780<br>Maltoni, Severi, Tentori, Vryonidou: arXiv:2401.08751<br>PDG: PRD 110, 030001 (2024)</div></div>` });
  /* threads from prediction chips to their reference group (catalogue continuity S14 → S15) */
  const threads = [['bc', 'gb'], ['et', 'gb'], ['Obbb', 'gl'], ['Occc', 'gl'], ['bbtt', 'gb']].map(([id, grp]) => { const l = W.cline([W.V3(), W.V3()], TC('wire'), 0); l.frustumCulled = false; W.crisp.add(l); return { id, grp, l }; });
  let tg = null;
  /* the threads stop at the panel's lower edge, under the column of their group: run on to the headings, they crossed the references */
  TICKS.push(() => { if (!threads[0].l.material.opacity) return; if (!tg) { const c = boardPoint(refs, '#rgl'), y = refs.anchor.y - refs.h * refs.w / refs.px / 2; tg = { gl: W.V3(c.x - 4, y, c.z), gb: W.V3(c.x + 4, y, c.z) }; }
    for (const t of threads) { const a = SHELF.wp(t.id), b = tg[t.grp]; t.l.geometry.setFromPoints([a, b]); } });
  scenes.push({
    id: 'S15', title: 'Key references', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    home: P(0, 320),                                         /* overview badge on the references: the scene opens on the map, over S14 */
    beats: [
      /* a stop on the map with the last line of its legend lit, then down onto the references */
      ...cut({ d: 14, win: [.22, .8], cam: [KS([0, 320, DIST], [0, 320, 0])], f(p) { OV.beat = 1 - eio(seg(p, .22, .65)); OV.trace = lerp(track.first[13] / track.first[14], 1, eio(seg(p, 0, .18)));   /* the line grows on to badge 15 during the stop */
        refs.show(seg(p, .55, .78)); threads.forEach((t, i) => t.l.material.opacity = 0.6 * seg(p, .7 + i * .05, .8 + i * .05)); } }, [3, 11]),
    ],
  });
})();
