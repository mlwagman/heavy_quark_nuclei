'use strict';
/* S08 Results I: quarkonium, tuning m_c and m_b · S09 Results II: toponium and mixed-flavour mesons.
   The camera is back at the shelf; every result lands on it. */

/* more shelf glyphs (created once, lit when their scene comes) */
SHELF.mk('bc', 2, { label: `${H.bc}<div class="lbl">B${sub('c')}, parameter-free</div>` });
SHELF.mk('et', 2, { row: 'top', side: true, label: `${H.et}: zero-width limit<br>(top decays)` });
SHELF.mk('tc', 2, { row: 'top', label: `${H.tc}` });
SHELF.mk('tb', 2, { row: 'top', label: `${H.tb}` });
SHELF.mk('Occb', 3, { label: `${H.Occb}` }); SHELF.mk('Ocbb', 3, { label: `${H.Ocbb}` }); SHELF.mk('Obbb', 3, { label: `${H.Obbb}` });
['Otcc', 'Otcb', 'Otbb', 'Ottc', 'Ottb', 'Ottt'].forEach(id => SHELF.mk(id, 3, { row: 'top', label: H[id] }));
SHELF.mk('bbtt', 4, { row: 'top', label: `${H.bbtt}`, tetra: true }); SHELF.mk('bctt', 4, { row: 'top', label: `${H.bctt}`, tetra: true }); SHELF.mk('cctt', 4, { row: 'top', label: `${H.cctt}`, tetra: true });
SHELF.mk('bbbb', 4, { row: 'top', label: `${H.bbbb}`, tetra: true }); SHELF.mk('ccbb', 4, { row: 'top', label: `${H.ccbb}`, tetra: true });
SHELF.layout('s09a', ['cc', 'bb', 'Occc', 'T4'], ['et']);
SHELF.layout('s09b', ['cc', 'bb', 'bc', 'Occc', 'T4'], ['et']);
SHELF.layout('s09c', ['cc', 'bb', 'bc', 'Occc', 'T4'], ['et', 'tc', 'tb']);
SHELF.layout('s10a', ['cc', 'bb', 'bc', 'Occc', 'Occb', 'Ocbb', 'Obbb', 'T4'], ['et', 'tc', 'tb']);
SHELF.layout('s10b', ['cc', 'bb', 'bc', 'Occc', 'Occb', 'Ocbb', 'Obbb', 'T4'], ['et', 'tc', 'tb', 'Otcc', 'Otcb', 'Otbb', 'Ottc', 'Ottb', 'Ottt']);
SHELF.layout('s12', ['cc', 'bb', 'bc', 'Occc', 'Occb', 'Ocbb', 'Obbb', 'T4'], ['et', 'tc', 'tb', 'Otcc', 'Otcb', 'Otbb', 'Ottc', 'Ottb', 'Ottt', 'bbtt', 'bctt', 'cctt']);

/* a raster paper figure on a white card inside the World (still a rebuilt scene: a board in space) */
function figBoard(SC, title, src, cap, at, px, dist, capCls = '') {
  return BOARDS.board({ scene: SC, cls: 'panel', px, w: U(px, dist), at, a: 0, html: `<h2>${title}</h2><div style="background:#fff;border-radius:4px;padding:6px"><img src="${src}" style="width:100%"></div><div class="small t ${capCls}" style="margin-top:8px">${cap}</div>` });
}

/* ======================= S08 ======================= */
(function () {
  const SC = 7, DIST = 120, { P, V, KS } = FR(SC);
  const P1 = figBoard(SC, 'Fixed order', 'assets/s08-fixed-order.png', 'broad bands: LO does not even overlap the higher orders', P(-56, 114, 0), 600, DIST);    /* Fig. 1 left */
  const P2 = figBoard(SC, 'MRS', 'assets/s08-mrs.png', `2–3× narrower at ${H.mb} (Fig. 1 caption): orders now partly overlap`, P(-8, 114, 0), 600, DIST);       /* Fig. 1 right */
  const dft = BOARDS.board({ scene: SC, cls: 'panel', px: 510, w: U(510, DIST), at: P(-30, 87.5, 0), a: 0,   /* wide enough for the full bar and its value; under the two figures, not on their lower edge */ html: `<h2>Tuned to spin-averaged 1S: mass drift by order</h2>
    <div class="row"><div class="small" style="width:110px;flex:none">fixed order</div><div class="bar hatch nnlo" id="d1" style="width:0;height:15px;flex:none"></div><div class="small num" id="v1" style="white-space:nowrap"></div></div>
    <div class="row"><div class="small" style="width:110px;flex:none">MRS</div><div class="bar solid nnlo" id="d2" style="width:0;height:15px;flex:none"></div><div class="small num" id="v2" style="white-space:nowrap"></div></div>
    <div class="tiny" style="margin-top:6px">drift LO → NNLO of the tuned ${H.mc}, ${H.mb} (Tab. I + caption)</div>` });
  /* two dials under cc̄ and bb̄: the Λ_R chip splits into them (S08 b3) */
  const dial = () => { const b = BOARDS.board({ px: 120, w: U(120, DIST), a: 0, html: `<svg viewBox="0 0 120 88" width="120" height="88"><path d="M15 60 A45 45 0 0 1 105 60" fill="none" stroke="var(--muted)" stroke-width="2"/><line id="nd" x1="60" y1="60" x2="60" y2="20" stroke="var(--ink)" stroke-width="2.5"/><text x="60" y="82" text-anchor="middle" font-size="15" fill="var(--muted)">Λ<tspan baseline-shift="sub" font-size="11">R</tspan> → m<tspan baseline-shift="sub" font-size="11">Q</tspan></text></svg>` }); return b; };   /* the words under the dial, clear of its needle */
  const dialC = dial(), dialB = dial();
  const mt = BOARDS.board({ px: 300, w: U(300, DIST), a: 0, html: `<span class="chip">${H.mt} = 172.52 GeV · pole mass, input</span>` }); mt.el.style.textAlign = 'center';    /* Tab. I caption, §III B */
  const pred = BOARDS.board({ scene: SC, px: 1000, w: U(1000, DIST), at: P(-32, 137, 0), a: 0, html: `<div class="banner" style="text-align:center;border:none;font-size:28px">Everything after this is a prediction: two tuned numbers + one input.</div>` });
  const lam1 = window.__S05.lam1;
  const MT = W.V3(31.5, LAY.shelfY - 8.2, 0), MTw = () => SHELF.at.clone().add(MT);   /* the input waits in the chip row, right of the family: two tuned numbers and one input on one line (shelf coordinates: it travels with the shelf) */
  TICKS.push(() => { const c = SHELF.wp('cc'), b = SHELF.wp('bb'); dialC.set(c.x, c.y - 10.4, c.z); dialB.set(b.x, b.y - 10.4, b.z); });
  scenes.push({
    id: 'S08', title: 'Quarkonium: tuning the charm and bottom masses', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: the walker cloud fades in place; the camera climbs back to the shelf, where cc̄ and bb̄ pulse on arrival */
      { d: 22, win: [0, .7], cam: [KS([-32, 96, DIST], [-32, 96, 0])], f(p) {   /* the shelf comes from its S02 place with the camera */
        SHELF.travel(1, SC, eio(seg(p, 0, .7)));
        WALK.set({ uOp: 0 }); PAIR.st.opacity = 0; PAIR.st.bOp = 0;   /* the quark and its cloud left in S07 b3, together: alone, the quark is not colour-neutral */
        window.__S07.insets.forEach(g => g.setOpacity(1 - seg(p, 0, .3)));
        ['cc', 'bb'].forEach(id => SHELF.glyphs[id].g.setGlow(1 + 0.8 * Math.sin(Math.PI * seg(p, .55, .85))));
        P1.show(seg(p, .6, .8)); rv(P1.q('.t'), seg(p, .85, 1));
      } },
      { d: 18, f(p) { P2.show(seg(p, 0, .35)); rv(P2.q('.t'), seg(p, .75, 1)); } },
      /* b3: the drift meter; the S05 chip arrives (a stop on it), then splits into two dials on the cc̄ and bb̄ glyphs */
      ...cut({ d: 20, f(p) {
        dft.show(seg(p, 0, .2)); const u1 = eo(seg(p, .2, .6)), u2 = eo(seg(p, .55, .85));
        dft.q('#d1').style.width = (260 * u1) + 'px'; dft.q('#v1').textContent = u1 > 0.45 ? '4–11%' : '';            /* Tab. I caption */
        dft.q('#d2').style.width = (21 * u2) + 'px'; dft.q('#v2').textContent = u2 > 0.5 ? '< 1%' : '';                 /* Tab. I caption */
        const c = SHELF.wp('cc'), b = SHELF.wp('bb');
        lam1.setV(W.V3((c.x + b.x) / 2, c.y - 10.4, c.z)); lam1.show(seg(p, .3, .45) * (1 - seg(p, .73, .82)));   /* round 18: the S05 chip appears where it lands (its S05 place is now across the World) */
        dialC.show(seg(p, .75, .9)); dialB.show(seg(p, .75, .9));
        const nd = 1 - seg(p, .8, 1); [dialC, dialB].forEach(d => d.q('#nd').setAttribute('transform', `rotate(${-60 + 120 * (1 - nd)} 60 60)`));
      } }, [14.5, 5.5]),
      /* b4: the tuned masses land on the shelf as chips; m_t arrives as an input */
      /* in two stages: close on the shelf while the two tuned numbers of the talk land (from the scene's frame they were 17 px),
         then back to the scene's frame for the sentence that says what they are for */
      { d: 16, slow: 1.5, camU: p => (eio(seg(p, 0, .25)) + eio(seg(p, .62, .9))) / 2,
        cam: [KS([-3.5, 76, 80], [-3.5, 76, 0]), KS([-32, 98.5, 126], [-32, 98.5, 0])], f(p) {   /* both frames hold the two-line chips whole */
        const k = seg(p, .2, .4); dialC.show(1 - k); dialB.show(1 - k);
        SHELF.glyphs.cc.chip.html(`<span class="chip pin">${H.mc} = 1.555 GeV</span><br><span class="tiny">effective mass · tuned</span>`).show(seg(p, .3, .42));   /* Tab. I, NNLO MRS 1.55489 */
        SHELF.glyphs.bb.chip.html(`<span class="chip pin">${H.mb} = 4.802 GeV</span><br><span class="tiny">effective mass · tuned</span>`).show(seg(p, .38, .5));   /* Tab. I, NNLO MRS 4.80221 */
        mt.setV(MTw()); mt.show(seg(p, .5, .6));
        pred.show(seg(p, .85, 1));
      } },
    ],
  });
  window.__S08 = { mt, MTw };
})();

/* ======================= S09 ======================= */
(function () {
  const SC = 8, DIST = 112, { P, V, KS } = FR(SC);
  const leg = BOARDS.board({ scene: SC, cls: 'panel', px: 900, w: U(900, DIST), at: P(48, 104, 0), a: 0, html: `<h2>Reading the numbers</h2><div class="big">first parenthesis: <span class="tc">GFMC statistics</span> · second: <span class="rc">scale variation ${I('λ')} ∈ {½, 1, 2}</span> &nbsp;·&nbsp; ${H.mt} = 172.52 GeV pole mass, input</div>` });
  const ghostC = BOARDS.board({ px: 220, w: U(220, DIST), a: 0, html: `<span class="chip">${H.mc}</span>` }), ghostB = BOARDS.board({ px: 220, w: U(220, DIST), a: 0, html: `<span class="chip">${H.mb}</span>` });
  const nocmp = BOARDS.board({ scene: SC, cls: 'panel', px: 520, w: U(520, DIST), at: P(48, 90, 0), a: 0, html: `<h2>No comparison yet</h2><div class="big">${H.tc} and ${H.tb}: benchmarks, not observables.</div>` });
  const mt = window.__S08.mt;
  /* the two number lines of this scene */
  const ET = { key: 'et', title: `${H.et} at NNLO MRS: second parenthesis = max deviation, ±0.77 GeV drawn to scale`, lo: 341.3, hi: 344.3, unit: 'GeV',
    items: [{ v: 342.47, dev: 0.77, cls: 'nnlo', style: 'solid', lab: '342.47(4)(77) GeV', side: 'down', devLab: 'max deviation (λ ∈ {½, 1, 2}), one envelope' },   /* Tab. I */
            { v: 343, style: 'tickmark', lab: '≈ 343 (NRQCD threshold)', side: 'up' }] };                /* Tab. I comparison column [75] */
  const BC = { key: 'bc', title: `B${sub('c')}: parameter-free. Once ${H.mc}, ${H.mb} are fixed there is nothing left to adjust`, lo: 6.262, hi: 6.334, unit: 'GeV',   /* from 6.262 since round 19: room for the measured B_c and its label */
    items: [{ v: 6.29121, dev: 0.00731, cls: 'nnlo', style: 'solid', lab: 'ours 6.29121 GeV', side: 'down', devLab: 'max deviation ±7.31 MeV' },          /* Tab. II */
            { v: 6.2745, style: 'square', lab: `exp. B${sub('c')} 6.2745(3) GeV`.replace('<sub>', '<tspan baseline-shift="sub" font-size="12">').replace('</sub>', '</tspan>'), side: 'up' },   /* the measured B_c, the lighter spin state (Sec. on Tab. II, [92]): it enters the spin average beside it */
            { v: 6.316, dev: 0.007, style: 'square', lab: 'exp. + lattice 6.316(7) GeV', side: 'up', devLab: '±7 MeV' }],                            /* Tab. II [92, 96] */
    gap: { a: 6.29121, b: 6.316, lab: '24.8 MeV gap' } };
  scenes.push({
    id: 'S09', title: 'Toponium and mixed-flavour mesons', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: the legend lands first (story edit E3) */
      { d: 20, cam: [KS([14, 92, DIST], [14, 92, 0])], f(p) { SHELF.travel(7, SC, eio(seg(p, 0, .6))); mt.setV(window.__S08.MTw());   /* the shelf, with the m_t chip on it, moves on with the camera */
        SHELF.showTop(seg(p, .2, .5)); leg.show(seg(p, .1, .4)); } },
      /* b2: η_t joins the dim top row; the m_t chip flies to it; the number line near 2m_t with the wide band drawn to scale */
      { d: 18, f(p) {
        SHELF.place('fam', 's09a', 1); const G = SHELF.glyphs.et; const k = seg(p, .1, .35);
        scenes[SC - 1].ghost = 1 - seg(p, 0, .15);               /* the S08 boards hang where the top row begins: they leave before its first name arrives */
        G.g.setOpacity(0.55 * k); G.g.setGlow(0.6); G.lab.show(k);
        mt.lerpTo(window.__S08.MTw(), SHELF.wp('et').add(W.V3(0, -5.58, 0)), eio(seg(p, .05, .35))); mt.show(1 - seg(p, .3, .4));   /* to the place where the η_t mass appears, not across the name */
        G.chip.html(`<span class="chip">342.47(4)(77) GeV</span>`).show(seg(p, .35, .5));                       /* Tab. I, η_t NNLO MRS */
        NLINE.config(ET);
        NLINE.b.set(...P(48, 120, 0)); NLINE.b.w = U(720, DIST); NLINE.b.show(seg(p, .4, .55)); NLINE.b.touch(); NLINE.reveal(seg(p, .45, 1));
      } },
      /* b3: the m_c and m_b chips meet (a stop on them), then the bc̄ glyph builds from them; the B_c number line */
      ...cut({ d: 13, f(p) {
        const u = eio(seg(p, 0, .44)); SHELF.place('s09a', 's09b', u); const G = SHELF.glyphs.bc;   /* the two mass chips travel ~3.5 s */
        /* m_c and m_b leave their glyphs and meet side by side where the bc̄ glyph forms: along the glyph row, clear of every name and chip */
        const c = SHELF.wp('cc'), b = SHELF.wp('bb'), t = SHELF.wp('bc');
        ghostC.lerpTo(c, t.clone().add(W.V3(-2.4, 0, 0)), u); ghostB.lerpTo(b, t.clone().add(W.V3(2.4, 0, 0)), u); ghostC.show(seg(p, 0, .1) * (1 - seg(p, .47, .55))); ghostB.show(seg(p, 0, .1) * (1 - seg(p, .47, .55)));
        const k = seg(p, .47, .62); G.g.setOpacity(k); G.g.setGlow(1); G.lab.show(k);
        G.chip.html(`<span class="chip" style="padding:4px 7px">6.29121(4)(731) GeV</span>`).show(seg(p, .6, .72));                     /* Tab. II, bc̄ NNLO */
        const z = eio(seg(p, .05, .44)); NLINE.zoom(ET, BC, z);   /* one ruler: it travels from 342 GeV to 6.3 GeV while the glyph forms, then B_c lands on it */
        NLINE.b.set(...P(48, 120, 0)); NLINE.b.show(1); NLINE.b.touch(); if (z >= 1) NLINE.reveal(seg(p, .55, 1), 0.2);
      } }, [6, 7]),
      /* b4: tc̄ and tb̄ join the dim top row */
      { d: 14, f(p) {
        const u = eio(seg(p, 0, .35)); SHELF.place('s09b', 's09c', u);
        [['tc', '173.98462(4)(883) GeV'], ['tb', '177.00684(39)(2372) GeV']].forEach(([id, v], i) => { const G = SHELF.glyphs[id]; const k = seg(p, .2 + i * .15, .45 + i * .15);
          G.g.setOpacity(0.55 * k); G.g.setGlow(0.6); G.lab.show(k); G.chip.html(`<span class="chip">${v}</span>`).show(k); });   /* Tab. II, NNLO */
        nocmp.show(seg(p, .5, .8));
      } },
    ],
  });
})();
