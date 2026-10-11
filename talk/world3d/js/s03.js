'use strict';
/* S03 pNRQCD: the dive through the EFT tower down to the static pair. */
(function () {
  const SC = 2, DIST = 50, TX = LAY.towerX, FY = LAY.floorsY, { P, V, KS } = FR(SC);
  /* b1's camera progress: arrive on QCD, hold; go down to NRQCD, hold; go down to pNRQCD */
  const CAMU = p => (eio(seg(p, 0, .16)) + eio(seg(p, .36, .52)) + eio(seg(p, .7, .88))) / 3;
  /* words that pass as the camera crosses each floor (S03-P1) */
  const pass = [
    BOARDS.board({ scene: SC, px: 520, w: U(520, 40), at: P(TX + 15, LAY.floorsY[0] + 10.5, 4), a: 0, html: `<div class="big ital">integrate out ${I('p')} ~ ${H.mQ}</div><div class="small muted">hard modes → matching coefficients</div>` }),
    BOARDS.board({ scene: SC, px: 520, w: U(520, 40), at: P(TX + 15, LAY.floorsY[1] + 8.7, 4), a: 0,   /* below the QCD floor's band in the NRQCD stop: both lines clear its edges */ html: `<div class="big ital">integrate out ${I('p')} ~ ${H.mQ}${I('v')}</div><div class="small muted">soft modes → the potentials ${H.VR}(${I('r')})</div>` }),
  ];
  const remains = BOARDS.board({ scene: SC, px: 520, w: U(520, DIST), at: P(TX, LAY.pairY + 3.7, 0), a: 0,   /* over the pair it is about, under the NRQCD floor (it stood on that floor's name) */ html: `<div class="big ital" style="text-align:center">what remains is quantum mechanics</div>` });
  /* labels that ride on the pair */
  const vr = BOARDS.board({ px: 120, w: U(120, DIST), a: 0, html: `<span class="chip">${H.VR}(${I('r')})</span>` }); vr.el.style.textAlign = 'center';
  const cf = BOARDS.board({ px: 150, w: U(150, DIST), a: 0, html: `<span class="chip">${H.CF} = 4/3</span>` }); cf.el.style.textAlign = 'center';   /* Eq. 2 */
  const cb1 = BOARDS.board({ scene: SC, px: 170, w: U(170, DIST), a: 0, html: `<span class="chip">${H.CF}/2 = 2/3</span>` }); cb1.el.style.textAlign = 'center';   /* Eq. 17 */
  const cb2 = BOARDS.board({ scene: SC, px: 170, w: U(170, DIST), a: 0, html: `<span class="chip">${H.CF}/2 = 2/3</span>` }); cb2.el.style.textAlign = 'center';
  const cb3 = BOARDS.board({ scene: SC, px: 170, w: U(170, DIST), a: 0, html: `<span class="chip">${H.CF}/2 = 2/3</span>` }); cb3.el.style.textAlign = 'center';   /* in the baryon all three pairs are QQ: the singlet's 4/3 must not stay on a quark-quark link */
  const tri = BOARDS.board({ scene: SC, px: 260, w: U(260, DIST), a: 0, html: `<span class="lbl">3-body: NNLO,<br>fixed order</span>` }); tri.el.style.textAlign = 'center';   /* two lines: one is wider than the triangle */
  const singlet = BOARDS.board({ px: 300, w: U(300, DIST), a: 0, html: `<span class="lbl">colour singlet: schematic</span>` }); singlet.el.style.textAlign = 'center';
  const stat = BOARDS.board({ scene: SC, px: 520, w: U(520, DIST), a: 0, html: `<span class="lbl">static = leading-order potential; the quarks still move (−∇²/2${I('m')})</span>` }); stat.el.style.textAlign = 'center';
  TICKS.push(() => {
    const A = PAIR.A.getWorldPosition(new THREE.Vector3()), B = PAIR.B.getWorldPosition(new THREE.Vector3()), C = PAIR.C.getWorldPosition(new THREE.Vector3());
    const s = PAIR.st.scale;
    vr.set((A.x + B.x) / 2, (A.y + B.y) / 2 + 2.2 * s, A.z); cf.set((A.x + B.x) / 2, (A.y + B.y) / 2 - 2.3 * s, A.z);
    cb1.set((A.x + C.x) / 2 - 3.2, (A.y + C.y) / 2, A.z); cb2.set((B.x + C.x) / 2 + 3.2, (B.y + C.y) / 2, A.z); cb3.set((A.x + B.x) / 2, (A.y + B.y) / 2 - 2.3 * s, A.z);
    tri.set((A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3 - 0.55, A.z + 1);
    singlet.set((A.x + B.x) / 2, A.y - 4.6 * s, A.z); stat.set((A.x + B.x) / 2, A.y - 5.9 * s, A.z);   /* above the floor's front edge in the b3 frame */
    vr.hidden = cf.hidden = singlet.hidden = stat.hidden = !PAIR.g.visible;
  });
  const ham = BOARDS.board({ scene: SC, cls: 'panel', px: 780, w: U(780, DIST), at: P(-66, -16, 0), a: 0, html: `<h2>The Hamiltonian</h2>
    <div class="m" style="font-size:28px;display:flex;align-items:center;flex-wrap:wrap;column-gap:6px">
      <span class="ti">${I('H')} =</span><span class="ti">${sum(I('i'))}${frac('−∇²' + sub(I('i')), '2' + I('m') + sub(I('i')))}</span><span class="ti">+ ${sum(I('i') + '&lt;' + I('j'))} ${H.VR}(${I('r')}${sub(I('ij'))}),</span>
      <span class="ti" style="margin-left:14px">${I('V')}${sub('1')}${sup('Q' + ov('Q'))} = −${H.CF} ${frac(I('α') + sub('V') + '(' + I('r') + ',' + I('μ') + ')', I('r'))}</span></div>
    <div class="small t" style="margin-top:8px">an ordinary Schrödinger problem (+ 3-body terms at NNLO): potentials <b>matched</b> to QCD, not modelled</div>` });
  const mt = BOARDS.board({ scene: SC, cls: 'panel', px: 780, w: U(780, DIST), at: P(-66, -27, 0), a: 0, html: `<h2>Matching</h2>
    <div class="big">Potentials known to <b>NNLO</b> in ${H.as}</div>
    <div style="margin-top:8px"><span class="chip t">Q${ov('Q')} singlet: ${H.CF} = 4/3</span> <span class="chip t">QQ in baryon: ${H.CF}/2</span></div>
    <div class="small t" style="margin-top:8px">3-body forces enter at NNLO: included at fixed order (no resummation known)</div>` });
  const st = BOARDS.board({ scene: SC, cls: 'panel', px: 780, w: U(780, DIST), at: P(-66, -36, 0), a: 0, html: `<h2>Static limit</h2>
    <div class="big">𝒪(1/${H.mQ}) and spin terms dropped here. Spin effects are 𝒪(${I('v')}²): ~5% (${I('b')}) / ~10% (${I('c')}) of the binding energy <span class="small">(§II A)</span></div>` });
  const hparts = ham.qa('.ti');
  scenes.push({
    id: 'S03', title: 'pNRQCD: QCD as a Schrödinger problem', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      /* b1: three stops, one per floor. On each floor the same pair is shown at that floor's resolution, by its own motion only
         (no gluon cloud): fast and wide on QCD (hard modes, p ~ m_Q), slower on NRQCD (soft modes, p ~ m_Q v), the two static
         quarks on pNRQCD, where the link that becomes V_R appears. Each step's words stay while the camera goes down. */
      ...cut({ d: 18, slow: 1.5, camU: CAMU, cam: [KS([TX + 5, FY[0] + 13, 46], [TX + 3, FY[0] + 4, -2]), KS([TX + 5, FY[1] + 13, 46], [TX + 3, FY[1] + 4, -2]), KS([TX - 12, LAY.pairY + 10, DIST], [TX - 14, LAY.pairY, -2])], f(p) {
        scenes[SC - 1].ghost = 1 - seg(p, .03, .14);           /* the S02 boards hang at the QCD floor's height: they leave, so the pair is read alone */
        TOWER.floors.forEach((f, i) => { const k = eo(seg(p, .02 + i * .06, .16 + i * .06)); TOWER.show(k, i); f.m.scale.set(0.4 + 0.6 * k, 1, 0.05 + 0.95 * k); f.edge.scale.copy(f.m.scale); });
        const lvl = clamp(CAMU(p) * 3 - 1, 0, 2);              /* 0 = on the QCD floor, 1 = NRQCD, 2 = pNRQCD */
        [1, 2].forEach(i => TOWER.floors[i].lab.show(seg(lvl, i - .8, i - .3)));   /* a floor's name comes on as the camera leaves the floor above: at that stop the stage edge cut it */
        const fy = lvl < 1 ? lerp(FY[0], FY[1], lvl) : lerp(FY[1], FY[2], lvl - 1);
        PAIR.g.position.set(...P(TX, fy + 4, LAY.pairZ)); PAIR.st.scale = 1; PAIR.st.sep = 7;
        PAIR.st.opacity = seg(p, .08, .16) * (1 - 0.75 * Math.sin(Math.PI * (lvl % 1)));   /* it dims while travelling between two floors */
        PAIR.st.jit = lvl < 1 ? lerp(7, 3.2, lvl) : lerp(3.2, 1, lvl - 1); PAIR.st.fq = lvl < 1 ? lerp(4.5, 2.2, lvl) : lerp(2.2, 1, lvl - 1);
        PAIR.st.lkOp = seg(lvl, 1.85, 2);                       /* the potential link exists only once the soft modes are integrated out */
        pass[0].show(seg(p, .2, .28) * (1 - seg(p, .48, .56))); pass[1].show(seg(p, .54, .62) * (1 - seg(p, .7, .78)));   /* each group of words has its own stop now: they go as the camera leaves the floor */
        remains.show(seg(p, .9, 1));
      } }, [6, 6.5, 5.5]),                                      /* three stops: QCD with its words, NRQCD with its words, pNRQCD (the camera holds at each cut) */
      { d: 14, cam: [KS([-54, -9, 52], [-54, -20, 0])], f(p) {
        remains.show(1 - seg(p, 0, .3));
        PAIR.st.opacity = 1; PAIR.st.third = 0; PAIR.st.tri = 0;
        vr.show(seg(p, .3, .45)); singlet.show(seg(p, .35, .5)); stat.show(seg(p, .5, .7));
        ham.show(seg(p, .25, .4)); hparts.forEach((e, i) => rv(e, seg(p, .35 + i * .12, .5 + i * .12))); rv(ham.q('.small'), seg(p, .8, 1));
        cf.show(0);
      } },
      ...cut({ d: 17, f(p) {
        /* 4/3 on the quark-antiquark singlet; then the antiquark splits (anti-red = green + blue) and every pair of the baryon is QQ */
        cf.show(seg(p, 0, .1) * (1 - seg(p, .18, .26))); vr.show(1 - seg(p, .18, .26));
        const k = seg(p, .2, .4);                                /* the baryon stays through b4; S04 b1 takes it away */
        PAIR.st.third = k; PAIR.st.tri = seg(p, .42, .55);
        cb1.show(seg(p, .36, .46)); cb2.show(seg(p, .39, .49)); cb3.show(seg(p, .42, .52)); tri.show(seg(p, .5, .62));
        mt.show(seg(p, .2, .35)); BOARDS.T(mt, '.t', seg(p, .4, .7)); rv(mt.q('.small'), seg(p, .6, .8));
      } }, [3, 14], [{ cam: [KS([-54, -11, 56], [-54, -22.5, 0])], win: [0, 1] }, null]),   /* a stop on the singlet's 4/3 (the camera, high enough for the whole NRQCD name, settles in it), then the baryon */
      { d: 14, f(p) { st.show(seg(p, 0, .4)); } },
    ],
  });
  window.__S03 = { vr, cf, singlet, cb1, cb2, cb3, tri };
})();
