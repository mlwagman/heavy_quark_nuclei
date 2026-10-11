'use strict';
/* S01 Title · S02 Motivation. The shelf is born here. */
const U = (px, dist) => px / W.ppu(dist);                     /* world width so a board shows at 1:1 css px from `dist` */
const K = W.KF;
/* a scene's own frame (round 18): P a point as [x, y, z], V as a vector, KS a camera keyframe, all in scene coordinates (objects.js LAY.T) */
const FR = sc => ({ P: (x, y, z) => TP(sc, x, y, z), V: (x, y, z) => TW(sc, x, y, z), KS: (a, b) => K(TP(sc, ...a), TP(sc, ...b)) });

/* ---------- shelf glyphs (created once; S02 b1 lights the four families) ---------- */
SHELF.mk('cc', 2, { label: `${H.cc}<div class="lbl">charmonium</div>` });
SHELF.mk('bb', 2, { label: `${H.bb}<div class="lbl">bottomonium</div>` });
SHELF.mk('Occc', 3, { label: `${H.Occc}<div class="lbl">triply charmed</div>` });
SHELF.mk('T4', 4, { label: `${H.T4}<div class="lbl">all-heavy tetraquark</div>`, tetra: true });
SHELF.layout('fam', ['cc', 'bb', 'Occc', 'T4'], []);
SHELF.place('fam', 'fam', 0);

/* ======================= S01 ======================= */
(function () {
  const SC = 0, DIST = 95, { P, V, KS } = FR(SC);
  const hero = W.glyph(3, { R: 5.2, r: 1.7, spin: 0.22, tilt: 0.7 }); hero.position.set(...P(-126, 88, 6)); W.scene.add(hero);
  const heroLab = BOARDS.board({ scene: SC, html: `<span class="lbl">colour singlet: schematic</span>`, px: 300, w: U(300, DIST), at: P(-126, 80, 6), a: 1 });
  heroLab.el.style.textAlign = 'center';
  const title = BOARDS.board({ scene: SC, px: 1000, w: U(1000, DIST), at: P(-172, 90, 0), a: 1, html: `
    <div class="t" style="font:400 48px/1.15 var(--serif)">Renormalon-subtracted nonrelativistic QCD<br>for heavy-hadron systems</div>
    <div class="t" style="font-size:27px;margin-top:22px">Benoît Assi · Andreas S. Kronfeld · <b>Simon Vaiva</b> · Michael L. Wagman</div>
    <div class="t small" style="margin-top:8px">Fermilab Theory Division &nbsp;·&nbsp; Aix-Marseille Univ, CPT &nbsp;·&nbsp; Univ. of Cincinnati</div>
    <div class="t small" style="margin-top:14px"><span class="chip pin">arXiv:2607.08817</span> <span class="chip dimc">FERMILAB-PUB-26-0061-T</span> <span class="chip dimc">July 2026</span></div>` });
  const qr = BOARDS.board({ scene: SC, px: 190, w: U(190, DIST), at: P(-100, 72, 0), a: 1, html: `<img src="assets/qr.png" style="width:150px;border-radius:6px;margin:0 auto"><div class="tiny" style="text-align:center;margin-top:4px">arxiv.org/abs/2607.08817</div>` });
  /* the meeting's logo (GdR QCD, Nantes, 7 to 9 October 2026; cropped from its Indico poster), top left over the title: the frame's one free corner in b1 and b2 */
  const gdr = BOARDS.board({ scene: SC, px: 240, w: U(240, DIST), at: P(-195, 109, 0), a: 1, html: `<img src="assets/gdr-qcd-2026.png" alt="GdR QCD 2026, Nantes" style="width:240px;display:block;border-radius:6px">` });
  /* the affiliations' logos, the three of the line under the authors (Fermilab: Wikimedia Commons "Fermilab Logo.svg"; CPT: the lab's logo with its name; Cincinnati: Commons "University of Cincinnati logo.png"), top right on a white card like the QR's */
  const labs = BOARDS.board({ scene: SC, px: 660, w: U(660, DIST), at: P(-115.5, 112.5, 0), a: 1, html: `<div style="display:flex;align-items:center;justify-content:space-between;background:#fff;padding:10px 18px;border-radius:6px"><img src="assets/fermilab.png" alt="Fermilab" style="height:46px"><img src="assets/cpt.png" alt="Centre de Physique Théorique" style="height:96px"><img src="assets/cincinnati.png" alt="University of Cincinnati" style="height:62px"></div>` });
  const thesis = BOARDS.board({ scene: SC, px: 900, w: U(900, DIST), at: P(-122, 64, 0), a: 1, html: `
    <div style="font-size:27px;font-style:italic;line-height:1.45"><span class="ti">A decades-old infrared obstruction: the <b>leading</b> renormalon</span> <span class="ti">→ removed</span> <span class="ti">→ a whole family of states becomes predictable.</span></div>` });
  const fam = ['cc', 'bb', 'Occc', 'T4'];
  const ROW = [[-190, 71], [-178, 71], [-166, 71], [-154, 71]].map(([x, y]) => V(x, y, 0));   /* where the family waits under the title (World) */
  const tparts = title.qa('.t'), th = thesis.qa('.ti');
  scenes.push({
    id: 'S01', title: 'Renormalon-subtracted nonrelativistic QCD for heavy-hadron systems', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      { d: 20, cam: [KS([-150, 88, DIST], [-150, 88, 0])], win: [0, .4], f(p) {          /* d: deck value (shim sum 1014); the World opens on this frame (engine CAM0), the title writes itself from p = .05 */
        hero.setOpacity(1); hero.setGlow(1); title.touch(); heroLab.touch(); qr.touch(); gdr.touch(); labs.touch();
        tparts.forEach((e, i) => rv(e, seg(p, .05 + i * .16, .3 + i * .16)));
        rv(heroLab.el.firstElementChild, seg(p, .1, .35)); qr.show(seg(p, .7, .95));   /* show, not rv: the board sets its own element's opacity every frame */
      } },
      { d: 24, f(p) {
        thesis.touch(); th.forEach((e, i) => rv(e, seg(p, i * .18, .22 + i * .18)));
        /* on "a whole family": four dim copies leave the hero and line up under the title; S02 b1 carries them to the shelf */
        const u = eo(seg(p, .5, 1));
        SHELF.place('fam', 'fam', 0);
        fam.forEach((id, i) => { const G = SHELF.glyphs[id];
          G.g.position.copy(hero.position).lerp(ROW[i], u).sub(SHELF.at);   /* glyph positions are the shelf's own: World minus its origin */
          G.g.setOpacity(0.35 * seg(p, .5, .7)); G.g.setGlow(0.5); G.lab.show(0); G.chip.show(0); });
        SHELF.showLines(0);
      } },
    ],
  });
  TICKS.push(t => { if (!hero.visible) return; hero.userData.live = liveOf('hero'); hero.tick(t); });   /* the title baryon turns while the talk is about it */
  window.__S01 = { hero, ROW };
})();

/* ======================= S02 ======================= */
(function () {
  const SC = 1, DIST = 108, { P, V, KS } = FR(SC);
  const hier = BOARDS.board({ scene: SC, cls: 'panel', px: 560, w: U(560, DIST), at: P(-44, 42, 0), a: 0, html: `
    <h2>Clean scale hierarchy</h2><div id="hb"></div><div class="small" style="margin-top:8px">⇒ controlled expansions in ${H.as} and ${I('v')}</div><div class="tiny">not to scale</div>` });
  const bars = [[`${H.mQ}`, 420, 'teal'], [`${H.mQ}${I('v')}`, 150, 'amber'], [`${H.mQ}${I('v')}²`, 52, 'rose']].map(x => {   /* the deck's scale colours; the S03 floors keep them */
    const row = D('div', 'row', hier.q('#hb')); D('div', 'm', row, x[0]).style.width = '64px';
    const bar = D('div', 'bar', row); bar.style.cssText = `height:16px;background:var(--${x[2]});width:0px`; return { bar, w: x[1] }; });
  hier.measure();
  const tri = BOARDS.board({ scene: SC, cls: 'panel', px: 500, w: U(500, DIST), at: P(0, 42, 0), a: 0, html: `<h2>Precision laboratory</h2><svg viewBox="0 0 440 170" width="440" height="170" id="tsv"></svg>` });
  const tsv = tri.q('#tsv');
  const tn = [[70, 130, 'EFT'], [370, 130, 'experiment'], [220, 28, 'lattice QCD']];
  const te = [[70, 130, 370, 130], [70, 130, 220, 28], [370, 130, 220, 28]].map(a => S('line', { x1: a[0], y1: a[1], x2: a[2], y2: a[3], stroke: 'var(--muted)', 'stroke-width': 1.5 }, tsv));
  tn.forEach(n => { S('circle', { cx: n[0], cy: n[1], r: 5, fill: 'var(--teal)' }, tsv); const t = S('text', { x: n[0], y: n[1] + (n[1] > 60 ? 26 : -14), 'text-anchor': 'middle' }, tsv); t.textContent = n[2]; });
  const tlab = S('text', { x: 220, y: 100, 'text-anchor': 'middle', class: 'it', opacity: 0, 'font-size': 20 }, tsv); tlab.textContent = 'compare at ~10 MeV';   /* Deck-only wording (S02-P3) */
  const tim = BOARDS.board({ scene: SC, cls: 'panel', px: 580, w: U(580, DIST), at: P(44, 42, 0), a: 0, html: `<h2>Timely</h2>
    <div><span class="chip t">lattice Ω${sub('QQQ')} at ±10–20 MeV</span> <span class="chip t rc">CMS 2025: &gt;5σ at t${ov('t')} threshold (§I)</span> <span class="chip t">ATLAS 2026: same excess</span></div>
    <svg viewBox="0 0 560 96" width="530" height="91" id="bsv"></svg>` });
  const bsv = tim.q('#bsv');
  S('line', { x1: 16, y1: 70, x2: 544, y2: 70, stroke: 'var(--muted)', 'stroke-width': 1.5 }, bsv);
  const bump = S('path', { d: '', fill: 'var(--rose)', 'fill-opacity': .3, stroke: 'var(--rose)', 'stroke-width': 2 }, bsv);
  S('line', { x1: 330, y1: 62, x2: 330, y2: 78, stroke: 'var(--muted)', 'stroke-width': 1.2 }, bsv);   /* the 2m_t tick; the peak sits left of it (S02-I1) */
  const bt = S('text', { x: 330, y: 92, 'text-anchor': 'middle', class: 'mt', 'font-size': 14 }, bsv); bt.innerHTML = `√s → 2<tspan class="it">m</tspan><tspan baseline-shift="sub" font-size="10">t</tspan>: toponium-like excess`;
  const bt2 = S('text', { x: 16, y: 24, class: 'mt it', 'font-size': 14 }, bsv); bt2.textContent = 'schematic: not CMS/ATLAS data';
  const bumpD = a => { let d = 'M16 70 '; for (let x = 16; x <= 544; x += 4) { const y = 70 - a * 44 * Math.exp(-Math.pow((x - 300) / 40, 2)); d += `L${x} ${y.toFixed(1)} `; } return d; };
  const timChips = tim.qa('.chip');
  tim.measure(); BOARDS.topAlign(hier, tri, tim);
  const GA = P(0, 21, 0);                                       /* the gauge's S02 place */
  /* the obstacle in words (round 24d: the scale-band board appears once, in S10; here its bars said what these lines say) */
  const obst = BOARDS.board({ scene: SC, cls: 'panel', px: 720, w: U(720, DIST), at: GA, a: 0, html: `<h2 class="rc">The obstacle</h2>
    <div class="t big" style="font-size:24px">target precision: <span class="num">~10 MeV</span></div>
    <div class="t big" style="font-size:24px">fixed-order scale dependence: <span class="num">~100 MeV</span> for quarkonium <span class="small">(§III A)</span>, <span class="num rc">~700–850 MeV</span> for the Ω masses at NNLO <span class="small">(Tab. III)</span></div>
    <div class="t small" style="margin-top:6px">This talk: remove the leading renormalon.</div>` });
  scenes.push({
    id: 'S02', title: 'Why all-heavy hadrons?', el: BOARDS.sceneEl(SC),
    notes: NOTES[SC],
    beats: [
      { d: 16, cam: [KS([0, 62, DIST], [0, 62, 0])], f(p) {
        const hero = window.__S01.hero; hero.setOpacity(1 - seg(p, 0, .5));
        SHELF.place('fam', 'fam', 0); SHELF.showLines(seg(p, .15, .55));
        const fly = eio(seg(p, 0, .6)), ROW = window.__S01.ROW;   /* the family travels from the title's place to the shelf, with the camera */
        ['cc', 'bb', 'Occc', 'T4'].forEach((id, i) => { const G = SHELF.glyphs[id]; const k = seg(p, .45 + i * .1, .65 + i * .1);
          const t = G.g.position, r = ROW[i].clone().sub(SHELF.at); t.set(lerp(r.x, t.x, fly), lerp(r.y, t.y, fly), t.z);
          G.g.setOpacity(0.35 + 0.65 * k); G.g.setGlow(0.5 + 0.5 * k); G.lab.show(k); });
        SHELF.endLabel.show(seg(p, .8, 1));
      } },
      { d: 13, f(p) { hier.show(seg(p, 0, .2)); bars.forEach((r, i) => r.bar.style.width = (r.w * eo(seg(p, .15 + i * .22, .5 + i * .22))) + 'px'); } },
      { d: 13, f(p) { tri.show(seg(p, 0, .2)); te.forEach((e, i) => dl(e, seg(p, .15 + i * .18, .5 + i * .18))); tlab.setAttribute('opacity', seg(p, .75, 1)); } },
      { d: 18, f(p) { tim.show(seg(p, 0, .25)); timChips.forEach((c, i) => rv(c, seg(p, .15 + i * .12, .35 + i * .12))); bump.setAttribute('d', bumpD(eo(seg(p, .45, .9)))); } },
      { d: 17, cam: [KS([0, 46, DIST], [0, 46, 0])], f(p) {
        obst.set(...GA); obst.show(seg(p, 0, .2)); obst.qa('.t').forEach((e, i) => rv(e, seg(p, .15 + i * .2, .35 + i * .2)));
      } },
    ],
  });
  window.__S02 = { hier, bars, GA };
})();
