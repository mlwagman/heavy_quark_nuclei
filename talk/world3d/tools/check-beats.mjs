// check-beats.mjs: what three reworked beats must show (round 11). Written before the change; it failed on the r10 build.
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-beats.mjs
import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH + '/');
const { chromium } = require('playwright');
import { installDeckShim } from '/home/gofer/mrs_talk_video/render/deck_shim.mjs';
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
const page = await b.newPage({ viewport: { width: 1920, height: 1080 } });
const errs = []; page.on('pageerror', e => errs.push(String(e)));
await page.goto(new URL('../index.html', import.meta.url).href); await installDeckShim(page); await page.evaluate(() => document.fonts.ready);
/* round 18 cut beats so that no text is written and gone inside one beat. The tests keep naming the r17 beats: at() maps an r17
   (scene, beat, p) to the beat it now falls in. CUT: r17 beat -> the durations of the beats it became (they sum to its d) */
const CUT = { 2: { 1: [6, 6.5, 5.5], 3: [3, 14] }, 4: { 2: [10, 7, 7, 6] }, 5: { 2: [4, 14] }, 7: { 3: [14.5, 5.5] }, 8: { 3: [6, 7] }, 9: { 2: [9, 9] },
  11: { 1: [12.5, 5.5], 3: [13, 13] }, 12: { 1: [13, 13] }, 13: { 4: [10, 10] }, 14: { 1: [3, 11] } };
const remap = (s, k, p) => { const C = CUT[s] || {}; let k2 = k; for (const j in C) if (+j < k) k2 += C[j].length - 1;
  const ds = C[k]; if (!ds) return [s, k2, p]; if (p == null) return [s, k2 + ds.length - 1, null];
  const D = ds.reduce((x, y) => x + y); let a = 0; for (let i = 0; i < ds.length; i++) { const c0 = a / D, c1 = (a += ds[i]) / D; if (p <= c1 || i === ds.length - 1) return [s, k2 + i, (p - c0) / (c1 - c0)]; } };
const at = async (s0, k0, p0) => { const [s, k, p] = remap(s0, k0, p0); await page.evaluate(([s, k, p]) => { window.__deck.show(s); if (p == null) window.__deck.applyUpTo(k); else { window.__deck.applyUpTo(k - 1); applyState(s, k - 1, p); } }, [s, k, p]);
  await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))); };
const R = []; const ok = (name, pass, got) => { R.push(pass); console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}  (${got})`); };
const sc = () => page.evaluate(() => { const e = SHELF.glyphs.bbtt.lab.el; const r = e.getBoundingClientRect(); return { s: +(/scale\(([\d.]+)\)/.exec(e.style.transform) || [0, 0])[1], vis: e.style.visibility, op: +e.style.opacity, in: r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080 }; });

/* 1. S12 b3: while the tetraquarks join the top row and the unbound ones fall apart, the camera is close on the shelf */
await at(11, 3, 0.4); let v = await sc();
ok('S12 b3 p=.40: bound-state label on stage at scale >= 0.60', v.vis === 'visible' && v.op > 0.5 && v.in && v.s >= 0.6, `scale ${v.s.toFixed(2)}, opacity ${v.op}`);
v = await page.evaluate(() => { const g = SHELF.glyphs.bbbb.g; const P = W.project(SHELF.wp('bbbb'));   /* World position: the shelf moves since round 18 */ return { op: g.userData.op, split: g.userData.split, x: P.x, y: P.y, ppu: W.ppu(P.depth) }; });
await at(11, 3, 0.5); v = await page.evaluate(() => { const g = SHELF.glyphs.bbbb.g; const P = W.project(SHELF.wp('bbbb'));   /* World position: the shelf moves since round 18 */ return { op: g.userData.op || 0, split: g.userData.split || 0, x: P.x, y: P.y, ppu: W.ppu(P.depth) }; });
ok('S12 b3 p=.50: the unbound state is on stage, splitting, at >= 12 px per unit', v.op > 0.3 && v.split > 0.3 && v.x > 0 && v.x < 1920 && v.y > 0 && v.y < 1080 && v.ppu >= 12, `opacity ${v.op.toFixed(2)}, split ${v.split.toFixed(2)}, ${v.ppu.toFixed(1)} px/unit`);
await at(11, 3, null); v = await page.evaluate(() => BOARDS.all.filter(b => /^(BOUND|UNBOUND)/.test(b.el.textContent.trim())).map(b => { const r = b.el.getBoundingClientRect(); return b.el.style.visibility === 'visible' && +b.el.style.opacity > 0.8 && r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080; }));
ok('S12 b3 end: both verdict boards whole on stage', v.length === 2 && v.every(Boolean), JSON.stringify(v));

/* 2. S10 b3: after the collapse the ~700 MeV width is still drawn, as an outline */
await at(9, 3, null); v = await page.evaluate(() => { const r = GAUGE.rows[0], o = r.bar.getBoundingClientRect(), f = r.bar.firstElementChild, v = r.val.getBoundingClientRect(), m = GAUGE.rows[1].bar.getBoundingClientRect();
  return { outer: o.width, fill: f ? f.getBoundingClientRect().width : -1, dashed: getComputedStyle(r.bar).borderTopStyle, valAfter: v.left >= o.right - 1, mrs: m.width }; });
ok('S10 b3 end: fixed-order row keeps a dashed outline >= 40x the MRS sliver, fill collapsed, value right of the outline', v.dashed === 'dashed' && v.fill >= 0 && v.fill <= v.mrs * 2.5 && v.outer >= 40 * v.mrs && v.valAfter, `outline ${v.outer.toFixed(0)} px, fill ${v.fill.toFixed(0)} px, MRS ${v.mrs.toFixed(1)} px, border ${v.dashed}`);
await at(9, 3, 0); v = await page.evaluate(() => { const r = GAUGE.rows[0], o = r.bar.getBoundingClientRect(), f = r.bar.firstElementChild; return { outer: o.width, fill: f ? f.getBoundingClientRect().width : -1 }; });
ok('S10 b3 start: the hatched bar fills its outline', v.fill > 0 && v.fill >= v.outer - 6, `outline ${v.outer.toFixed(0)} px, fill ${v.fill.toFixed(0)} px`);
await at(3, 3, null); v = await page.evaluate(() => GAUGE.rows.slice(0, 2).map(r => ({ kids: r.bar.childElementCount, st: getComputedStyle(r.bar).borderTopStyle })));
ok('S04 b3 (another user of the gauge): no outline, no inner fill', v.every(x => x.kids === 0 && x.st !== 'dashed'), JSON.stringify(v));

/* 3. S07: once the camera has arrived, the S06 boards are gone from the middle of the stage */
await at(6, 1, 0.3); v = await page.evaluate(() => BOARDS.all.filter(b => b.scene === 5 && b.el.style.visibility === 'visible').length);
ok('S07 b1 p=.30: S06 boards still ghosted during the pan', v > 0, `${v} visible`);
await at(6, 2, null); v = await page.evaluate(() => BOARDS.all.filter(b => b.scene === 5 && b.el.style.visibility === 'visible').length);
ok('S07 b2 end: no S06 board visible', v === 0, `${v} visible`);

/* 4. the tower after S03 (round 12): it stays, quieter, and its names are whole wherever they show */
const tower = () => page.evaluate(() => TOWER.floors.map(f => { const e = f.lab.el, r = e.getBoundingClientRect(), vis = e.style.visibility === 'visible' && +e.style.opacity > 0.3;
  const out = r.right < 0 || r.left > 1920 || r.bottom < 0 || r.top > 1080, whole = r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080;
  return { name: e.textContent.trim().slice(0, 6), vis, whole, out, edge: +f.edge.material.opacity.toFixed(2), z: f.lab.anchor.z }; }));
await at(2, 2, null); v = await tower();
ok('S03 b2 end: the tower is as in S03 (outlines 0.7, names on the front edge)', v.every(f => f.edge === 0.7 && f.z === 10), JSON.stringify(v.map(f => [f.edge, f.z])));
/* round 18: the tower stays in S03's place and S06 no longer plays on its pNRQCD floor, so the S06 frames need not show its names */
for (const k of [1, 2, 3, 4]) { await at(5, k, null); v = await tower();
  ok(`S06 b${k} end: no tower name cut by the stage edge, outlines kept`, v.every(f => (!f.vis || f.whole || f.out) && f.edge > 0.1), JSON.stringify(v.map(f => [f.name, f.vis ? (f.whole ? 'whole' : f.out ? 'off stage' : 'CUT') : 'not shown', f.edge]))); }
for (const k of [1, 2, 3, 4]) { await at(6, k, null); v = await tower();
  ok(`S07 b${k} end: no tower name cut by the stage edge, outlines kept`, v.every(f => (!f.vis || f.whole || f.out) && f.edge > 0.1), JSON.stringify(v.map(f => [f.name, f.vis ? (f.whole ? 'whole' : f.out ? 'off stage' : 'CUT') : 'not shown', f.edge]))); }

for (const [sc, k, pp, nm] of [[2, 3, null, 'S03 b3 end'], [2, 4, null, 'S03 b4 end'], [13, 4, 0.5, 'S14 b4 p=.50']]) { await at(sc, k, pp); v = await tower();
  ok(`${nm}: no tower name cut by the stage edge`, v.every(f => !f.vis || f.whole || f.out), JSON.stringify(v.map(f => [f.name, f.vis ? (f.whole ? 'whole' : f.out ? 'off stage' : 'CUT') : 'not shown']))); }

/* 5. round 13: the renormalon (amber) and remainder (teal) dots exist only from the S05 b1 split on, and the split is a motion;
      no jump on the click into S05 b1 (branch point) or S09 b3 (number line) */
const dots = () => page.evaluate(() => { const c = [...TERMS.b.el.querySelectorAll('circle')], n = c.length / 3; const seen = e => +e.getAttribute('r') > 0.2 && e.getAttribute('opacity') !== '0';
  return { amber: c.slice(n, 2 * n).filter(seen).length, teal: c.slice(2 * n).filter(seen).length, r0: +c[n].getAttribute('r'), cy0: +c[n].getAttribute('cy') }; });
for (const k of [2, 3, 4]) { await at(3, k, null); v = await dots(); ok(`S04 b${k} end: only the plain term dots on the |term| plot`, v.amber === 0 && v.teal === 0, `${v.amber} amber, ${v.teal} teal`); }
await at(4, 1, 0.4); v = await dots(); ok('S05 b1 p=.40: the order-0 amber dot is on its way out of the term dot (between y 85 and 169, not yet full size)', v.cy0 > 90 && v.cy0 < 164 && v.r0 < 6.4, `cy ${v.cy0.toFixed(0)}, r ${v.r0.toFixed(1)}`);
await at(4, 1, null); v = await dots(); ok('S05 b1 end: 10 amber and 3 teal dots', v.amber === 10 && v.teal === 3, `${v.amber} amber, ${v.teal} teal`);
await at(3, 4, null); const bp0 = await page.evaluate(() => BOREL.circ.material.opacity); await at(4, 1, 0); const bp1 = await page.evaluate(() => BOREL.circ.material.opacity);
ok('S05 b1 p=0: the branch point is as S04 b4 left it', Math.abs(bp0 - bp1) < 0.02, `${bp0.toFixed(2)} → ${bp1.toFixed(2)}`);
await at(8, 2, null); const nl0 = await page.evaluate(() => NLINE.b.q('#nt').textContent.slice(0, 12)); await at(8, 3, 0); const nl1 = await page.evaluate(() => NLINE.b.q('#nt').textContent.slice(0, 12));
await at(8, 3, null); const nl2 = await page.evaluate(() => NLINE.b.q('#nt').textContent.slice(0, 12));
ok('S09 b3 p=0: the number line still shows what b2 left; by the end it shows B_c', nl0 === nl1 && nl2.startsWith('B'), `"${nl0}" → "${nl1}" … "${nl2}"`);

/* 6. round 14: smooth zoom-pan camera, main boards that do not dim, the emptied |term| board leaves */
const cams = await page.evaluate(() => beatList.map((e, idx) => { const b = e.b; if (!b.cam) return null; const dur = b.d / SPEED * (b.slow || 1); let prev = camAt(idx, 0), pk = 0; const N = 400;
  for (let i = 1; i <= N; i++) { const c = camAt(idx, i / N), wdt = 2 * c.pos.distanceTo(c.look) * Math.tan(38 * Math.PI / 360) * 16 / 9; pk = Math.max(pk, Math.hypot(c.look.x - prev.look.x, c.look.y - prev.look.y) / wdt * N / dur); prev = c; }
  return { id: scenes[e.s].id + ' b' + (e.k + 1), pk }; }).filter(Boolean));
const fast = cams.filter(c => c.pk > 1);
ok('camera: no move pans faster than 1.0 stage widths per second', fast.length === 0, fast.length ? fast.map(c => `${c.id} ${c.pk.toFixed(2)}`).join(', ') : `fastest ${cams.reduce((a, c) => c.pk > a.pk ? c : a).id} ${Math.max(...cams.map(c => c.pk)).toFixed(2)}`);
const dists = (s, k, ps) => page.evaluate(([s, k, ps]) => { const idx = beatList.findIndex(e => e.s === s && e.k === k); return ps.map(p => { const c = camAt(idx, p); return c.pos.distanceTo(c.look); }); }, [s, k, ps]);
v = await dists(1, 0, [0, 0.3, 1]);
ok('S02 b1 (a 1.2-width move): half-way the camera is further back than at either end by 15 % or more', v[1] >= 1.15 * Math.max(v[0], v[2]), v.map(x => x.toFixed(0)).join(' → '));
v = await dists(3, 2, [0, 0.3, 1]);
ok('S04 b3 (a short move): no pull-back, the distance stays between its ends', v[1] <= Math.max(v[0], v[2]) * 1.01 && v[1] >= Math.min(v[0], v[2]) * 0.99, v.map(x => x.toFixed(1)).join(' → '));
const opOf = re => page.evaluate(src => { const b = BOARDS.all.find(b => new RegExp(src).test(b.el.textContent)); return b && b.el.style.visibility === 'visible' ? +b.el.style.opacity : 0; }, re);
for (const [sc, k, re, nm] of [[4, 4, '^\\s*𝒱', 'S05 b4: the closed form'], [4, 5, '^\\s*𝒱', 'S05 b5: the closed form'], [12, 2, '^diagnostic', 'S13 b2: the table'], [12, 3, '^diagnostic', 'S13 b3: the table']]) {
  await at(sc, k, null); v = await opOf(re); ok(`${nm} stays at full strength`, v >= 0.95, `opacity ${v.toFixed(2)}`); }
await at(5, 1, null); v = await opOf('^\\s*𝒱'); ok('S06 b1: the closed form is only a ghost once its scene is over', v <= 0.12, `opacity ${v.toFixed(2)}`);
await at(4, 2, null); const tb2 = await page.evaluate(() => TERMS.b.a); await at(4, 3, 0.3); const tb3m = await page.evaluate(() => TERMS.b.a); await at(4, 3, null); const tb3 = await page.evaluate(() => TERMS.b.a);
ok('S05 b3: the |term| board is there while the haze leaves it, and gone by the end of the beat', tb2 > 0.9 && tb3m > 0.9 && tb3 < 0.01, `b2 end ${tb2}, b3 p=.3 ${tb3m}, b3 end ${tb3}`);

/* 7. round 15: the overview is a menu, the number line re-zooms, the S13 rows carry bars, closer frames for the beats whose
      text was small, dim boards fade at the stage edge instead of being cut */
const wait = ms => page.waitForTimeout(ms);
const state = () => page.evaluate(() => ({ si, done, n: scenes[si].beats.length, ovOn: OV.on, ovBeat: OV.beat, kk: OV.kk, off: W.cam.pos.distanceTo(beatCam.pos) }));
await at(13, 4, null); await wait(300); await page.evaluate(() => scenes[8].badge.el.click()); await wait(200); v = await state();
ok('overview (S14 b4): a click on badge 9 opens scene 9, complete, and leaves the overview', v.si === 8 && v.done === v.n && !v.ovOn && v.ovBeat === 0, JSON.stringify({ scene: v.si + 1, done: v.done, of: v.n }));
ok('… without a camera jump: just after the click the camera is still far from the scene', v.kk > 0.5 && v.off > 100, `overview blend ${v.kk.toFixed(2)}, camera ${v.off.toFixed(0)} units from its beat`);
await wait(2400); v = await state(); ok('… and 2.6 s later it has landed', v.off < 1.5 && v.kk < 0.02, `camera ${v.off.toFixed(1)} units from its beat`);
await at(13, 4, null); await wait(300); await page.evaluate(() => scenes[10].lg.click()); await wait(200); v = await state();
ok('overview: a click on a legend line does the same (line 11)', v.si === 10 && v.done === v.n, JSON.stringify({ scene: v.si + 1, done: v.done }));
await at(13, 4, null); await wait(300); await page.mouse.click(960, 1060); await wait(200); v = await state();
ok('overview: a click anywhere else still goes on to the next beat (S15 b1)', v.si === 14, JSON.stringify({ scene: v.si + 1 }));
await at(6, 2, null); await page.keyboard.press('o'); await wait(2400); await page.keyboard.press('1'); await page.keyboard.press('2'); await wait(1200); v = await state();
ok('overview: typing 12 opens scene 12', v.si === 11 && v.done === v.n && !v.ovOn, JSON.stringify({ scene: v.si + 1, done: v.done }));
const ovOff = () => page.evaluate(() => { OV.on = false; OV.k = 0; });   /* whatever the tests above left, the rest runs outside the overview */
await ovOff(); await at(6, 2, null); await page.keyboard.press('3'); await wait(1000); v = await state();
ok('outside the overview a digit does nothing', v.si === 6 && v.done === 2, JSON.stringify({ scene: v.si + 1, done: v.done }));

await ovOff();
/* half-way through the zoom: p = .245 since round 18 (the zoom ends at .44, so the chips stop on arrival) */
const ruler = () => page.evaluate(() => { const sv = NLINE.b.el.querySelector('svg'); return { a: NLINE.b.a, ticks: sv.querySelectorAll('.tk').length, w: +sv.dataset.hi - +sv.dataset.lo, lo: +sv.dataset.lo, hi: +sv.dataset.hi, title: NLINE.b.q('#nt').textContent.slice(0, 8) }; });
await at(8, 2, null); const r0 = await ruler(); await at(8, 3, 0.245); const r1 = await ruler(); await at(8, 3, null); const r2 = await ruler();
ok('S09 b3: the number line stays on and re-zooms from the η_t window to the B_c window: half-way it is wider than both and carries ticks',
  r0.a > 0.99 && r1.a > 0.99 && r2.a > 0.99 && r1.ticks >= 4 && r1.w > 10 * r0.w && r1.w > 10 * r2.w && Math.abs(r0.w - 3) < 0.01 && Math.abs(r2.w - 0.062) < 0.001 && r2.title.startsWith('B'),
  `windows [${r0.lo}, ${r0.hi}] → [${r1.lo.toFixed(0)}, ${r1.hi.toFixed(0)}] with ${r1.ticks} ticks → [${r2.lo}, ${r2.hi}]`);

await at(12, 1, null); v = await page.evaluate(() => { const tb = BOARDS.all.find(b => /^diagnostic/.test(b.el.textContent)); const W_ = id => { const e = tb.el.querySelector(id); return e ? e.getBoundingClientRect().width / (tb.el.getBoundingClientRect().width / tb.px) : -1; };
  return { text: tb.el.textContent, fo: [0, 1, 2, 3].map(i => W_('#fb' + i)), mrs: [0, 1, 2, 3].map(i => W_('#mb' + i)) }; });
const TBTEXT = 'diagnostic (c, b)fixed orderMRSscale band, Ωbbb at NNLO~700 MeV→~10 MeVorder-by-order shifts (baryons, |NNLO−NLO|)33–55 MeV→4–18 MeVtuned mc, mb drift4–11%→< 1%LO / NLO / NNLO bandsLO apart→overlap (partial)';
const want = [10 / 700, 18 / 55, 1 / 11];
ok('S13 b1 end: rows 1 to 3 carry a fixed-order bar and an MRS bar in the ratio of the numbers beside them; row 4 has none; no word of the table changed',
  v.text === TBTEXT && want.every((r, i) => v.fo[i] > 60 && Math.abs(v.mrs[i] / v.fo[i] - r) < 0.02) && v.fo[3] < 0 && v.mrs[3] < 0,
  `text ${v.text === TBTEXT ? 'unchanged' : 'CHANGED'}; MRS / fixed order = ${v.mrs.slice(0, 3).map((m, i) => (m / v.fo[i]).toFixed(3)).join(', ')} (want ${want.map(x => x.toFixed(3)).join(', ')})`);
await at(12, 1, 0.3); v = await page.evaluate(() => { const tb = BOARDS.all.find(b => /^diagnostic/.test(b.el.textContent)); return [0, 1, 2].map(i => { const e = tb.el.querySelector('#mb' + i); return e ? e.getBoundingClientRect().width : -1; }); });
ok('S13 b1 p=.30: no bar before its row is there', v.every(w => w >= 0 && w < 0.5), v.map(w => w.toFixed(1)).join(', '));

const scaleOf = re => page.evaluate(src => { const b = BOARDS.all.find(b => new RegExp(src).test(b.el.textContent) && b.el.style.visibility === 'visible'); if (!b) return { s: 0, whole: false };
  const r = b.el.getBoundingClientRect(); return { s: +(/scale\(([\d.]+)\)/.exec(b.el.style.transform) || [0, 0])[1], whole: r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080 }; }, re);
for (const [sc, k, pp, re, min, nm] of [[11, 1, null, '^A fourth body', 1.0, 'S12 b1: the ratio R board'], [11, 2, null, '^Binding ratio', 1.0, 'S12 b2: the figure'], [11, 4, null, '^Colour structure', 1.0, 'S12 b4: the colour board'],
  [7, 4, 0.5, '= 1\\.555 GeV', 1.1, 'S08 b4 p=.50: the tuned charm mass'], [7, 4, null, '^Everything after this', 0.9, 'S08 b4 end: the banner']]) {
  await at(sc, k, pp); v = await scaleOf(re); ok(`${nm} is whole on the stage at scale ${min} or more`, v.whole && v.s >= min, `scale ${v.s.toFixed(2)}${v.whole ? '' : ', not whole'}`); }

await at(6, 3, null); v = await page.evaluate(() => { const o = re => { const b = BOARDS.all.find(b => re.test(b.el.textContent)); return b.el.style.visibility === 'visible' ? +b.el.style.opacity : 0; }; return { vmc: o(/^VMC: variational/), gfmc: o(/^GFMC: imaginary/) }; });
ok('S07 b3: the dim VMC board, which the top edge would cut, has faded; the dim GFMC board inside the stage has not', v.vmc <= 0.05 && v.gfmc > 0.38 && v.gfmc < 0.46, `VMC ${v.vmc.toFixed(2)}, GFMC ${v.gfmc.toFixed(2)}`);

/* S10 b5 (round 17): the beat ends on numbers that can be read. Every number chip on the stage and the summary board are drawn
   with letters 18 px high or more (1080p stage), and the six top-quark masses and the summary are whole on the stage */
await at(9, 5, null); v = await page.evaluate(() => { const px = e => parseFloat(getComputedStyle(e).fontSize) * e.getBoundingClientRect().height / e.offsetHeight, inside = r => r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080;
  const chips = []; for (const b of BOARDS.all) { if (b.el.style.visibility !== 'visible' || +b.el.style.opacity < 0.3) continue; for (const c of b.el.querySelectorAll('.chip')) { const r = c.getBoundingClientRect(); if (r.right < 0 || r.left > 1920 || r.bottom < 0 || r.top > 1080) continue; chips.push({ t: c.textContent.trim(), px: px(c), whole: inside(r) }); } }
  const top = ['175.55827(6)(648)', '178.76384(5)(995)', '181.85756(8)(1444)', '346.44124(7)(1273)', '349.33347(34)(4681)', '514.46(4)(94)'].map(t => chips.find(c => c.t === t));
  const sb = BOARDS.all.find(b => /^Top-quark benchmarks/.test(b.el.textContent.trim())), big = sb.q('.big');
  return { n: chips.length, min: Math.min(...chips.map(c => c.px)), small: chips.filter(c => c.px < 18).length, top: top.filter(c => c && c.whole).length, sum: px(big), sumWhole: sb.el.style.visibility === 'visible' && +sb.el.style.opacity > 0.8 && inside(sb.el.getBoundingClientRect()) }; });
ok('S10 b5 end: every number chip on the stage is 18 px or more, the six top-quark masses and the summary are whole, the summary 20 px or more', v.n >= 17 && v.small === 0 && v.top === 6 && v.sumWhole && v.sum >= 20, `${v.n} chips, smallest ${v.min.toFixed(1)} px, ${v.small} under 18 px, ${v.top} of 6 top masses whole, summary ${v.sum.toFixed(1)} px${v.sumWhole ? '' : ', not whole'}`);

console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
await b.close(); process.exit(R.every(Boolean) && !errs.length ? 0 : 1);
