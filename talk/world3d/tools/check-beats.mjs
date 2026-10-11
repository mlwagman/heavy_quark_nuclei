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
const CUT = { 2: { 1: [6, 6.5, 5.5], 3: [3, 14] }, 3: { 4: [6, 5, 5, 4] }, 4: { 2: [10, 7, 3, 4, 6] }, 5: { 2: [4, 14] }, 7: { 3: [14.5, 5.5] }, 8: { 3: [6, 7] }, 9: { 2: [9, 9] },
  11: { 1: [12.5, 5.5], 3: [13, 13] }, 12: { 1: [13, 13] }, 13: { 1: [3, 11] } };   /* round 24e: S14 is the references (the old S15) */
const remap = (s, k, p) => { const C = CUT[s] || {}; let k2 = k + (s === 12 ? 1 : 0);   /* round 24e: S13 opens on the conclusion claim, one beat before the old S13 */ for (const j in C) if (+j < k) k2 += C[j].length - 1;
  const ds = C[k]; if (!ds) return [s, k2, p]; if (p == null) return [s, k2 + ds.length - 1, null];
  const D = ds.reduce((x, y) => x + y); let a = 0; for (let i = 0; i < ds.length; i++) { const c0 = a / D, c1 = (a += ds[i]) / D; if (p <= c1 || i === ds.length - 1) return [s, k2 + i, (p - c0) / (c1 - c0)]; } };
const raw = async (s, k, p) => { await page.evaluate(([s, k, p]) => { window.__deck.show(s); if (p == null) window.__deck.applyUpTo(k); else { window.__deck.applyUpTo(k - 1); applyState(s, k - 1, p); } }, [s, k, p]);
  await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))); };
const at = (s0, k0, p0) => raw(...remap(s0, k0, p0));   /* at: an r17 beat name; raw: the build's own beat numbers */
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
/* round 25: the panel counter. On every panel the stage shows "N / 14" in the panel's section colour (the colours of the map's
   badges), bottom left, whole, on no board; on the map (S13 b8) it is hidden: the badges carry the numbers there */
{ const probe = async (s, k) => { await raw(s, k, null); return page.evaluate(i => { const e = document.getElementById('panel'); if (!e) return null; const r = e.getBoundingClientRect(), cs = getComputedStyle(e);
    const want = (() => { const d = document.createElement('div'); d.style.background = SECC[SEC(i)]; document.body.appendChild(d); const c = getComputedStyle(d).backgroundColor; d.remove(); return c; })();
    const vis = b => b.el.style.visibility === 'visible' && +b.el.style.opacity > 0.3;
    return { text: e.textContent.trim(), op: +cs.opacity, bg: cs.backgroundColor, want, in: r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080 && r.width > 40 && r.height > 20,
      hit: BOARDS.all.filter(vis).filter(b => { const q = b.el.getBoundingClientRect(); return Math.min(r.right, q.right) - Math.max(r.left, q.left) > 0 && Math.min(r.bottom, q.bottom) - Math.max(r.top, q.top) > 0; }).map(b => b.el.textContent.trim().slice(0, 24)) }; }, s); };
  const bad = [];
  for (let s = 0; s < 14; s++) { const k = await page.evaluate(i => scenes[i].beats.length, s); const v = await probe(s, s === 12 ? 7 : k);   /* S13: before the map lifts off */
    if (!v || v.text !== `${s + 1} / 14` || v.op < 0.9 || v.bg !== v.want || !v.in || v.hit.length) bad.push(`S${s + 1}: ${JSON.stringify(v)}`); }
  ok('every panel shows its counter "N / 14" in its section colour, whole on the stage, on no board', bad.length === 0, bad.slice(0, 2).join(' | ') || 'all 14');
  v = await probe(12, 8); ok('S13 b8 (the map): the counter is hidden', !!v && v.op < 0.1, v ? `opacity ${v.op}` : 'no counter'); }

/* round 24e: one conclusion panel. S13 "Conclusions" (8 beats): the claim, the table (2 stops), hallmarks, payoff, the family,
   next + thank you, the map; S14 the references. The old S14's "scale sensitivity down by two orders" board is gone (row 1 says it) */
{ v = await page.evaluate(() => ({ n: scenes.length, ids: scenes.map(s => s.id).slice(11).join(), t13: scenes[12].title, b13: scenes[12].beats.length, t14: (scenes[13] || {}).title,
    dup: BOARDS.all.some(b => /Scale sensitivity of charm and bottom hadrons down/.test(b.el.textContent)) }));
  ok('one conclusion panel: 14 scenes, S13 "Conclusions" with 8 beats, S14 the references, no repeated "two orders" board', v.n === 14 && v.ids === 'S12,S13,S14' && v.t13 === 'Conclusions' && v.b13 === 8 && v.t14 === 'Key references' && !v.dup, JSON.stringify(v));
  if (v.n === 14) { await raw(12, 1, null); v = await page.evaluate(() => BOARDS.all.filter(b => b.el.style.visibility === 'visible' && +b.el.style.opacity > 0.8 && /First application of MRS inside a bound-state/.test(b.el.textContent)).map(b => { const r = b.el.getBoundingClientRect(); return r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080; }));
    ok('S13 b1 end: "To conclude": the first-application claim is on stage, whole', v.length === 1 && v[0], JSON.stringify(v)); } }

/* round 24d: the scale-band board appears once, in S10 (the author: "this plot appears too often"); S02 b5 and S04 b3 keep their
   words as plain text, S13 starts on its table */
{ const seen = await page.evaluate(async () => { const res = []; for (let s = 0; s < scenes.length; s++) for (let k = 1; k <= scenes[s].beats.length; k++) for (const p of [0.5, null]) {
    window.__deck.show(s); if (p == null) window.__deck.applyUpTo(k); else { window.__deck.applyUpTo(k - 1); applyState(s, k - 1, p); }
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    const e = GAUGE.b.el, r = e.getBoundingClientRect(); if (e.style.visibility === 'visible' && +e.style.opacity > 0.3 && r.right > 0 && r.left < 1920 && r.bottom > 0 && r.top < 1080) res.push(scenes[s].id); } return [...new Set(res)]; });
  ok('the scale-band board is on screen in S10 only', seen.join() === 'S10', JSON.stringify(seen));
  const words = async (s, k, re) => { await raw(s, k, null); return page.evaluate(src => BOARDS.all.some(b => b.el.style.visibility === 'visible' && +b.el.style.opacity > 0.5 && b !== GAUGE.b && new RegExp(src).test(b.el.textContent)), re); };
  ok('S02 b5 end: the obstacle is still said, in words (~10 MeV target, ~700–850 MeV at NNLO)', await words(1, 5, 'obstacle[\\s\\S]*10 MeV[\\s\\S]*700–850 MeV'), 'text board');
  ok('S04 b3 end: the symptom is still said, in words (~700 MeV at NNLO)', await words(3, 3, 'Symptom[\\s\\S]*700 MeV'), 'text board'); }
  await raw(12, 2, null); v = await page.evaluate(() => { const T = BOARDS.all.find(b => b.scene === 12 && b.el.querySelector('#r0')); const op = id => { const e = T.el.querySelector(id); return e ? +getComputedStyle(e).opacity : 0; };
    return { a: op('#a0'), r: op('#r0'), c: op('#c0'), r1: op('#r1'), txt: T.el.querySelector('#r0').textContent + ' → ' + T.el.querySelector('#c0').textContent }; });
  ok('S13 b2 end (the table\'s first stop): the first row (the scale band, ~700 MeV → ~10 MeV) is written, the second not yet: the narration names it', v.a > 0.9 && v.r > 0.9 && v.c > 0.9 && v.r1 < 0.1, JSON.stringify(v));

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
for (const [sc, k, re, nm] of [[4, 4, '^\\s*𝒱', 'S05 b8 (r17 b4): the equation board'], [4, 5, '^\\s*𝒱', 'S05 b9 (r17 b5): the equation board'], [12, 2, '^diagnostic', 'S13 b2: the table'], [12, 3, '^diagnostic', 'S13 b3: the table']]) {
  await at(sc, k, null); v = await opOf(re); ok(`${nm} stays at full strength`, v >= 0.95, `opacity ${v.toFixed(2)}`); }
await at(5, 1, null); v = await opOf('^\\s*𝒱'); ok('S06 b1: the closed form is only a ghost once its scene is over', v <= 0.12, `opacity ${v.toFixed(2)}`);
/* round 19: the haze no longer rises in the |term| graph, so that board leaves once its dots have (section 8), not after the haze */

/* 7. round 15: the overview is a menu, the number line re-zooms, the S13 rows carry bars, closer frames for the beats whose
      text was small, dim boards fade at the stage edge instead of being cut */
const wait = ms => page.waitForTimeout(ms);
const state = () => page.evaluate(() => ({ si, done, n: scenes[si].beats.length, ovOn: OV.on, ovBeat: OV.beat, kk: OV.kk, off: W.cam.pos.distanceTo(beatCam.pos) }));
await raw(12, 8, null); await wait(300); await page.evaluate(() => scenes[8].badge.el.click()); await wait(200); v = await state();
ok('overview (S13 b8, the map): a click on badge 9 opens scene 9, complete, and leaves the overview', v.si === 8 && v.done === v.n && !v.ovOn && v.ovBeat === 0, JSON.stringify({ scene: v.si + 1, done: v.done, of: v.n }));
ok('… without a camera jump: just after the click the camera is still far from the scene', v.kk > 0.5 && v.off > 100, `overview blend ${v.kk.toFixed(2)}, camera ${v.off.toFixed(0)} units from its beat`);
await wait(2400); v = await state(); ok('… and 2.6 s later it has landed', v.off < 1.5 && v.kk < 0.02, `camera ${v.off.toFixed(1)} units from its beat`);
await raw(12, 8, null); await wait(300); await page.evaluate(() => scenes[10].lg.click()); await wait(200); v = await state();
ok('overview: a click on a legend line does the same (line 11)', v.si === 10 && v.done === v.n, JSON.stringify({ scene: v.si + 1, done: v.done }));
await raw(12, 8, null); await wait(300); await page.mouse.click(960, 1060); await wait(200); v = await state();
ok('overview: a click anywhere else still goes on to the next beat (S14 b1, the references)', v.si === 13, JSON.stringify({ scene: v.si + 1 }));
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
  r0.a > 0.99 && r1.a > 0.99 && r2.a > 0.99 && r1.ticks >= 4 && r1.w > 10 * r0.w && r1.w > 10 * r2.w && Math.abs(r0.w - 3) < 0.01 && Math.abs(r2.w - 0.072) < 0.001 && r2.title.startsWith('B'),
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
for (const [sc, k, pp, re, min, nm] of [[11, 1, null, '^A fourth body', 1.0, 'S12 b1: the ratio R board'], [11, 2, null, '^Binding ratio', 1.0, 'S12 b2: the figure'],   /* round 19: the colour-structure board and its beat are gone */
  [7, 4, 0.5, '= 1\\.555 GeV', 1.1, 'S08 b4 p=.50: the tuned charm mass'], [7, 4, null, '^Everything after this', 0.9, 'S08 b4 end: the banner']]) {
  await at(sc, k, pp); v = await scaleOf(re); ok(`${nm} is whole on the stage at scale ${min} or more`, v.whole && v.s >= min, `scale ${v.s.toFixed(2)}${v.whole ? '' : ', not whole'}`); }

await at(6, 3, null); v = await page.evaluate(() => { const o = re => { const b = BOARDS.all.find(b => re.test(b.el.textContent)); return b.el.style.visibility === 'visible' ? +b.el.style.opacity : 0; }; return { vmc: o(/^VMC: variational/), gfmc: o(/^GFMC: imaginary/) }; });
ok('S07 b3: the dim VMC board, which the top edge would cut, has faded; the dim GFMC board inside the stage has not', v.vmc <= 0.05 && v.gfmc > 0.38 && v.gfmc < 0.46, `VMC ${v.vmc.toFixed(2)}, GFMC ${v.gfmc.toFixed(2)}`);

/* S10 b5 (round 17): the beat ends on numbers that can be read. Every number chip on the stage and the summary board are drawn
   with letters 18 px high or more (1080p stage), and the six top-quark masses and the summary are whole on the stage */
await at(9, 5, null); v = await page.evaluate(() => { const px = e => parseFloat(getComputedStyle(e).fontSize) * e.getBoundingClientRect().height / e.offsetHeight, inside = r => r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080;
  const chips = []; for (const b of BOARDS.all) { if (b.el.style.visibility !== 'visible' || +b.el.style.opacity < 0.3) continue; for (const c of b.el.querySelectorAll('.chip')) { const r = c.getBoundingClientRect(); if (r.right < 0 || r.left > 1920 || r.bottom < 0 || r.top > 1080) continue; chips.push({ t: c.textContent.trim(), px: px(c), whole: inside(r) }); } }
  const top = ['175.55827(6)(648) GeV', '178.76384(5)(995) GeV', '181.85756(8)(1444) GeV', '346.44124(7)(1273) GeV', '349.33347(34)(4681) GeV', '514.46(4)(94) GeV'].map(t => chips.find(c => c.t === t));   /* round 19: each mass carries its unit */
  const sb = BOARDS.all.find(b => /^Top-quark benchmarks/.test(b.el.textContent.trim())), big = sb.q('.big');
  return { n: chips.length, min: Math.min(...chips.map(c => c.px)), small: chips.filter(c => c.px < 18).length, top: top.filter(c => c && c.whole).length, sum: px(big), sumWhole: sb.el.style.visibility === 'visible' && +sb.el.style.opacity > 0.8 && inside(sb.el.getBoundingClientRect()) }; });
ok('S10 b5 end: every number chip on the stage is 18 px or more, the six top-quark masses and the summary are whole, the summary 20 px or more', v.n >= 17 && v.small === 0 && v.top === 6 && v.sumWhole && v.sum >= 20, `${v.n} chips, smallest ${v.min.toFixed(1)} px, ${v.small} under 18 px, ${v.top} of 6 top masses whole, summary ${v.sum.toFixed(1)} px${v.sumWhole ? '' : ', not whole'}`);

/* 8. round 19 (S05): the renormalon series is shown as an infinite sum before its Borel sum; one letter for the order; the pink bar
      rises from the branch cut, not in the emptied graph; baryons and tetraquarks use the same tuned masses. Beats are the build's own. */
{
  const EQ = `BOARDS.all.find(b => /^\\s*𝒱/.test(b.el.textContent))`;
  const eqv = () => page.evaluate(src => { const eq = eval(src), o = e => { let a = 1; for (let n = e; n && n !== eq.el; n = n.parentElement) a *= +getComputedStyle(n).opacity; return a; };
    const s = eq.el.querySelector('#b2s'), j = eq.el.querySelector('#b2j'), rc = id => { const r = eq.el.querySelector(id).getBoundingClientRect(); return [r.left, r.top, r.width, r.height].map(x => +x.toFixed(1)); };
    const half = [...BOARDS.all].find(b => b.el.textContent.trim() === '½');
    return { n: scenes[4].beats.length, sum: s ? o(s) : -1, sumText: s ? s.textContent.replace(/\s+/g, '') : '', J: j ? o(j) : -1, box: ['#b1', '#b2', '#b3'].map(rc), contour: BOREL.contour.material.opacity,
      caps: [BOREL.label, BOREL.cLabel].map(b => b.el.style.visibility === 'visible' ? +b.el.style.opacity : 0), bp: BOREL.circ.material.opacity, cut: BOREL.cut.material.opacity, tick: half && half.el.style.visibility === 'visible' ? +half.el.style.opacity : 0, halo: getComputedStyle(eq.el.querySelector('#b3')).boxShadow }; }, EQ);
  await raw(4, 2, null); v = await eqv(); ok('S05 has 9 beats', v.n === 9, `${v.n}`);
  for (const k of [2, 3, 4]) { await raw(4, k, null); v = await eqv();
    ok(`S05 b${k} end: the second box shows the infinite sum of the V_l, not yet the closed form`, v.sum > 0.95 && v.J < 0.02 && /∞/.test(v.sumText) && /V/.test(v.sumText), `sum ${v.sum}, closed form ${v.J}, "${v.sumText}"`); }
  const box4 = v.box; await raw(4, 5, null); v = await eqv();
  ok('S05 b5 end: the sum has become the closed form 𝒥, with the Borel contour lit', v.J > 0.95 && v.sum >= 0 && v.sum < 0.02 && v.contour > 0.5, `sum ${v.sum}, closed form ${v.J}, contour ${v.contour.toFixed(2)}`);
  /* sizes, and places relative to the first box, to 0.3 px: the whole board drifts by a tenth of a pixel with the ambient camera */
  const rel = B => B.map(r => [r[0] - B[0][0], r[1] - B[0][1], r[2], r[3]]), same = (A, B) => rel(A).every((r, i) => r.every((x, j) => Math.abs(x - rel(B)[i][j]) <= 0.3));
  ok('S05 b4 → b5: none of the three boxes moves or changes size while the sum transforms', same(box4, v.box), `${JSON.stringify(rel(box4))} → ${JSON.stringify(rel(v.box))}`);
  await page.evaluate(src => { eval(src).el.querySelector('#b2j').style.display = 'none'; }, EQ); const planted = (await eqv()).box;
  await page.evaluate(src => { eval(src).el.querySelector('#b2j').style.display = ''; }, EQ);
  ok('… and the check can fail: with the closed form taken out of the shared cell, the boxes do move', !same(box4, planted), `${JSON.stringify(rel(planted))}`);
  /* round 19c: the sum rolls out of the box as the closed form rolls in: never one on the other, and the box never empty */
  let over = 0, least = 9; for (let i = 0; i <= 40; i++) { await raw(4, 5, .64 + .22 * i / 40); const w = await page.evaluate(src => { const eq = eval(src), B = eq.el.querySelector('#b2').getBoundingClientRect();
      const vis = id => { const e = eq.el.querySelector(id), r = e.getBoundingClientRect(), t = Math.max(r.top, B.top), b = Math.min(r.bottom, B.bottom); return { t, b, f: +getComputedStyle(e).opacity * Math.max(0, b - t) / r.height }; };
      const a = vis('#b2s'), c = vis('#b2j'); return { over: a.f > 0.02 && c.f > 0.02 ? Math.max(0, Math.min(a.b, c.b) - Math.max(a.t, c.t)) : 0, shown: a.f + c.f }; }, EQ);
    over = Math.max(over, w.over); least = Math.min(least, w.shown); }
  ok('S05 b5: while the sum turns into the closed form the two are never drawn on each other and the box is never empty (the shown parts add up to 0.8 of a formula or more)', over <= 1 && least >= 0.8, `largest overlap ${over.toFixed(1)} px, least shown ${least.toFixed(2)}`);
  for (const k of [5, 6]) { await raw(4, k, null); v = await eqv();
    ok(`S05 b${k} end: the branch point, the cut, the tick labels and the two captions of the inset are at full strength while the narration points at them`, v.bp >= 0.95 && v.cut >= 0.85 && v.tick >= 0.9 && v.caps.every(a => a >= 0.9), `branch point ${v.bp.toFixed(2)}, cut ${v.cut.toFixed(2)}, tick label ${v.tick}, captions ${v.caps}`); }
  v = await page.evaluate(() => { const lg = [...TERMS.b.el.querySelectorAll('div.small')].pop(); return { t: TERMS.b.el.textContent, lines: lg.offsetHeight / parseFloat(getComputedStyle(lg).lineHeight), over: lg.scrollWidth - lg.clientWidth }; });
  ok('|term| board: one letter for the order (no "L = 3", no script ℓ), bracket "used: l = 0, 1, 2"', !/L = 3/.test(v.t) && !/ℓ/.test(v.t) && /used: l = 0, 1, 2/.test(v.t), JSON.stringify(v.t.slice(0, 90)));
  ok('|term| board: the legend holds on one line, inside its panel', v.lines < 1.5 && v.over <= 1, `${v.lines.toFixed(2)} lines, ${v.over} px past its box`);
  const haze = () => page.evaluate(src => { const g = BOREL.circ.parent, P = u => W.project(g.localToWorld(new THREE.Vector3(30 * u - 15, .06, 0)));
    const eq = eval(src), h = document.querySelector('.hazeband').closest('.board'), r = h.getBoundingClientRect(), lr = eq.el.querySelector('#b3').getBoundingClientRect();
    return { over: +h.style.zIndex > +eq.el.style.zIndex, a: h.style.visibility === 'visible' ? +h.style.opacity : 0, cx: (r.left + r.right) / 2, cy: (r.top + r.bottom) / 2, w: r.width, bp: P(.5), far: P(1.45), mid: P(1), terms: TERMS.b.a, lr: [(lr.left + lr.right) / 2, (lr.top + lr.bottom) / 2], lrw: lr.width, halo: getComputedStyle(eq.el.querySelector('#b3')).boxShadow }; }, EQ);
  { const seen = []; for (const pp of [0.2, 0.4, 0.6, 0.7, 0.8, 0.84]) { await raw(4, 6, pp); seen.push((await haze()).a); }
    ok('S05 b6 up to p=.84: the pink bar is not there before its sentence', seen.every(a => a < 0.01), `bar opacity ${seen}`); }
  await raw(4, 6, null); v = await haze();
  ok('S05 b6 end (the Λ_R stop): the pink bar lies on the branch cut of the Borel plane, right of u = ½', v.a > 0.8 && v.cx - v.w / 2 > v.bp.x + 5 && v.cx + v.w / 2 < v.far.x + 5 && Math.abs(v.cy - v.mid.y) < 30,
    `bar opacity ${v.a}, x ${(v.cx - v.w / 2).toFixed(0)} to ${(v.cx + v.w / 2).toFixed(0)}, y ${v.cy.toFixed(0)}; branch point x ${v.bp.x.toFixed(0)}, cut ends x ${v.far.x.toFixed(0)}, cut y ${v.mid.y.toFixed(0)}`);
  ok('S05 b6 end: the emptied |term| graph is gone', v.terms < 0.01, `|term| board ${v.terms}`);
  await raw(4, 3, 0.5); const t3 = await page.evaluate(() => TERMS.b.a); await raw(4, 3, null); const t4 = await page.evaluate(() => TERMS.b.a);
  ok('|term| graph: there while the dots leave it (b3 p=.50), gone by the stop that follows (b3 end)', t3 > 0.9 && t4 < 0.01, `b3 p=.5 ${t3}, b3 end ${t4}`);
  await raw(4, 7, 0.2); v = await haze(); const dx = v.lr[0] - v.cx, dy = v.lr[1] - v.cy;
  ok('S05 b7 p=.20: the bar is on its way from the cut to the Λ_R box, drawn over the equation panel (not behind it)', v.a > 0.8 && v.cy < v.mid.y - 20 && Math.hypot(dx, dy) > 40 && v.over, `${v.over ? 'over' : 'BEHIND'} the panel, centre (${v.cx.toFixed(0)}, ${v.cy.toFixed(0)}), Λ_R at (${v.lr.map(x => x.toFixed(0))}), cut y ${v.mid.y.toFixed(0)}`);
  ok('S05 b7 p=.20: Λ_R still carries its halo', /px/.test(v.halo) && !/^none/.test(v.halo) && !/ 0px 0px$/.test(v.halo), v.halo);
  await raw(4, 7, 0.285); v = await haze();
  ok('S05 b7 p=.285: the bar is under the Λ_R box, about to reach it, already fading and no wider than the box', v.cy > v.lr[1] + 25 && v.a > 0.15 && v.a < 0.8 && v.w <= v.lrw + 2, `bar ${(v.cy - v.lr[1]).toFixed(0)} px under the box centre, opacity ${v.a}, width ${v.w.toFixed(0)} (box ${v.lrw.toFixed(0)})`);
  { const on = []; for (const pp of [0.31, 0.33, 0.35, 0.37, 0.39]) { await raw(4, 7, pp); on.push((await haze()).a); }
    ok('S05 b7 from p=.31: the bar has dissolved at the edge of the box: it never lies across the Λ_R symbol', on.every(a => a < 0.02), `bar opacity ${on}`); }
  /* its aim, while it can still be seen: seen from the Λ_R box, the bar at p=.24 and at p=.29 lies on one line through the box's centre
     (the camera moves in this beat, so both places are taken relative to the box) */
  { const rel = []; for (const pp of [0.24, 0.29]) { await raw(4, 7, pp); const w = await haze(); rel.push([w.cx - w.lr[0], w.cy - w.lr[1]]); }
    const off = Math.abs(rel[0][0] * rel[1][1] - rel[0][1] * rel[1][0]) / Math.hypot(...rel[0]);
    ok('S05 b7: the bar heads for the centre of the Λ_R box as it is drawn (its line passes within 1.5 px of it; 2.5 px without the as-drawn correction)', off <= 1.5 && Math.hypot(...rel[1]) < Math.hypot(...rel[0]), `from the box: ${rel.map(r => `(${r.map(x => x.toFixed(0))})`).join(' then ')}; the line misses the centre by ${off.toFixed(1)} px`); }
  for (const [k, pp] of [[7, null], [9, null]]) { await raw(4, k, pp); v = await haze();
    ok(`S05 b${k} end: the bar is gone and Λ_R is sharp (no halo left)`, v.a < 0.01 && (v.halo === 'none' || / 0px 0px$/.test(v.halo)), `bar ${v.a}, box-shadow "${v.halo}"`); }
  const wt = await page.evaluate(() => BOARDS.all.filter(b => b.scene === 4).map(b => b.el.textContent.replace(/\s+/g, ' ').trim()));
  const WHERE = 'Where does the renormalon go? ΛR comes out free of the leading-renormalon ambiguity. It is absorbed into the quark mass, tuned so that 2mQ + E matches the spin-averaged 1S states. For baryons and tetraquarks it is neglected, because they use the same tuned masses, which already include it.';
  ok('S05: the board on where the renormalon goes reads exactly as approved (round 19c: "neglected, because … already include it"); no Λ₁ anywhere in the scene, the chip says Λ_R', wt.includes(WHERE) && !wt.some(t => /Λ₁/.test(t)) && (await page.evaluate(() => BOARDS.all.some(b => b.el.textContent.trim() === 'ΛR absorbed into mQ'))), wt.includes(WHERE) ? 'exact' : JSON.stringify(wt.find(t => /^Where/.test(t))));
}

/* 9. round 19 (S04): Eq. A.1 as two statements; the smallest term's size is carried to the V(r) plot as the width of its band; the
      ambiguity is a slide of the whole curve; the slope is the same on every copy; the bands figure has an x axis (the order).
      Beats are the build's own. */
{
  const s4 = () => page.evaluate(() => {
    const vis = e => { if (!e) return -1; let a = 1; for (let n = e; n && n.nodeType === 1 && !n.classList.contains('board'); n = n.parentElement) { const o = n.getAttribute('opacity'); a *= (o == null ? 1 : +o) * +getComputedStyle(n).opacity; } return a; };
    const B = re => BOARDS.all.find(b => re.test(b.el.textContent.trim())), on = e => e && e.style.visibility === 'visible' ? +e.style.opacity : 0;
    const ser = B(/^The series/), bands = B(/^Bands at successive/), vp = BOARDS.all.find(b => b.el.querySelector('#vs')), q = id => vp.el.querySelector(id), R = e => { const r = e.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; };
    const dy = e => e ? +((/translate\(\s*[-\d.]+[ ,]+([-\d.]+)/.exec(e.getAttribute('transform') || '') || [0, 0])[1]) : null;
    const tm = TERMS.b.el.querySelector('#tmin'), a1 = q('#va1'), a2 = q('#va2'), fly = document.querySelector('.dvfly'), H = e => e ? e.getBoundingClientRect().height : 0, X = e => e ? e.getBoundingClientRect().left : 0;
    const tans = [...vp.el.querySelectorAll('.vtan')].map(e => ({ a: vis(e), slope: (+e.getAttribute('y2') - +e.getAttribute('y1')) / (+e.getAttribute('x2') - +e.getAttribute('x1')), x: +e.getAttribute('x1') }));
    const bars = [...bands.el.querySelectorAll('#bs rect')].map(R), tick = k => { const t = [...bands.el.querySelectorAll('#bs text')].find(t => t.textContent === k); return t ? R(t) : null; };
    const tr = TERMS.b.el.getBoundingClientRect();
    return { n: scenes[3].beats.length, ser: ser.el.textContent.replace(/\s+/g, ' ').trim(), cap: bands.el.textContent.replace(/\s+/g, ' ').trim(), bars, ticks: ['LO', 'NLO', 'NNLO'].map(tick), axisBottom: bands.el.querySelector('#bs path.ax').getBoundingClientRect().bottom,
      title: vp.el.querySelector('#vt').textContent, vp: on(vp.el), terms: on(TERMS.b.el), termsIn: tr.left >= 0 && tr.right <= 1920 && tr.top >= 0 && tr.bottom <= 1080,
      tmin: vis(tm), tminH: H(tm), a1: vis(a1), a1H: H(a1), a2: vis(a2), a2H: H(a2), a1x: X(a1), a2x: X(a2), fly: fly ? on(fly.closest('.board')) : -1,
      curveDy: dy(q('#vcurve')), g1: vis(q('#vg1')), g2: vis(q('#vg2')), band: vis(q('#vband')), bl: vis(q('#vbl')), fl: vis(q('#vfl')), fl2: vis(q('#vfl2')), tans, inset: BOREL.circ.material.opacity };
  });
  await raw(3, 3, null); v = await s4();
  ok('S04 has 7 beats', v.n === 7, `${v.n}`);
  ok('S04 series board: Eq. A.1 as two statements, V_R = the series, then 𝒱_R(r) = V_R(r) + Λ_R', /VR\(r\) = −CRr\s*Σl\s*vl αsl\+1\(μ\)\s*𝒱R\(r\) = VR\(r\) \+\s*ΛR\s*\(Eq\. A\.1\)/.test(v.ser), JSON.stringify(v.ser.slice(0, 80)));
  const cx = r => r[0] + r[2] / 2, yr = r => [r[1], r[1] + r[3]], apart = (a, b) => a[1] <= b[0] || b[1] <= a[0];
  const [LO, NLO, NNLO] = v.bars.length === 3 ? v.bars : [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  ok('S04 b3 end, bands figure: three ranges side by side along the x axis (the order), LO apart in |ΔE|, NLO and NNLO overlapping in |ΔE|',
    v.bars.length === 3 && LO[0] + LO[2] < NLO[0] && NLO[0] + NLO[2] < NNLO[0] && apart(yr(LO), yr(NLO)) && apart(yr(LO), yr(NNLO)) && !apart(yr(NLO), yr(NNLO)), JSON.stringify(v.bars.map(r => r.map(x => +x.toFixed(0)))));
  ok('S04 b3 end, bands figure: LO, NLO, NNLO are tick labels under the x axis, each under its range', v.ticks.every((t, i) => t && t[1] >= v.axisBottom - 2 && Math.abs(cx(t) - cx(v.bars[i])) < 6), JSON.stringify(v.ticks && v.ticks.map(t => t && t.map(x => +x.toFixed(0)))) + ` axis bottom ${v.axisBottom.toFixed(0)}`);
  ok('S04 bands figure: the caption says "at fixed mQ", not "vs mQ"', /binding energy at fixed mQ \(Figs\. 1–2 left, schematic\)/.test(v.cap) && !/vs mQ/.test(v.cap), JSON.stringify(v.cap.slice(-70)));
  await raw(3, 4, 0.85); v = await s4(); ok('S04 b4 p=.85: the δV marker is on its way from the smallest term to the V(r) plot', v.fly > 0.8 && v.tmin > 0.9, `flying marker ${v.fly}, mark on the term plot ${v.tmin}`);
  await raw(3, 4, null); v = await s4();
  ok('S04 b4 end (the link): the mark on the smallest term and the arrow across V(r) are both shown, the same length on the stage, both plots whole and at full strength', v.tmin > 0.9 && v.a1 > 0.9 && Math.abs(v.a1H - v.tminH) <= 1.5 && v.tminH > 20 && v.fly < 0.01 && v.vp > 0.9 && v.terms > 0.9 && v.termsIn && v.title === 'Origin: infrared',
    `mark ${v.tmin} (${v.tminH.toFixed(1)} px), arrow ${v.a1} (${v.a1H.toFixed(1)} px), flying ${v.fly}, V plot ${v.vp}, term plot ${v.terms}${v.termsIn ? '' : ' (not whole)'}, title "${v.title}"`);
  ok('S04 b4 end: no band, second arrow or force label yet', v.band < 0.01 && v.a2 < 0.01 && v.fl < 0.01, `band ${v.band}, second arrow ${v.a2}, force label ${v.fl}`);
  await raw(3, 5, 0.48); const up = (await s4()).curveDy; await raw(3, 5, 0.66); const dn = (await s4()).curveDy; await raw(3, 5, null); v = await s4();
  ok('S04 b5: the whole V(r) curve slides up, then down, and comes back', up !== null && up <= -15 && dn >= 15 && Math.abs(v.curveDy) < 0.5, `up ${up}, down ${dn}, end ${v.curveDy}`);
  ok('S04 b5 end: the two dashed copies and the band between them; two arrows of the same length, far apart in r; the band is named; no force label yet', v.g1 > 0.9 && v.g2 > 0.9 && v.band > 0.9 && v.a1 > 0.9 && v.a2 > 0.9 && Math.abs(v.a1H - v.a2H) <= 0.5 && v.a2x - v.a1x > 150 && v.bl > 0.9 && v.fl < 0.01,
    `copies ${v.g1}, ${v.g2}; band ${v.band}; arrows ${v.a1H.toFixed(1)} and ${v.a2H.toFixed(1)} px, ${(v.a2x - v.a1x).toFixed(0)} px apart; label ${v.bl}; force label ${v.fl}`);
  await raw(3, 6, null); v = await s4();
  ok('S04 b6 end: three tangent marks at the same r with the same slope, the force labels shown, the Borel inset not yet', v.tans.length === 3 && v.tans.every(t => t.a > 0.9 && Math.abs(t.slope - v.tans[0].slope) < 1e-6 && t.x === v.tans[0].x) && v.fl > 0.9 && v.fl2 > 0.9 && v.inset < 0.01,
    `${v.tans.length} marks, slopes ${v.tans.map(t => t.slope.toFixed(3))}, labels ${v.fl}, ${v.fl2}, inset ${v.inset}`);
  await raw(3, 7, null); v = await s4(); ok('S04 b7 end: the Borel inset is there', v.inset > 0.95, `branch point ${v.inset}`);
  await raw(4, 1, null); v = await s4(); ok('S05 b1 end: the mark on the smallest term is gone with the dashed line it sat on', v.tmin < 0.01, `mark ${v.tmin}`);
}

/* 10. round 19 (S06): the interpolating-scale plot shows the smooth cutoff only: no hard cutoff, no frozen α_s, one dashed line (the cutoff level) */
{
  await raw(5, 5, null); v = await page.evaluate(() => { const pl = BOARDS.all.find(b => b.el.querySelector('#ps')), sv = pl.el.querySelector('#ps'), shown = e => e.getAttribute('opacity') !== '0';
    const g2 = sv.querySelector('#g2'), ys = g2 ? [...g2.getAttribute('d').matchAll(/[ML][\d.]+ ([\d.]+)/g)].map(m => +m[1]) : [];
    const cnt = {}; ys.forEach(y => { cnt[y] = (cnt[y] || 0) + 1; });
    return { text: BOARDS.all.filter(b => b.scene === 5).map(b => b.el.textContent).join(' ¦ '), dashed: [...sv.querySelectorAll('[stroke-dasharray]')].filter(shown).length, flat: ys.length ? Math.max(...Object.values(cnt)) / ys.length : 1, n: ys.length,
      lab: [...sv.querySelectorAll('text')].map(t => t.textContent).find(t => /2 GeV/.test(t)) || '' }; });
  ok('S06 end: nothing names a hard cutoff or frozen α_s', !/frozen|hard cutoff|kink at/i.test(v.text), (/frozen|hard cutoff|kink at/i.exec(v.text) || ['none'])[0]);
  ok('S06 end: one dashed line on the plot (the cutoff level), no dotted ones', v.dashed === 1, `${v.dashed} dashed or dotted strokes shown`);
  ok('S06 end: the 2 GeV curve is the smooth μ′ too (no flat stretch after a kink: under 30 % of its points at one height), named "μcut = 2 GeV → fixed order", and nothing in the scene cites Fig. 7', v.n > 50 && v.flat < 0.3 && v.lab === 'μcut = 2 GeV → fixed order' && !(await page.evaluate(() => BOARDS.all.some(b => b.scene === 5 && /Fig\. 7/.test(b.el.textContent)))), `${(100 * v.flat).toFixed(0)} % of its points at one height; label "${v.lab}"`);
}

/* 10d. round 24 (S06 b5): the scale variation λ ∈ {½, 1, 2} is one half-transparent band between the λ = ½ and λ = 2
   curves (the systematic uncertainty), under the λ = 1 line; no separate λ = ½ and λ = 2 lines */
{
  await raw(5, 5, null); v = await page.evaluate(() => { const sv = BOARDS.all.find(b => b.el.querySelector('#ps')).el.querySelector('#ps'), shown = e => +(e.getAttribute('opacity') ?? 1) > 0.5;
    const band = sv.querySelector('#lamband'), lines = [...sv.querySelectorAll('path')].filter(e => e.getAttribute('fill') === 'none' && shown(e) && !e.classList.contains('ax'));
    const c1 = [...sv.querySelectorAll('path')].find(e => e.getAttribute('stroke-width') === '2.4');
    return band ? { shown: shown(band), fo: +band.getAttribute('fill-opacity'), closed: /Z\s*$/.test(band.getAttribute('d')), under: !!c1 && !!(band.compareDocumentPosition(c1) & Node.DOCUMENT_POSITION_FOLLOWING), lines: lines.length } : { lines: lines.length }; });
  ok('S06 end: the λ variation is one half-transparent band (fill-opacity .15 to .5), closed, under the λ = 1 line', !!v.shown && v.fo >= 0.15 && v.fo <= 0.5 && v.closed && v.under, JSON.stringify(v));
  ok('S06 end: two lines left on the plot (μ′ and the 2 GeV curve), no λ = ½ or λ = 2 line', v.lines === 2, `${v.lines} lines`);
  /* review 5.3: the band's edges are μ′(r; 2) on top and μ′(r; ½) below (plot: x = 60 + r/1.2·560, y = 260 − μ/4.4·240) */
  v = await page.evaluate(() => { const d = document.querySelector('#lamband').getAttribute('d'), P = [...d.matchAll(/([\d.]+) ([\d.]+)/g)].map(m => [+m[1], +m[2]]);
    const X = r => 60 + r / 1.2 * 560, Y = m => 260 - Math.min(m, 4.3) / 4.4 * 240, mu = (r, l) => Math.sqrt(1 + (l * 0.1973 / r) ** 2);
    return [0.3, 0.6, 1.0].map(r => { const at = P.filter(p => Math.abs(p[0] - X(r)) < 0.6).map(p => p[1]); return { r, top: Math.min(...at) - Y(mu(r, 2)), bot: Math.max(...at) - Y(mu(r, 0.5)) }; }); });
  ok('S06 end: the band runs from μ′(r; ½) to μ′(r; 2), within 1 px at 0.3, 0.6, 1.0 fm', v.every(x => Math.abs(x.top) < 1 && Math.abs(x.bot) < 1), JSON.stringify(v));
}

/* 10b. round 24 (S06): the 3D shows what the cutoff does with a gauge for α_s (three versions were compared; the author chose the gauge).
   b1: quark B starts close (0.2 fm) and walks out with μ = 1/r: past 0.56 fm α_s(1/r) blows up, and the version shows it
   broken (bond: red, torn apart; gauge: needle past the end stop, "∞"; lens: the probe ring, 1/μ across, red, out in the fog).
   b3: B walks out again with μ′: α_s(μ′) stays near 0.48 at 1.25 fm, nothing broken. α_s: two loops, n_f = 3,
   Λ = 0.35 GeV (pole at 1/r = Λ, 0.56 fm, the ruler's red edge); the label keeps the caveat (no σr in V) */
{
  const shown = re => page.evaluate(src => BOARDS.all.filter(b => new RegExp(src).test(b.el.textContent)).map(b => b.el.style.visibility === 'visible' ? +b.el.style.opacity : 0).reduce((a, x) => Math.max(a, x), 0), re);
  const layout = () => page.evaluate(() => { const vis = b => b.el.style.visibility === 'visible' && +b.el.style.opacity > 0.3, mine = (PSI.labels || []).filter(vis);
    return mine.map(L => { const r = L.el.getBoundingClientRect(); return { t: L.el.textContent.trim().slice(0, 24), in: r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080,
      hit: BOARDS.all.filter(b => b !== L && vis(b)).filter(b => { const q = b.el.getBoundingClientRect(); return Math.min(r.right, q.right) - Math.max(r.left, q.left) > 0 && Math.min(r.bottom, q.bottom) - Math.max(r.top, q.top) > 0; }).map(b => b.el.textContent.trim().slice(0, 24)) }; }); });
  /* what each version draws, read back from the drawn objects */
  const drawn = () => page.evaluate(() => { if (typeof PSI === 'undefined' || !PSI.parts) return null; const P = PSI.parts, v = 'gauge', c = m => { const k = m.material.color; return { r: k.r, g: k.g, b: k.b }; };
    if (v === 'bond') return { v, vis: P.bondA.visible && P.bondA.material.opacity > 0.3, col: c(P.bondA), gap: P.gap(), th: P.bondA.userData.th, text: P.tag.el.textContent };
    if (v === 'gauge') { const n = P.gauge.el.querySelector('#ndl'), x = +n.getAttribute('x2') - 130, y = 130 - +n.getAttribute('y2');   /* the drawn needle, not the variable behind it */
      return { v, vis: P.gauge.el.style.visibility === 'visible' && +P.gauge.el.style.opacity > 0.3, angle: 180 - Math.atan2(y, x) * 180 / Math.PI, text: P.gauge.el.textContent }; }
    return { v, vis: P.ring.visible && P.ring.material.opacity > 0.3, col: c(P.ring), rad: P.ring.userData.r, fog: P.fogR, text: P.tag.el.textContent }; });
  const red = k => k.r > 0.3 && k.g < k.r / 3, num = t => +(/αs\s*[=≤]\s*([\d.]+)/.exec(t.replace(/\s+/g, ' ')) || [0, NaN])[1];   /* H.as reads "αs" as text */
  v = await page.evaluate(() => typeof PSI !== 'undefined' && PSI.alpha ? [PSI.alpha(1), PSI.alpha(0.34), PSI.alpha(0.1973 / 0.2)] : null);
  ok('S06: α_s(1 GeV) is 0.45 to 0.50, α_s blows up under Λ = 0.35 GeV', !!v && v[0] > 0.45 && v[0] < 0.5 && !isFinite(v[1]), JSON.stringify(v));
  for (const ver of ['gauge']) {   /* round 24: the author chose the gauge; bond and lens are gone */
    await raw(5, 1, 0.56); v = await page.evaluate(() => ({ r: PAIR.st.sep / 12, st: PSI.view() }));
    ok(`S06 ${ver} b1 p=.56: B starts close (under 0.3 fm), nothing broken yet`, v.r < 0.3 && !v.st.broken, `r ${v.r.toFixed(2)} fm, ${JSON.stringify(v.st)}`);
    await raw(5, 1, null); v = await drawn(); let w = await page.evaluate(() => PSI.view());
    const brk = !!v && v.vis && (ver === 'bond' ? red(v.col) && v.gap > 0.5 && /∞/.test(v.text) : ver === 'gauge' ? v.angle >= 179 && /∞/.test(v.text) : red(v.col) && v.rad > v.fog);
    ok(`S06 ${ver} b1 end (1.25 fm, μ = 1/r): drawn broken`, w.broken && brk, JSON.stringify({ ...v, view: w }));
    v = await layout(); ok(`S06 ${ver} b1 end: labels whole on stage, on no other board`, v.length >= 1 && v.every(x => x.in && x.hit.length === 0), JSON.stringify(v));
    await raw(5, 3, null); v = await drawn(); w = await page.evaluate(() => PSI.view());
    const calm = !!v && v.vis && (ver === 'bond' ? !red(v.col) && v.gap < 0.01 && Math.abs(num(v.text) - 0.48) < 0.03 : ver === 'gauge' ? Math.abs(v.angle - 180 * 0.48 / 1.48) < 4 && Math.abs(num(v.text) - 0.48) < 0.03 : !red(v.col) && v.rad < 1.25 && v.rad < v.fog);
    ok(`S06 ${ver} b3 end (1.25 fm, μ′): drawn calm, α_s near 0.48`, !w.broken && Math.abs(w.r - 1.25) < 0.05 && calm, JSON.stringify({ ...v, view: w }));
    v = await layout(); ok(`S06 ${ver} b3 end: labels whole on stage, on no other board`, v.length >= 1 && v.every(x => x.in && x.hit.length === 0), JSON.stringify(v));
    v = await shown('confin'); ok(`S06 ${ver} b3 end: the label keeps the caveat (no confining term in V)`, v > 0.8, `opacity ${v.toFixed(2)}`);
  }
  v = await page.evaluate(() => ({ sw: typeof S06V !== 'undefined', parts: Object.keys(PSI.parts || {}).filter(k => /bond|ring|fog|tag/.test(k)) }));
  ok('S06: only the gauge is left (no version switch, no bond, ring or tag)', !v.sw && v.parts.length === 0, JSON.stringify(v));
  v = await page.evaluate(() => PSI.RC / 12);   /* 1 fm = 12 world units */
  ok('S06: the ruler turns red at 1/r = 0.35 GeV (0.56 fm)', Math.abs(v - 0.1973 / 0.35) < 0.02, `${(+v).toFixed(2)} fm`);
  /* review 1.3: the ruler is coloured by the gauge's own zones: teal while α_s(1/r) < 0.82, amber to 1.5, red beyond */
  v = await page.evaluate(() => { const rAt = a => { let lo = 0.05, hi = 0.5636; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; (PSI.alpha(0.1973 / m) < a ? lo = m : hi = m); } return lo; };
    const end = l => l ? l.geometry.attributes.position.array[3] / 12 : NaN, Z = PSI.zones || {};
    return { want: [rAt(0.82), rAt(1.5)], got: [end(Z.teal), end(Z.amber)], far: Z.red ? Z.red.geometry.attributes.position.array[3] / 12 : NaN }; });
  ok('S06: the ruler is teal to α_s(1/r) = 0.82, amber to 1.5, red beyond (the gauge\'s zones), and ends by 1.7 fm', Math.abs(v.got[0] - v.want[0]) < 0.01 && Math.abs(v.got[1] - v.want[1]) < 0.01 && v.far <= 1.7, JSON.stringify(v));
  v = await page.evaluate(() => PSI.labels.map(b => b.el.textContent).join(' ¦ '));
  ok('S06: the gauge says it is schematic; the hill is r²|ψ|², bottomonium-like', /schematic/.test(await page.evaluate(() => PSI.parts.gauge.el.textContent)) && /r²\|ψ\|²/.test(v) && /bottomonium/.test(v), v.slice(0, 300));
  await raw(5, 1, 0.53); v = await page.evaluate(() => { const e = PSI.parts.gauge.el; return e.style.visibility === 'visible' ? +e.style.opacity || 0 : 0; });
  ok('S06 b1 p=.53: the gauge is not up yet while B comes in (no red-then-teal swing)', v < 0.05, `opacity ${v.toFixed(2)}`);
  /* review 2.2: the λ line says what the band does, in the paper's words */
  await raw(5, 5, null); v = await page.evaluate(() => [...document.querySelectorAll('#ps text')].map(t => [...t.childNodes].map(n => n.textContent).join(' ').replace(/\s+/g, ' ')).find(t => /λ ∈/.test(t)) || '');   /* its two lines read with a space between */
  ok('S06 end: the λ line reads "the band probes the residual scale dependence"', /band probes the residual scale dependence/.test(v), v);
  await raw(5, 3, null);
  /* the |ψ|² hill is a heavy bound state's size (Bohr radius: cc̄ 0.49 fm, bb̄ 0.21 fm): it peaks short of the red zone and
     most of its weight is outside it, which is why a V without confinement is a small systematic */
  v = await page.evaluate(() => { const m = PSI.mesh, a = m.geometry.attributes.position.array, A = PAIR.posA(); let pk = 0, hx = -1, w = 0, wr = 0;
    for (let i = 0; i < a.length; i += 3) { if (Math.abs(a[i + 1]) > 1e-6) continue; const fm = (m.position.x + a[i] * m.scale.x - A.x) / (12 * m.scale.x), h = a[i + 2];
      if (h > hx) { hx = h; pk = fm; } w += h; if (fm > PSI.RC / 12) wr += h; } return { pk, out: wr / w }; });   /* the height is r²|ψ|²: the weight per unit r */
  ok('S06 b3 end: the |ψ|² hill peaks under 0.4 fm, with under 35 % of its weight in the red zone', v.pk < 0.4 && v.out < 0.35, `peak ${v.pk.toFixed(2)} fm, ${(100 * v.out).toFixed(0)} % in the red`);
}

/* 10c. round 20/23 (S01): the GdR QCD 2026 logo (the meeting where the talk is given) and the Fermilab and CPT logos (the
   affiliations) are on the title frame, loaded, whole on
   the stage, clear of every other board shown */
for (const [src, name, minW] of [['gdr-qcd-2026', 'GdR QCD 2026', 180], ['fermilab', 'Fermilab', 150], ['cpt', 'CPT', 80], ['cincinnati', 'University of Cincinnati', 80]]) for (const k of [1, 2]) {
  await raw(0, k, null); await page.evaluate(() => Promise.all([...document.images].map(i => i.decode().catch(() => {}))));
  /* b2 moves the focus to the thesis line and the engine dims every other S01 board to the same level: the logo must go with the title */
  v = await page.evaluate(([k, src]) => { const tl = BOARDS.all.find(b => /^Renormalon-subtracted/.test(b.el.textContent.trim())), vis = b => b.el.style.visibility === 'visible' && (k === 1 ? +b.el.style.opacity > 0.5 : +b.el.style.opacity > 0.3 && b.el.style.opacity === tl.el.style.opacity), L = BOARDS.all.find(b => b.el.querySelector(`img[src*="${src}"]`));
    if (!L) return { there: false }; const img = L.el.querySelector(`img[src*="${src}"]`), r = img.getBoundingClientRect();
    const hit = BOARDS.all.filter(b => b !== L && b.el.style.visibility === 'visible' && +b.el.style.opacity > 0.3).filter(b => { const q = b.el.getBoundingClientRect(); return Math.min(r.right, q.right) - Math.max(r.left, q.left) > 0 && Math.min(r.bottom, q.bottom) - Math.max(r.top, q.top) > 0; }).map(b => b.el.textContent.trim().slice(0, 30));
    return { there: true, shown: vis(L), op: L.el.style.opacity, loaded: img.naturalWidth > 0, w: r.width, in: r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080, hit }; }, [k, src]);
  ok(`S01 b${k} end: the ${name} logo is shown (b2: at the title's own dimmed level), loaded, >= ${minW} px wide, whole on stage, on no other board`, v.there && v.shown && v.loaded && v.w >= minW && v.in && v.hit.length === 0, JSON.stringify(v));
}

/* 11. round 19 (S09, S10): the measured B_c is on the number line beside the spin average it enters; every mass on the shelf carries its unit */
{
  await raw(8, 4, null); v = await page.evaluate(() => { const sv = NLINE.b.el.querySelector('svg'), T = [...sv.querySelectorAll('text')].map(t => { const r = t.getBoundingClientRect(), g = t.closest('g'); return { t: t.textContent, x0: r.left, x1: r.right, y0: r.top, y1: r.bottom, a: g ? +g.getAttribute('opacity') : 1 }; }), br = NLINE.b.el.getBoundingClientRect();
    const hit = (a, b) => Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0) >= 2 && Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) >= 4,   /* 4 px: the boxes of two stacked lines share their leading */ lab = T.filter(t => t.a > 0.9 && t.t), exp = T.find(t => t.t === 'exp. Bc 6.2745(3) GeV'), ours = T.find(t => /^ours/.test(t.t));
    return { exp: !!exp && exp.a > 0.9, inside: !!exp && exp.x0 >= br.left && exp.x1 <= br.right, left: !!exp && !!ours && exp.x0 < ours.x0, clash: lab.filter((a, i) => lab.some((b, j) => j > i && hit(a, b))).map(a => a.t), all: lab.map(t => t.t) }; });
  ok('S09 b4 end: the measured B_c, "exp. Bc 6.2745(3)", is on the number line, left of ours, whole inside the panel, and no two labels of the line touch', v.exp && v.inside && v.left && v.clash.length === 0, v.exp ? `labels: ${JSON.stringify(v.all)}${v.clash.length ? '; touching: ' + JSON.stringify(v.clash) : ''}` : `not there: ${JSON.stringify(v.all)}`);
  await raw(9, 6, null); v = await page.evaluate(() => { const c = []; for (const b of BOARDS.all) { if (b.el.style.visibility !== 'visible' || +b.el.style.opacity < 0.3) continue; for (const e of b.el.querySelectorAll('.chip')) if (/\(\d+\)/.test(e.textContent)) c.push(e.textContent.trim()); }
    return { c, dup: BOARDS.all.some(b => /±0\.94 GeV band/.test(b.el.textContent)), hdr: BOARDS.all.filter(b => b.scene === 9).map(b => b.el.textContent.trim()).filter(t => /^(NNLO MRS|lattice QCD)/.test(t)) }; });
  ok('S10 end: no "±0.94 GeV band" note beside the Ω_ttt mass (its (94) says it)', !v.dup, v.dup ? 'still there' : 'gone');
  ok('S10 end: every mass on the shelf carries its unit (GeV), the row headers no longer do', v.c.length >= 17 && v.c.every(t => / GeV/.test(t)) && v.hdr.length === 2 && v.hdr.every(t => !/GeV/.test(t)), `${v.c.filter(t => !/ GeV/.test(t)).length} of ${v.c.length} masses without a unit: ${JSON.stringify(v.c.filter(t => !/ GeV/.test(t)).slice(0, 4))}; headers ${JSON.stringify(v.hdr)}`);
}

/* 12. round 19 (S12): the colour-structure board and its beat are gone; the scene ends on the two verdict boards, whole on the stage */
{
  await raw(11, 5, null); v = await page.evaluate(() => ({ n: scenes[11].beats.length, col: BOARDS.all.some(b => /Colour structure|diquark/.test(b.el.textContent)),
    ends: BOARDS.all.filter(b => /^(BOUND|UNBOUND)/.test(b.el.textContent.trim())).map(b => { const r = b.el.getBoundingClientRect(); return b.el.style.visibility === 'visible' && +b.el.style.opacity > 0.8 && r.left >= 0 && r.right <= 1920 && r.top >= 0 && r.bottom <= 1080; }) }));
  const tq = await page.evaluate(() => ['bbtt', 'bctt', 'cctt'].map(id => SHELF.glyphs[id].chip.el.innerText.replace(/\s+/g, ' ').trim()));
  ok('S12 end: the bound tetraquarks carry their MRS masses, marked NLO (Tab. V), not "top decays"', JSON.stringify(tq) === JSON.stringify(['353.479(10)(371) GeV NLO', '350.376(9)(401) GeV NLO', '347.216(8)(414) GeV NLO']), JSON.stringify(tq));
  ok('S12 has 5 beats, no colour-structure board, and ends on the two verdict boards whole on the stage', v.n === 5 && !v.col && v.ends.length === 2 && v.ends.every(Boolean), `${v.n} beats, colour board ${v.col ? 'still there' : 'gone'}, verdict boards ${JSON.stringify(v.ends)}`);
}

/* 12b. round 24 (S12): the BOUND board says one thing: each state's binding energy (dot) with its scale range (band, ± the
   second parenthesis of Tab. V, round 24c) on one MeV axis from 0, every band clear of 0. No fixed-order bars, no "MRS is wider" */
{
  await raw(11, 5, null); v = await page.evaluate(() => { const B = BOARDS.all.find(b => /^BOUND/.test(b.el.textContent.trim())); if (!B) return null;
    const zero = B.el.querySelector('[data-zero]'), Z = zero ? zero.getBoundingClientRect().left : NaN;
    const rows = [...B.el.querySelectorAll('[data-band]')].map(e => { const r = e.getBoundingClientRect(), d = e.parentElement.querySelector('[data-dot]').getBoundingClientRect(); return { l: r.left - Z, w: r.width, dot: d.left + d.width / 2 - Z, mid: r.left + r.width / 2 - Z }; });
    const t1 = [...B.el.querySelectorAll('[data-tick]')].find(e => /1000/.test(e.textContent)), X1 = t1 ? t1.getBoundingClientRect().left + t1.getBoundingClientRect().width / 2 - Z : NaN;
    const axisW = zero ? zero.parentElement.getBoundingClientRect().width : 0;
    return { text: B.el.textContent, rows, X1, axisW, ax: [...B.el.querySelectorAll('[data-tick]')].map(e => e.textContent.trim()) }; });
  ok('S12 end: the BOUND board has no fixed-order bars and no "MRS is wider"', !!v && !/hatched|fixed order|wider/i.test(v.text), v ? (/hatched|fixed order|wider/i.exec(v.text) || ['clean'])[0] : 'no board');
  ok('S12 end: three bands, each clear of the 0 line, the dot at its middle, widths in the ratio 340 : 410 : 479', !!v && v.rows.length === 3 && v.rows.every(r => r.l > 4 && Math.abs(r.dot - r.mid) < 2) && Math.abs(v.rows[1].w / v.rows[0].w - 410 / 340) < 0.03 && Math.abs(v.rows[2].w / v.rows[0].w - 479 / 340) < 0.03, v ? JSON.stringify(v.rows) : 'none');
  ok('S12 end: one MeV axis with 0 marked', !!v && v.ax[0] === '0' && v.ax.some(t => /MeV/.test(t)), v ? JSON.stringify(v.ax) : 'none');
  /* review 5.4: positions in MeV, read against the 1000 MeV tick: dots at B, bands as wide as the range of Tab. V */
  /* the second parenthesis of Tab. V is ± (the author, round 24c): 509(10)(340) is 509 ± 340, a band from 169 to 849 MeV */
  ok('S12 end: dots at 509, 604, 742 MeV and bands ± 340, 410, 479 MeV on the axis (within 10 MeV), the widest end (1221 MeV) on the axis', !!v && [[509, 680], [604, 820], [742, 958]].every(([b, w], i) => Math.abs(v.rows[i].dot / v.X1 * 1000 - b) < 10 && Math.abs(v.rows[i].w / v.X1 * 1000 - w) < 10) && v.rows[2].l + v.rows[2].w <= v.axisW + 1, v ? JSON.stringify(v.rows.map(r => [Math.round(r.dot / v.X1 * 1000), Math.round(r.w / v.X1 * 1000)])) : 'none');
  ok('S12 end: the header names B (not ΔE) and the caption says bound at every λ', !!v && /B ≈ 509–742 MeV/.test(v.text) && !/ΔE/.test(v.text) && /every λ/.test(v.text), v ? v.text.slice(0, 200) : 'none');
  v = await page.evaluate(() => [KEYS[11].map(k => k.k.join(' ')).join(' | '), NOTES[11].join(' ')]);
  ok('S12: keynotes and notes no longer claim a small shift between orders (the board does not show it)', !/shift between orders|order to order/.test(v.join(' ')), (/.{0,40}(shift between orders|order to order).{0,20}/.exec(v.join(' ')) || ['clean'])[0]);
}

/* 13. round 19c: the open points of the third review, changed at the author's request */
{
  await raw(8, 4, null); v = await page.evaluate(() => [...document.querySelectorAll('svg text')].map(t => t.textContent.trim()).filter(t => /MeV gap|three times/.test(t)));
  ok('S09 b4: the gap label is just "24.8 MeV gap" (no statement about the scale range)', v.length === 1 && v[0] === '24.8 MeV gap', JSON.stringify(v));
  await raw(9, 3, null); v = await page.evaluate(() => [...document.querySelectorAll('svg text')].map(t => t.textContent.trim()).filter(t => /^(MRS|fixed order|lattice) 14\./.test(t)));
  ok('S10 b3: the three masses on the Ω_bbb number line carry GeV', v.length === 3 && v.every(t => / GeV/.test(t)), JSON.stringify(v));
  await raw(9, 6, null); v = await page.evaluate(() => BOARDS.all.filter(b => /Top-quark benchmarks/.test(b.el.textContent)).map(b => b.el.textContent.trim()));
  ok('S10 end: the summary board gives the Ω_ttt mass once, without "±0.94 GeV"', v.length === 1 && /514\.46\(4\)\(94\) GeV/.test(v[0]) && !/0\.94/.test(v[0]), JSON.stringify(v));
  await raw(11, 5, null); v = await page.evaluate(() => BOARDS.all.find(b => /^BOUND/.test(b.el.textContent.trim())).el.innerText.match(/\d+\([\d()]+\) MeV/g));
  ok('S12 end: the binding energies carry both errors of Tab. V', JSON.stringify(v) === JSON.stringify(['509(10)(340) MeV', '604(9)(410) MeV', '742(8)(479) MeV']), JSON.stringify(v));
  await raw(7, 3, null); v = await page.evaluate(() => BOARDS.all.some(b => /Λ₁/.test(b.el.textContent)));
  ok('no Λ₁ on any board (S08 dials included): the constant is Λ_R everywhere', !v, v ? 'still there' : 'gone');
}

let v2;
/* 14. round 21 (S04): the potential plot reads as −C/r (steep at short distance, flattening under a zero line) and its axis is 𝒱 */
{
  await raw(3, 5, null); v = await page.evaluate(() => { const vs = document.querySelector('#vs'), c = vs.querySelector('#vcurve'), n = c.getTotalLength(), a = c.getPointAtLength(0), m = c.getPointAtLength(n * 0.25), e = c.getPointAtLength(n), z = vs.querySelector('#vzero');
    return { drop: a.y - e.y, early: (a.y - m.y) / (a.y - e.y), zero: z ? +z.getAttribute('y1') : null, top: e.y, axis: [...vs.querySelectorAll('text')].map(t => t.textContent.trim()).filter(t => /^[V𝒱0]$/u.test(t)) }; });
  ok('S04: the curve falls steeply at short distance (a drop of 120 px or more, most of it in the first quarter of its length) and flattens under a dashed zero line', v.drop >= 120 && v.early >= 0.55 /* 0.6 before the scale that keeps the upper copy under zero: the steep part is longer now */ && v.zero != null && v.zero < v.top, `drop ${v.drop.toFixed(0)} px, ${(100 * v.early).toFixed(0)} % of it in the first quarter, zero line at y ${v.zero}, curve ends at y ${v.top.toFixed(0)}`);
  v2 = await page.evaluate(() => { const vs = document.querySelector('#vs'), g = vs.querySelector('#vg1'), n = g.getTotalLength(); let top = 1e9; for (let i = 0; i <= 50; i++) top = Math.min(top, g.getPointAtLength(n * i / 50).y); return { top, zero: +vs.querySelector('#vzero').getAttribute('y1'), op: +g.getAttribute('opacity') }; });
  ok('S04 b5: the upper dashed copy of the curve stays under the zero line', v2.op > 0.9 && v2.top >= v2.zero + 3, `copy reaches y ${v2.top.toFixed(1)}, zero line at y ${v2.zero}`);
  ok('S04: the y axis is the script 𝒱 of Eq. A.1, with a 0 at the zero line', JSON.stringify(v.axis.slice().sort()) === JSON.stringify(['0', '𝒱'].sort()), JSON.stringify(v.axis));
}

console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
await b.close(); process.exit(R.every(Boolean) && !errs.length ? 0 : 1);
