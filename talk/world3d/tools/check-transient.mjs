// check-transient.mjs: is any text written and gone again inside one beat? Every beat is played on a virtual 30 fps clock
// (the World's own loop, the capture's timing) and every text unit is read at every frame.
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-transient.mjs [--only S03] [--fps 30]
//   ... --strings FILE   also write every text ever drawn at >= .3 (the on-screen string set, for a before/after diff)
//   ... --selftest       two probes in S02 b2: one rises and falls (must be flagged), one rises and stays (must not be)
// VISIBLE = drawn opacity >= .3, at least half of its ink on the stage, and readable: its tallest line 8 px or more (round 18: the
// lift-off to the map shrank the S14 b4 outlook from 31 px to 4 px inside the beat that wrote it). An offender is not visible when the beat starts,
// written in this beat (shown by .15 or more more than at the start, before tier dimming and the edge fade: scenery a camera move
// merely passes is not written), visible at some frame, not visible when it has settled: FADE (it faded) or LEFT-STAGE (the camera left it). MARK = a small
// round filled element without text (a flying dot) doing the same. Any offender fails the run.
import fs from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH + '/');
const { chromium } = require('playwright');
const argv = process.argv.slice(2), arg = n => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : null; };
const selftest = argv.includes('--selftest'), only = selftest ? 'S02' : arg('--only'), FPS = Number(arg('--fps') || 30), strOut = arg('--strings');
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
const page = await b.newPage({ viewport: { width: 1920, height: 1080 } });
const errs = []; page.on('pageerror', e => errs.push(String(e)));
await page.addInitScript(() => { let q = []; window.requestAnimationFrame = cb => { q.push(cb); return q.length; }; window.cancelAnimationFrame = () => { };
  window.__step = ts => { const c = q; q = []; c.forEach(f => f(ts)); }; });
await page.goto(new URL('../index.html', import.meta.url).href);
await page.evaluate(() => document.fonts.ready);
await page.evaluate(() => {
  W.composer.render = () => { }; W.renderer.render = () => { };   /* no GPU draw: the camera is still applied, the boards still projected */
  const boards = [...document.querySelectorAll('.board')], idx = new Map(boards.map((e, i) => [e, i])), obj = new Map(BOARDS.all.map(b => [b.el, b]));
  /* written: how far the unit itself is shown (its opacity inside its board x the board's show amount), before the tier dims and the edge fade */
  const wrOf = el => { let o = 1; for (let e = el; e && e.id !== 'stage'; e = e.parentElement) { const bo = obj.get(e); if (bo) return o * bo.a;
      const cs = getComputedStyle(e); if (cs.display === 'none') return 0; o *= parseFloat(cs.opacity); } return o; };   /* visibility: inherited from the board's off-stage cull, not a choice */
  const op = (el, ops) => { if (ops.has(el)) return ops.get(el); const cs = getComputedStyle(el); let o = cs.display === 'none' || cs.visibility === 'hidden' ? 0 : parseFloat(cs.opacity);
    if (o > 0 && el.parentElement && el.id !== 'stage') o *= op(el.parentElement, ops); ops.set(el, o); return o; };
  const unitOf = el => { const chip = el.closest('.chip'); if (chip) return chip; if (el.ownerSVGElement) return el.closest('text') || el;
    for (let e = el; e; e = e.parentElement) if (!getComputedStyle(e).display.startsWith('inline') || e.classList.contains('board')) return e; return el; };
  /* every text unit and mark: key -> [drawn opacity, fraction of its ink on the stage] */
  window.__units = () => {
    const stage = document.getElementById('stage'), ops = new Map(), U = new Map(), rg = document.createRange(), tw = document.createTreeWalker(stage, NodeFilter.SHOW_TEXT);
    const add = (key, o, rects, wr) => { let a = 0, inA = 0; for (const r of rects) { const w = r.width * r.height; a += w;
        inA += Math.max(0, Math.min(r.right, 1920) - Math.max(r.left, 0)) * Math.max(0, Math.min(r.bottom, 1080) - Math.max(r.top, 0)); }
      const u = U.get(key) || U.set(key, { o: 0, a: 0, inA: 0, wr: 0, h: 0 }).get(key); u.o = Math.max(u.o, o); u.wr = Math.max(u.wr, wr); for (const r of rects) u.h = Math.max(u.h, r.height); u.a += a; u.inA += inA; };
    for (let n = tw.nextNode(); n; n = tw.nextNode()) { const t = n.nodeValue.trim(); if (!t) continue; const el = n.parentElement;
      if (el.closest('.badge') || el.closest('#tau') || el.closest('#fps') || el.closest('#beatid') || el.closest('#notes')) continue;
      const o = op(el, ops), wr = wrOf(el); if (o < 0.05 && wr < 0.05) continue; const unit = unitOf(el), bd = el.closest('.board');   /* hidden off stage still counts as shown */
      const key = (bd ? 'b' + idx.get(bd) : 'x') + ':' + (unit.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 60);
      rg.selectNodeContents(n); add(key, o, [...rg.getClientRects()].filter(r => r.width >= 1 && r.height >= 1), wr); }
    /* marks: filled elements without text, keyed by element; round = a small dot at this frame (a growing bar is round only while it is short,
       so the end is judged by opacity and place, not by shape) */
    const round = new Set();
    for (const bd of boards) { if (obj.has(bd) && obj.get(bd).a < 0.05) continue; [...bd.querySelectorAll('div, span')].forEach((el, j) => { if (el.textContent.trim()) return;
      const cs = getComputedStyle(el); if (/rgba\(.*, 0\)|transparent/.test(cs.backgroundColor)) return; const o = op(el, ops); if (o < 0.05 && wrOf(el) < 0.05) return;
      const r = el.getBoundingClientRect(), key = `MARK b${idx.get(bd)}#${j}`; add(key, o, [r], wrOf(el));
      if (r.width >= 3 && r.width <= 30 && r.height >= 3 && r.height <= 30 && parseFloat(cs.borderTopLeftRadius) >= 0.4 * Math.min(r.width, r.height)) round.add(key); }); }
    const out = {}; for (const [k, u] of U) out[k] = [+u.o.toFixed(3), u.a ? +(u.inA / u.a).toFixed(3) : 0, k.startsWith('MARK') ? +round.has(k) : 1, +u.wr.toFixed(3), +u.h.toFixed(1)]; return out; };
});
if (selftest) await page.evaluate(() => {
  /* two probes on the stage, driven by S02 b2: BAD rises and falls inside the beat, GOOD rises and stays */
  const mk = t => { const d = document.createElement('div'); d.textContent = t; d.style.cssText = 'position:absolute;left:900px;top:500px;font-size:30px;opacity:0'; document.getElementById('stage').appendChild(d); return d; };
  const bad = mk('PROBE BAD'), good = mk('PROBE GOOD'), B = scenes[1].beats[1], f = B.f;
  B.f = p => { f(p); bad.style.opacity = seg(p, .2, .3) * (1 - seg(p, .6, .7)); good.style.opacity = seg(p, .2, .3); };
});

const VIS = u => u && u[0] >= 0.3 && u[1] >= 0.5 && u[4] >= 8;
const inv = await page.evaluate(() => ({ per: scenes.map(s => s.beats.length) }));
const offenders = [], strings = new Set();
let T = 0, frames = 0;
const step = () => page.evaluate(ts => { __step(ts); return { u: __units(), run: run ? run.i : null, p: cur.p }; }, (T += 1000 / FPS));
for (let s = 0; s < inv.per.length; s++) { const id = 'S' + String(s + 1).padStart(2, '0'); if (only && only !== id) continue;
  for (let k = 0; k < inv.per[s]; k++) {
    await page.evaluate(([s, k]) => { if (k > 0) { show(s); applyUpTo(k); } else if (s > 0) show(s - 1, true); else show(0); last = null; }, [s, k]);
    await step();                                             /* the previous beat settled, drawn */
    const start = await page.evaluate(() => { next(); return __units(); });   /* the click: what is drawn, and what the beat's f(0) shows */
    const peak = new Map(); let st;
    do { st = await step(); frames++;
      for (const [key, u] of Object.entries(st.u)) { if (u[0] >= 0.3 && !key.startsWith('MARK')) strings.add(key.replace(/^b\d+:|^x:/, ''));
        if (VIS(u) && u[2] && !VIS(start[key]) && u[3] - (start[key] ? start[key][3] : 0) >= 0.15) {   /* written in this beat: shown further, not just carried into the frame */ const pk = peak.get(key); if (!pk || u[0] > pk[0]) peak.set(key, [u[0], st.p, u[3], start[key]]); } }
    } while (st.run !== null);
    const end = st.u;
    for (const [key, [o, p, w, s0]] of peak) { if (VIS(end[key])) continue;
      const kind = key.startsWith('MARK') ? 'MARK' : end[key] && end[key][0] >= 0.3 && end[key][1] >= 0.5 ? 'SHRUNK' : end[key] && end[key][0] >= 0.3 ? 'LEFT-STAGE' : 'FADE';
      offenders.push(`${id}b${k + 1} ${kind} "${key.replace(/^b\d+:|^x:/, '')}" (peak ${o} at p=${p == null ? 'end' : p.toFixed(2)}; shown ${s0 ? s0[3] : 0} -> ${w})`); } } }
if (strOut) fs.writeFileSync(strOut, [...strings].sort().join('\n') + '\n');
for (const o of offenders) console.log(o);
console.log(`\n${frames} frames at ${FPS} fps: ${offenders.filter(o => !o.includes(' MARK ')).length} transient texts, ${offenders.filter(o => o.includes(' MARK ')).length} transient marks` + (errs.length ? `\nPAGE ERRORS: ${errs.join(' | ')}` : ''));
await b.close();
if (selftest) { const bad = offenders.some(o => o.includes('PROBE BAD')), good = offenders.some(o => o.includes('PROBE GOOD'));
  console.log(`selftest: bad ${bad ? 'flagged' : 'MISSED'}, good ${good ? 'WRONGLY flagged' : 'not flagged'}`); process.exit(bad && !good && !errs.length ? 0 : 1); }
process.exit(offenders.length || errs.length ? 1 : 0);
