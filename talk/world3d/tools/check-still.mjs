// check-still.mjs: once a beat has settled, no text moves (round 26: "even after the animation ends, the text is slightly moving").
// Every beat end is held on a virtual clock (the World's own loop, 20 steps a second) with nothing running, and the drawn place
// of every text on the stage (the box of each text node, not the board's box) is read every few frames. A text MOVES if an edge
// of its ink travels more than --tol px during the hold; it COMES AND GOES if it is drawn in some of the readings only.
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-still.mjs [--only S03] [--hold 12] [--tol 0.05]
//   ... --selftest   two probes at S02 b1: one sways by 2 px (must be flagged), one stands (must not be)
// Read: the opening frame (S01 b0), the end of every beat, and the overview map opened by hand (key o) at the last beat.
// Drawn = opacity 0.05 or more down to the stage, visible, at least 1 px in each direction, touching the stage.
// Not read: the fps counter, the beat id, the notes strip, the practice strip (clocks and help, not the talk's text).
import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH + '/');
const { chromium } = require('playwright');
const argv = process.argv.slice(2), arg = n => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : null; };
const selftest = argv.includes('--selftest'), only = selftest ? 'S02' : arg('--only'), HOLD = Number(arg('--hold') || 12), TOL = Number(arg('--tol') || 0.05);
const DT = 50, WARM = 30, EVERY = 4, N = Math.round(HOLD * 1000 / DT);   /* 50 ms: the engine's largest step, so the ambient clock runs at full rate; 1.5 s first, past the easing of a particle just left */
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
const page = await b.newPage({ viewport: { width: 1920, height: 1080 } });
const errs = []; page.on('pageerror', e => errs.push(String(e)));
await page.addInitScript(() => { let q = []; window.requestAnimationFrame = cb => { q.push(cb); return q.length; }; window.cancelAnimationFrame = () => { };
  window.__step = ts => { const c = q; q = []; c.forEach(f => f(ts)); }; });
await page.goto(new URL('../index.html', import.meta.url).href);
await page.evaluate(() => document.fonts.ready);
await page.evaluate(() => {
  W.composer.render = () => { }; W.renderer.render = () => { };   /* no GPU draw: the camera is still applied, the boards still projected */
  const stage = document.getElementById('stage'), SKIP = new Set(['fps', 'beatid', 'notes', 'practice']);
  const bIdx = new Map(BOARDS.all.map((bd, i) => [bd.el, i])), ids = new WeakMap(), info = new Map(); let seq = 0;
  /* every drawn text node: id -> [left, top, right, bottom] in stage px */
  window.__ink = () => { const out = new Map(), rg = document.createRange();
    const walk = (el, o, bd) => { if (SKIP.has(el.id)) return;
      if (bIdx.has(el)) { if (el.style.visibility === 'hidden') return; bd = el; }
      const cs = getComputedStyle(el); if (cs.display === 'none') return; o *= parseFloat(cs.opacity); if (!(o >= 0.05)) return;
      const vis = cs.visibility !== 'hidden';
      for (const n of el.childNodes) { if (n.nodeType === 1) walk(n, o, bd);
        else if (n.nodeType === 3 && vis && n.nodeValue.trim()) { rg.selectNodeContents(n); const r = rg.getBoundingClientRect();
          if (r.width < 1 || r.height < 1 || r.right < 0 || r.left > 1920 || r.bottom < 0 || r.top > 1080) continue;
          let id = ids.get(n); if (!id) { id = ++seq; ids.set(n, id); info.set(id, { bd: bd ? bIdx.get(bd) : -1, cls: bd ? bd.className.replace(/^board ?/, '') : el.id || el.className || el.tagName, text: n.nodeValue.trim().replace(/\s+/g, ' ').slice(0, 34) }); }
          out.set(id, [r.left, r.top, r.right, r.bottom]); } } };
    walk(stage, 1, null); return out; };
  window.__T = 0;
  /* hold the settled state for n steps, read every `every`: per text, how far each edge travelled and in how many readings it was drawn */
  window.__hold = (warm, n, every, dt) => { for (let i = 0; i < warm; i++) __step(__T += dt);
    const acc = new Map(); let reads = 0;
    const read = () => { reads++; for (const [id, r] of __ink()) { const a = acc.get(id); if (!a) acc.set(id, { lo: r.slice(), hi: r.slice(), n: 1 }); else { a.n++; for (let c = 0; c < 4; c++) { a.lo[c] = Math.min(a.lo[c], r[c]); a.hi[c] = Math.max(a.hi[c], r[c]); } } } };
    read(); for (let i = 1; i <= n; i++) { __step(__T += dt); if (i % every === 0) read(); }
    return { reads, drawn: acc.size, running: run != null, texts: [...acc].map(([id, a]) => ({ ...info.get(id), n: a.n, x: Math.max(a.hi[0] - a.lo[0], a.hi[2] - a.lo[2]), y: Math.max(a.hi[1] - a.lo[1], a.hi[3] - a.lo[3]) })) }; };
});
if (selftest) await page.evaluate(() => {
  const mk = (t, top) => { const d = document.createElement('div'); d.textContent = t; d.style.cssText = `position:absolute;left:900px;top:${top}px;font-size:30px`; document.getElementById('stage').appendChild(d); return d; };
  const sway = mk('PROBE SWAY', 20); mk('PROBE STILL', 60); TICKS.push(t => { sway.style.left = (900 + 2 * Math.sin(t)) + 'px'; });
});

const per = await page.evaluate(() => scenes.map(s => s.beats.length));
const stops = [[0, 0, 'S01 b0 (the opening frame)']];
per.forEach((n, s) => { for (let k = 1; k <= n; k++) stops.push([s, k, `S${String(s + 1).padStart(2, '0')} b${k}`]); });
stops.push([per.length - 1, per[per.length - 1], 'the map (key o)', true]);
let bad = 0, worst = 0, held = 0; const movers = new Map(), flagged = new Set();
for (const [s, k, name, map] of stops) { if (only && !name.startsWith(only)) continue;
  const R = await page.evaluate(([s, k, map, WARM, N, EVERY, DT]) => { show(s); applyUpTo(k); if (map) setOverview(true); return __hold(map ? WARM + 60 : WARM, N, EVERY, DT); }, [s, k, !!map, WARM, N, EVERY, DT]);
  held++;
  const mv = R.texts.filter(t => Math.max(t.x, t.y) > TOL).sort((a, c) => Math.max(c.x, c.y) - Math.max(a.x, a.y)), cg = R.texts.filter(t => t.n < R.reads);
  for (const t of [...mv, ...cg]) flagged.add(t.text);
  if (R.running) { bad++; console.log(`${name}: a beat is still running (the check itself is wrong)`); continue; }
  if (!mv.length && !cg.length) { console.log(`${name}: ${R.drawn} texts, all still`); continue; }
  bad++; if (mv.length) worst = Math.max(worst, Math.max(mv[0].x, mv[0].y));
  console.log(`${name}: ${mv.length} of ${R.drawn} texts move${cg.length ? `, ${cg.length} come and go` : ''}`);
  const byBoard = new Map(); for (const t of mv) { const key = `board ${t.bd} (${t.cls})`, e = byBoard.get(key) || byBoard.set(key, { n: 0, x: 0, y: 0, text: t.text }).get(key); e.n++; e.x = Math.max(e.x, t.x); e.y = Math.max(e.y, t.y); }
  for (const [key, e] of byBoard) { console.log(`    ${key} "${e.text}": ${e.n} texts, ${e.x.toFixed(1)} px across, ${e.y.toFixed(1)} px up and down`); const m = movers.get(key) || movers.set(key, { text: e.text, beats: [], max: 0 }).get(key); m.beats.push(name.replace(/ \(.*/, '')); m.max = Math.max(m.max, e.x, e.y); }
  for (const t of cg.slice(0, 6)) console.log(`    board ${t.bd} (${t.cls}) "${t.text}": drawn in ${t.n} of ${R.reads} readings`);
}
if (selftest) { const ok = flagged.has('PROBE SWAY') && !flagged.has('PROBE STILL');
  console.log(`\nselftest: PROBE SWAY ${flagged.has('PROBE SWAY') ? 'flagged' : 'MISSED'}, PROBE STILL ${flagged.has('PROBE STILL') ? 'FLAGGED' : 'left alone'}: ${ok ? 'PASS' : 'FAIL'}`);
  await b.close(); process.exit(ok && !errs.length ? 0 : 1); }
if (movers.size) { console.log('\nwhat moves, and where:'); for (const [key, m] of [...movers].sort((a, c) => c[1].max - a[1].max)) console.log(`  ${key} "${m.text}": up to ${m.max.toFixed(1)} px, in ${m.beats.length} stops (${m.beats.slice(0, 8).join(' ')}${m.beats.length > 8 ? ' …' : ''})`); }
console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
console.log(`\n${bad} of ${held} stops have text that moves at rest (held ${HOLD} s each, tolerance ${TOL} px${worst ? `, largest ${worst.toFixed(1)} px` : ''})`);
await b.close(); process.exit(bad || errs.length || !held ? 1 : 0);
