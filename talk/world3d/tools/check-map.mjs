// check-map.mjs: at the end of the talk, does every scene have its own place on the map? Measured on the screen, on the map
// frames themselves (S13 b8 settled, S14 b1 at p = .1, the manual overview after S14; round 24e: 14 scenes), not on world boxes.
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-map.mjs
// A scene's region = the box around its drawn boards and its badge; boards without a scene and the shelf glyphs join the nearest
// region (one left between two scenes merges them). Two regions closer than 8 px, a region on a legend or off the stage fail.
// The shelf: its place at the end of each scene that uses it (S02, S08, S09, S10, S12, S13) must be 6 distinct places.
// Prints the World's box and the overview camera distance that fits it between the legends.
import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH + '/');
const { chromium } = require('playwright');
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
const page = await b.newPage({ viewport: { width: 1920, height: 1080 } });
const errs = []; page.on('pageerror', e => errs.push(String(e)));
await page.goto(new URL('../index.html', import.meta.url).href);
await page.evaluate(() => document.fonts.ready);
const settle = () => page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
const CLEAR = 8;

const regions = () => page.evaluate(() => {
  /* scene boards and badges make the regions; a board without a scene (the shelf's labels and chips, the gauge, the |term| dots ...)
     and the shelf glyphs join the nearest region, so one left between two scenes merges them */
  const R = {}, grow = (s, r) => { const o = R[s] || (R[s] = { l: 1e9, r: -1e9, t: 1e9, b: -1e9, n: 0 });
    o.l = Math.min(o.l, r.left); o.r = Math.max(o.r, r.right); o.t = Math.min(o.t, r.top); o.b = Math.max(o.b, r.bottom); o.n++; };
  const drawn = e => e.style.visibility === 'visible' && (parseFloat(e.style.opacity) || 0) >= 0.05, loose = [];
  for (const bd of BOARDS.all) { if (!drawn(bd.el)) continue; const r = bd.el.getBoundingClientRect(); if (r.width < 1 || r.height < 1) continue;
    if (bd.scene != null) grow(bd.scene, r); else loose.push(r); }
  scenes.forEach((sc, i) => { if (sc.badge && drawn(sc.badge.el)) grow(i, sc.badge.el.getBoundingClientRect()); });
  for (const G of Object.values(SHELF.glyphs)) { if (!G.g.visible) continue; const p = W.project(G.g.getWorldPosition(new THREE.Vector3())); if (p.behind) continue;
    loose.push({ left: p.x - 12, right: p.x + 12, top: p.y - 12, bottom: p.y + 12 }); }
  const own = Object.entries(R).map(([s, o]) => [s, { ...o }]);
  for (const r of loose) { const cx = (r.left + r.right) / 2, cy = (r.top + r.bottom) / 2; let best = null, bd = 1e18;
    for (const [s, o] of own) { const dx = Math.max(o.l - cx, 0, cx - o.r), dy = Math.max(o.t - cy, 0, cy - o.b), d = dx * dx + dy * dy; if (d < bd) { bd = d; best = s; } }
    if (best != null) grow(best, r); }
  const legends = [...document.querySelectorAll('.legend')].filter(l => parseFloat(getComputedStyle(l).opacity) > 0.05).map(l => l.getBoundingClientRect()).map(r => ({ l: r.left, r: r.right, t: r.top, b: r.bottom }));
  return { R, legends };
});

let bad = 0;
const judge = (name, { R, legends }) => {
  const ids = Object.keys(R).map(Number).sort((a, c) => a - c), S = i => 'S' + String(i + 1).padStart(2, '0'), out = [];
  for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) { const A = R[ids[i]], B = R[ids[j]];
    const w = Math.min(A.r, B.r) - Math.max(A.l, B.l) + CLEAR, h = Math.min(A.b, B.b) - Math.max(A.t, B.t) + CLEAR;
    if (w > 0 && h > 0) out.push(`${S(ids[i])}/${S(ids[j])} overlap ${Math.round(w)}x${Math.round(h)} px`); }
  for (const s of ids) { const A = R[s];
    for (const L of legends) if (Math.min(A.r, L.r) - Math.max(A.l, L.l) > 0 && Math.min(A.b, L.b) - Math.max(A.t, L.t) > 0) out.push(`${S(s)} on a legend`);
    if (A.l < 0 || A.t < 0 || A.r > 1920 || A.b > 1080) out.push(`${S(s)} off the stage (${Math.round(A.l)},${Math.round(A.t)})-(${Math.round(A.r)},${Math.round(A.b)})`); }
  console.log(`${name}: ${ids.length} regions, ${out.length} problems`); for (const o of out) console.log('  ' + o); bad += out.length;
  if (process.argv.includes('--boxes')) for (const s of ids) { const A = R[s]; console.log(`    ${S(s)} x ${Math.round(A.l)}..${Math.round(A.r)}  y ${Math.round(A.t)}..${Math.round(A.b)}  (${A.n} parts)`); } };

await page.evaluate(() => { show(12); applyUpTo(8); }); await settle(); await settle(); judge('S13 b8 settled (the map)', await regions());
await page.evaluate(() => { show(13); applyUpTo(0); applyState(13, 0, 0.1); }); await settle(); await settle(); judge('S14 b1 p=.1', await regions());
await page.evaluate(() => { show(13, true); OV.on = true; OV.k = 1; }); await settle(); await settle(); judge('overview after S14', await regions());

/* the shelf's place at the end of each scene that uses it */
const places = [];
for (const s of [1, 7, 8, 9, 11, 12]) {   /* the shelf's scenes; the conclusion is scene 12 since round 24e */ const p = await page.evaluate(s => { OV.on = false; OV.k = 0; show(s, true); const v = SHELF.at ? SHELF.at.clone() : SHELF.glyphs.cc.g.getWorldPosition(new THREE.Vector3()); return [Math.round(v.x / 20) * 20, Math.round(v.y / 20) * 20];   /* the shelf's origin (r17: its first glyph, which the layouts shift a little) */ }, s); places.push(p.join(',')); }
const distinct = new Set(places).size; console.log(`shelf: ${distinct} distinct places (want 6): ${places.join(' | ')}`); if (distinct < 6) bad++;

/* the World's box (every board anchor) and the overview camera that fits it between the legends */
const box = await page.evaluate(() => { let l = 1e9, r = -1e9, t = -1e9, bt = 1e9; for (const bd of BOARDS.all) { if (bd.noLayout) continue; const hw = bd.w / 2, hh = (bd.h || 10) * bd.w / bd.px / 2;
  l = Math.min(l, bd.anchor.x - hw); r = Math.max(r, bd.anchor.x + hw); bt = Math.min(bt, bd.anchor.y - hh); t = Math.max(t, bd.anchor.y + hh); } return { l, r, t, b: bt }; });
const legW = await page.evaluate(() => Math.max(...[...document.querySelectorAll('.legend')].map(l => l.getBoundingClientRect().width)));
const fit = (units, px) => units * 1080 / (2 * 0.3443 * px), D = Math.max(fit(box.r - box.l, 1920 - 2 * legW - 60), fit(box.t - box.b, 1000));
console.log(`World box x ${box.l.toFixed(0)}..${box.r.toFixed(0)}  y ${box.b.toFixed(0)}..${box.t.toFixed(0)}; overview camera to fit: [${((box.l + box.r) / 2).toFixed(1)}, ${((box.t + box.b) / 2).toFixed(1)}, ${D.toFixed(0)}]`);
console.log(bad ? `\n${bad} map problems` : '\nmap clean') ; if (errs.length) console.log('PAGE ERRORS: ' + errs.join(' | '));
await b.close(); process.exit(bad || errs.length ? 1 : 0);
