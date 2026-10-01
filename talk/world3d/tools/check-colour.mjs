// check-colour.mjs: is every hadron DRAWN colour-neutral, at the end of every beat and at p = .25 .5 .75 inside it?
// Reads the colour each visible quark is painted with (its material), not the charge it was created with.
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-colour.mjs
// Neutral: a colour with its anticolour, or r g b, or sums of those. A sphere may split into two whose colours add up to its own
// (anti-red = green + blue); while the two still overlap they count as the one they came from (listed as SPLIT, not a failure).
// The S07 walker cloud stands for the sampled partner of the quark at its centre: it counts with its own colour (once, while the
// antiquark it replaces is still fading).
import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH + '/'); const { chromium } = require('playwright');
import { installDeckShim } from '/home/gofer/mrs_talk_video/render/deck_shim.mjs';
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
const page = await b.newPage({ viewport: { width: 1920, height: 1080 } });
await page.goto(new URL('../index.html', import.meta.url).href); const inv = await installDeckShim(page); await page.evaluate(() => document.fonts.ready);
const bad = {}, split = {}; let groups = 0;
for (let s = 0; s < inv.scenes; s++) for (let k = 1; k <= inv.per_scene[s]; k++) for (const p of [0.25, 0.5, 0.75, null]) {
  await page.evaluate(([s, k, p]) => { window.__deck.show(s); if (p == null) window.__deck.applyUpTo(k); else { window.__deck.applyUpTo(k - 1); applyState(s, k - 1, p); } }, [s, k, p]);
  await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
  const R = await page.evaluate(() => { const REF = Object.entries(W.QCOL).map(([k, v]) => [k, new THREE.Color(v)]), G = new Map(), h = {}, hr = {};
    const name = c => { c.getHSL(h); let best = '?', bd = 0.04; for (const [k, r] of REF) { r.getHSL(hr); const d = Math.min(Math.abs(h.h - hr.h), 1 - Math.abs(h.h - hr.h)); if (d < bd) { bd = d; best = k; } } return best; };
    const add = (key, q) => (G.get(key) || G.set(key, []).get(key)).push(q);
    W.scene.traverse(o => { if (!o.isMesh || !o.userData.c) return; for (let e = o; e; e = e.parent) if (!e.visible) return; if (o.material.opacity < 0.08) return;
      const wp = new THREE.Vector3(); o.getWorldPosition(wp); const ws = new THREE.Vector3(); o.getWorldScale(ws);
      add(o.parent.userData.id || ('group at ' + o.parent.position.toArray().map(v => v.toFixed(0)).join(',')), { c: name(o.material.color), p: wp, r: ws.x }); });
    if (WALK.pts.visible && WALK.u.uOp.value > 0.08) add('group at ' + PAIR.g.position.toArray().map(v => v.toFixed(0)).join(','), { c: name(WALK.u.uCol.value), p: WALK.pts.position.clone(), r: 0, cloud: true });
    const out = []; for (const [key, all] of G) { const n = { r: 0, g: 0, b: 0 }; let odd = false;
      const qs = all.filter(q => !q.cloud || !all.some(o => !o.cloud && o.c === q.c));   /* while the antiquark dissolves into the cloud the two are one */
      for (const q of qs) { if (q.c === '?') { odd = true; continue; } if (q.c.length === 1) n[q.c]++; else n[q.c[1]]--; }
      const neutral = n.r === n.g && n.g === n.b && !odd, touching = qs.some((a, i) => qs.some((c, j) => j > i && !a.cloud && !c.cloud && a.p.distanceTo(c.p) < a.r + c.r));
      out.push({ key, neutral, touching, what: qs.map(q => q.c + (q.cloud ? ' cloud' : '')).join(' ') }); } return out; });
  const where = `S${String(s + 1).padStart(2, '0')}b${k}` + (p == null ? '' : '@' + p);
  for (const r of R) { groups++; if (r.neutral) continue; const line = `${r.key}: ${r.what}`; ((r.touching ? split : bad)[line] ||= []).push(where); } }
for (const l in split) console.log(`SPLIT  ${l}\n       ${split[l].join(' ')}`);
for (const l in bad) console.log(`FAIL   ${l}\n       ${bad[l].join(' ')}`);
const n = Object.keys(bad).length; console.log(`\n${groups} drawn hadrons checked over ${inv.per_scene.reduce((a, c) => a + c, 0)} beats x 4 moments: ${n ? n + ' not colour-neutral' : 'all colour-neutral'}`);
await b.close(); process.exit(n ? 1 : 0);
