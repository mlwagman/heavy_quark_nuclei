// check-live.mjs: a particle moves only in the beats that are about it (round 20). Prints, per beat, what moves once the
// beat has settled, and fails if a particle moves in a beat that does not show its name, its number or its own state.
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-live.mjs
import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH + '/');
const { chromium } = require('playwright');
import { installDeckShim } from '/home/gofer/mrs_talk_video/render/deck_shim.mjs';
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
const page = await b.newPage({ viewport: { width: 1920, height: 1080 } });
const errs = []; page.on('pageerror', e => errs.push(String(e)));
await page.goto(new URL('../index.html', import.meta.url).href); await installDeckShim(page); await page.evaluate(() => document.fonts.ready);
const settle = (s, k) => page.evaluate(([s, k]) => { window.__deck.show(s); window.__deck.applyUpTo(k); }, [s, k]);
/* every visible particle's quark positions (its own frame), now */
const pose = () => page.evaluate(() => { const o = {};
  for (const id in SHELF.glyphs) { const g = SHELF.glyphs[id].g; if (g.visible && (g.userData.op ?? 0) > 0.05) o[id] = g.userData.quarks.flatMap(q => q.position.toArray()); }
  if (PAIR.st.opacity > 0.05) o.pair = [...PAIR.A.position.toArray(), ...PAIR.B.position.toArray()];
  const h = window.__S01.hero; if (h.visible && (h.userData.op ?? 1) > 0.05) o.hero = h.userData.quarks.flatMap(q => q.position.toArray());
  return o; });
const wait = ms => page.evaluate(ms => new Promise(r => setTimeout(() => requestAnimationFrame(() => requestAnimationFrame(r)), ms)), ms);
const moved = (a, c) => Object.keys(a).filter(id => c[id] && a[id].some((x, i) => Math.abs(x - c[id][i]) > 0.004));
const nB = await page.evaluate(() => scenes.map(s => s.beats.length));
const T = {}; const still = {};
for (let s = 0; s < nB.length; s++) for (let k = 1; k <= nB[s]; k++) {
  await settle(s, k); await wait(700);                       /* past the easing of a particle that has just been left */
  const a = await pose(); await wait(350); const c = await pose(); const m = moved(a, c);
  T[`S${String(s + 1).padStart(2, '0')}b${k}`] = { m, n: Object.keys(a).length }; still[`${s},${k}`] = a;
  console.log(`S${String(s + 1).padStart(2, '0')} b${k}: ${Object.keys(a).length} shown, moving: ${m.join(' ') || 'none'}`);
}
const R = []; const ok = (name, pass, got) => { R.push(pass); console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}  (${got})`); };
const mv = id => T[id].m;
const has = (id, ...ps) => ps.every(x => mv(id).includes(x)), show = id => `${id}: ${mv(id).join(' ') || 'none'}`;
/* a particle keeps moving for as long as the talk is about it, not only in the beat that writes its name (round 20b) */
ok('S01: the title baryon turns', has('S01b1', 'hero'), show('S01b1'));
ok('S02 b1 to b3 (the four families, their scales, the precision laboratory): the four move', ['S02b1', 'S02b2', 'S02b3'].every(id => has(id, 'cc', 'bb', 'Occc', 'T4')), ['S02b1', 'S02b2', 'S02b3'].map(show).join('; '));
ok('S02 b5 (the obstacle board): no particle moves', mv('S02b5').length === 0, show('S02b5'));
ok('S03 b1 and b7 (the pair at the QCD scale; the static limit): the pair moves', has('S03b1', 'pair') && has('S03b7', 'pair'), [show('S03b1'), show('S03b7')].join('; '));
ok('S04 b3 (the symptom board): the pair in the corner is still', !has('S04b3', 'pair'), show('S04b3'));
ok('S08 b3 to b5 (tuning the charm and bottom masses): cc and bb move, nothing else', ['S08b3', 'S08b4', 'S08b5'].every(id => has(id, 'cc', 'bb') && mv(id).length === 2), ['S08b3', 'S08b4', 'S08b5'].map(show).join('; '));
ok('S09 b3 (the two tuned masses meet): cc and bb move; b4 (B_c against experiment): bc moves', has('S09b3', 'cc', 'bb') && has('S09b4', 'bc'), [show('S09b3'), show('S09b4')].join('; '));
ok('S10 b1 and b2 (our row, the lattice row): the four Ω move, the mesons beside them do not', ['S10b1', 'S10b2'].every(id => has(id, 'Occc', 'Occb', 'Ocbb', 'Obbb') && !has(id, 'cc') && !has(id, 'bb')), [show('S10b1'), show('S10b2')].join('; '));
ok('S10 b3 and b4 (Ω_bbb on its number line, its scale band): Ω_bbb alone moves', ['S10b3', 'S10b4'].every(id => JSON.stringify(mv(id)) === '["Obbb"]'), [show('S10b3'), show('S10b4')].join('; '));
ok('S11, S13, S15 (plots, table, references): nothing moves', Object.keys(T).filter(id => /^S1[135]/.test(id)).every(id => mv(id).length === 0), Object.keys(T).filter(id => /^S1[135]/.test(id) && mv(id).length).join(' ') || 'none moving');
ok('S14 b3 (the whole family of predictions): every particle on the shelf moves', mv('S14b3').length >= 15 && mv('S14b3').length === T.S14b3.n, `${mv('S14b3').length} of ${T.S14b3.n}`);
ok('S14 b1 and b2 (first use; scale sensitivity): nothing moves', mv('S14b1').length === 0 && mv('S14b2').length === 0, [show('S14b1'), show('S14b2')].join('; '));
/* a still particle has one pose: the same on a second visit, whatever the ambient clock says by then */
await settle(1, 5); await wait(700); const again = await pose(); const first = still['1,5'];
const drift = Object.keys(first).filter(id => again[id] && first[id].some((x, i) => Math.abs(x - again[id][i]) > 0.004));
ok('S02 b5 visited again later: every still particle is in the pose it had the first time', Object.keys(first).length > 0 && drift.length === 0, drift.join(' ') || `${Object.keys(first).length} particles, same pose`);
console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
console.log(`${R.filter(Boolean).length} pass, ${R.filter(x => !x).length} fail`);
await b.close(); process.exit(R.every(Boolean) && !errs.length ? 0 : 1);
