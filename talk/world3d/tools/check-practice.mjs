// check-practice.mjs: the phone practice file (round 22). One self-contained HTML (talk/mrs-talk-practice.html, built by
// tools/build-practice.py) opened on an emulated phone, portrait and landscape: it loads nothing else, taps drive it (right two
// thirds forward, left third back), and under the World it shows the current beat's keynotes, readable.
//   python3 talk/world3d/tools/build-practice.py && NODE_PATH=$(npm root -g) node talk/world3d/tools/check-practice.mjs
import fs from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH + '/');
const { chromium, devices } = require('playwright');
const FILE = new URL('../../mrs-talk-practice.html', import.meta.url);
let v2; const R = []; const ok = (name, pass, got) => { R.push(pass); console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}  (${got})`); };
if (!fs.existsSync(FILE)) { ok('talk/mrs-talk-practice.html exists', false, 'missing'); process.exit(1); }
/* review 5.5: the phone file must be built from the current sources (build-practice.py stamps their hash) */
{ const { createHash } = await import('node:crypto'); const W = new URL('../', import.meta.url), h = createHash('sha256');
  for (const f of ['index.html', ...fs.readdirSync(new URL('css/', W)).map(x => 'css/' + x), ...fs.readdirSync(new URL('js/', W)).sort().map(x => 'js/' + x)].sort()) h.update(fs.readFileSync(new URL(f, W)));
  const want = h.digest('hex').slice(0, 16), got = (/<meta name="src-hash" content="([0-9a-f]+)">/.exec(fs.readFileSync(FILE, 'utf8')) || [0, 'none'])[1];
  ok('the phone file is built from the current sources', got === want, `file ${got}, sources ${want}`); }
const KEYS = new Function(fs.readFileSync(new URL('../js/keys.js', import.meta.url), 'utf8') + '; return KEYS;')();
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
for (const [name, vp] of [['portrait', { width: 412, height: 915 }], ['landscape', { width: 915, height: 412 }]]) {
  const ctx = await b.newContext({ ...devices['Pixel 7'], viewport: vp }); const bad = [];
  ctx.on('request', r => { if (r.url() !== FILE.href && !r.url().startsWith('data:') && !r.url().startsWith('blob:')) bad.push(r.url()); });
  const page = await ctx.newPage(); page.on('pageerror', e => bad.push(String(e)));
  await page.goto(FILE.href); await page.waitForTimeout(2500);
  const tap = async x => { await page.touchscreen.tap(x, vp.height * 0.3); await page.waitForTimeout(300); await page.evaluate(() => { if (run) run.t = 1e9; }); await page.waitForTimeout(400); };
  for (let i = 0; i < 4; i++) await tap(vp.width * 0.8);
  let s = await page.evaluate(() => ({ si, done }));
  ok(`${name}: four taps on the right go four beats forward`, s.si === 1 && s.done === 2, `scene ${s.si + 1}, beat ${s.done}`);
  await tap(vp.width * 0.1); s = await page.evaluate(() => ({ si, done }));
  ok(`${name}: a tap on the left third goes one beat back`, s.si === 1 && s.done === 1, `scene ${s.si + 1}, beat ${s.done}`);
  const v = await page.evaluate(() => { const p = document.getElementById('practice'); if (!p) return null; const st = document.getElementById('stage').getBoundingClientRect(), r = p.getBoundingClientRect();
    return { li: [...p.querySelectorAll('li')].map(e => e.textContent), fs: Math.min(...[...p.querySelectorAll('li')].map(e => parseFloat(getComputedStyle(e).fontSize))), where: (p.querySelector('.where') || {}).textContent || '', stageBottom: st.bottom, stageW: st.width, top: r.top, bottom: r.bottom, H: innerHeight }; });
  if (!v) { ok(`${name}: a practice strip`, false, 'no #practice element'); await ctx.close(); continue; }
  ok(`${name}: the strip shows the keynotes of the current beat (S02 b1)`, JSON.stringify(v.li) === JSON.stringify(KEYS[1][0].k), JSON.stringify(v.li));
  ok(`${name}: keynotes at 14 px or more, the beat and the target time shown`, v.fs >= 14 && /2\s*\/\s*14/.test(v.where) && /target 0:31/.test(v.where), `${v.fs}px, "${v.where}"`);
  ok(`${name}: the World sits above the strip, the strip on the screen`, v.stageBottom <= v.top + 1 && v.bottom <= v.H + 1 && v.stageW >= vp.width * 0.6, `stage bottom ${v.stageBottom.toFixed(0)}, strip ${v.top.toFixed(0)}..${v.bottom.toFixed(0)} of ${v.H}, stage width ${v.stageW.toFixed(0)}`);
  await page.evaluate(() => { show(5); applyUpTo(1); }); await page.waitForTimeout(1200);
  v2 = await page.evaluate(() => document.querySelectorAll('#practice [data-v]').length);
  ok(`${name}: no version buttons on panel 6 (the gauge was chosen)`, v2 === 0, `${v2} buttons`);
  ok(`${name}: nothing loaded besides the file itself, no page error`, bad.length === 0, bad.slice(0, 3).join(' | ') || 'none');
  await ctx.close();
}
await b.close();
console.log(`${R.filter(Boolean).length} pass, ${R.filter(x => !x).length} fail`); process.exit(R.every(Boolean) ? 0 : 1);
