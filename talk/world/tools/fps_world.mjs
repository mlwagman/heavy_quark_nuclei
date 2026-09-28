#!/usr/bin/env node
// fps_world.mjs — does a rebuilt World scene hold the 30 fps floor (ticket 03) in a real stage browser? (ticket 06)
//   NODE_PATH=$(npm root -g) node talk/world/tools/fps_world.mjs --browser chrome|firefox-stock [--page world.html]
//       [--scenes 4,5] [--warm 2] [--meas 10] --out DIR          headed, kiosk, AC power, hands off; dark scheme
//   node talk/world/tools/fps_world.mjs --selftest                no browser
// Ticket 03's meter, ported from speedtest.py/speed.js: every rAF timestamp; fps = frames / time over --meas s after
// --warm s; "holds 30" = mean ≥ 29.5 fps and p95 frame time ≤ 34.8 ms. The run is void unless the meter passes its
// own calibration: cal-empty (hosted S01, no 3D) ≥ 55 fps and 3D not drawn, cal-burn40 (40 ms busy per frame)
// 18-26 fps, viewport 1920x1080, an unmasked hardware GPU, every 3D row lit, and the page dark.
// Rows: the idle state of every beat of the rebuilt scenes, every beat's tween replayed in a loop, and the Deck glides
// into, between and out of them. Writes DIR/results.json + results.md. Exit 0 all hold, 1 a row fails or void, 2 setup.
import fs from 'node:fs';
import path from 'node:path';
import { launch, glideEnd, SOFTWARE_GL } from './world_test_lib.mjs';

/* sustained floor = mean fps at the floor and 95 % of frames no slower than one floor frame (+1.5 ms) — speedtest.py */
export const holds = (r, floor) => r.fps >= floor - 0.5 && r.p95 <= 1000 / floor + 1.5;
export function calibrationErrors(rows) {
  const errs = rows.filter(r => r.error).map(r => `${r.browser} ${r.id}: page error: ${r.error}`);
  for (const b of [...new Set(rows.map(r => r.browser))].sort()) {
    const by = Object.fromEntries(rows.filter(r => r.browser === b && !r.error).map(r => [r.id, r]));
    const empty = by['cal-empty'], burn = by['cal-burn40'];
    if (!empty || !burn) { errs.push(`${b}: calibration rows missing`); continue; }
    if (empty.vw !== 1920 || empty.vh !== 1080) errs.push(`${b}: viewport ${empty.vw}x${empty.vh}, not 1920x1080`);
    if (SOFTWARE_GL.test(empty.gpu)) errs.push(`${b}: software WebGL (${empty.gpu})`);
    for (const r of Object.values(by)) {
      if (r.threeD && !(r.lit > 0.001)) errs.push(`${b} ${r.id}: only ${(100 * (r.lit || 0)).toFixed(2)} % of pixels lit: the 3D did not draw`);
      if (r.dark !== true) errs.push(`${b} ${r.id}: not dark (${r.bg})`);
    }
    if (!(empty.lit === 0)) errs.push(`${b}: empty scene reads ${(100 * empty.lit).toFixed(2)} % lit: the pixel check counts background`);
    if (empty.fps < 55) errs.push(`${b}: empty scene reads ${empty.fps.toFixed(1)} fps (< 55)`);
    if (!(burn.fps >= 18 && burn.fps <= 26)) errs.push(`${b}: 40 ms/frame burn reads ${burn.fps.toFixed(1)} fps (outside 18-26)`);
  }
  return errs;
}
function selftest() {
  const ok = (c, m) => { if (!c) throw new Error('selftest: ' + m); };
  const good = [{ browser: 'x', id: 'cal-empty', vw: 1920, vh: 1080, gpu: 'AMD Radeon (radeonsi)', fps: 59.9, lit: 0, dark: true },
    { browser: 'x', id: 'cal-burn40', vw: 1920, vh: 1080, gpu: 'AMD', fps: 20.0, lit: 0, dark: true }];
  ok(calibrationErrors(good).length === 0, 'good calibration passes');
  ok(calibrationErrors([good[0], { ...good[1], fps: 59 }]).length, 'a burn that costs nothing must fail');
  ok(calibrationErrors([{ ...good[0], gpu: 'ANGLE (Google, Vulkan (SwiftShader))' }, good[1]]).length, 'software GL must fail');
  ok(calibrationErrors([{ ...good[0], vw: 1280 }, good[1]]).length, 'wrong viewport must fail');
  ok(calibrationErrors(good.slice(0, 1)).length, 'missing calibration row must fail');
  ok(calibrationErrors([...good, { browser: 'x', id: 'S04b1-idle', error: 'boom' }]).length, 'page error must fail');
  ok(calibrationErrors([{ ...good[0], gpu: 'Radeon HD 3200 Graphics, or similar' }, good[1]]).length, 'masked GPU must fail');
  ok(calibrationErrors([{ ...good[0], lit: 1 }, good[1]]).length, 'a checker that counts background must fail');
  ok(calibrationErrors([...good, { browser: 'x', id: 'S04b1-idle', threeD: true, lit: 0, dark: true }]).length, 'a blank canvas must fail');
  ok(!calibrationErrors([...good, { browser: 'x', id: 'S04b1-idle', threeD: true, lit: 0.4, dark: true }]).length, 'a drawn canvas passes');
  ok(calibrationErrors([...good, { browser: 'x', id: 'S04b1-idle', threeD: true, lit: 0.4, dark: false, bg: 'rgb(246, 247, 244)' }]).length, 'a light page must fail');
  ok(holds({ fps: 30.0, p95: 33.4 }, 30) && !holds({ fps: 45.0, p95: 50.0 }, 30) && !holds({ fps: 29.0, p95: 33 }, 30), 'holds');
  console.log('selftest OK');
}

const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
if (args.includes('--selftest')) { selftest(); process.exit(0); }
const KEY = opt('--browser', 'chrome'), PAGE = opt('--page', 'world.html'), OUT = opt('--out');
const WARM = +opt('--warm', 2), MEAS = +opt('--meas', 10), SCENES = opt('--scenes', '4,5').split(',').map(Number);
const KIOSK = { chrome: ['--kiosk'], chromium: ['--kiosk'], 'firefox-stock': ['-kiosk'], firefox: ['-kiosk'] }[KEY];

// the meter, in the page: every rAF timestamp; an optional busy wait per frame; stats as speed.js
const METER = () => {
  window.__ts = []; window.__burn = 0; window.__loop = null;
  requestAnimationFrame(function f(now) {
    if (window.__burn) { const end = performance.now() + window.__burn; while (performance.now() < end); }
    window.__ts.push(now); if (window.__loop) window.__loop();
    requestAnimationFrame(f);
  });
  window.__stats = (a, b) => {
    const t = window.__ts.filter(x => x >= a && x <= b);
    if (t.length < 2) return { frames: t.length, fps: 0, p50: 99999, p95: 99999, p99: 99999, max: 99999, over34: 1 };
    const dt = t.slice(1).map((x, i) => x - t[i]).sort((x, y) => x - y), pct = p => dt[Math.min(dt.length - 1, Math.floor(p * dt.length))];
    return { frames: t.length, fps: (t.length - 1) * 1000 / (t.at(-1) - t[0]), p50: pct(0.5), p95: pct(0.95), p99: pct(0.99), max: dt.at(-1),
      over34: dt.filter(d => d > 34.8).length / dt.length };
  };
  // proof the GPU drew: render once more and read the canvas in the same task (a black or bg-only frame reads 0 %)
  window.__lit = () => {
    if (!(window.__rebuiltSpace && __rebuiltSpace.on) || !WorldSpace.renderer) return 0;
    WorldSpace.render();
    const gl = WorldSpace.renderer.getContext(), W = gl.drawingBufferWidth, H = gl.drawingBufferHeight, buf = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    let n = 0, m = 0;
    for (let y = 0; y < H; y += 4) for (let x = 0; x < W; x += 4) { const i = (y * W + x) * 4; m++; n += buf[i] + buf[i + 1] + buf[i + 2] > 10 + 14 + 23 + 30; }
    return n / m;
  };
};

const rows = [];
let rc = 0;
const browser = await launch(KEY, { scheme: 'dark', args: KIOSK });
try {
  const ctx = await browser.newContext({ viewport: null, colorScheme: 'dark' }); // kiosk window = the real 1920x1080 screen
  await ctx.addInitScript(METER);
  const pg = await ctx.newPage(), errors = [];
  pg.on('pageerror', e => errors.push(String(e)));
  await pg.goto('file://' + path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', PAGE));
  await pg.waitForFunction(() => { try { return scenes.length === 15; } catch (e) { return false; } });
  // Chrome under Playwright ignores --kiosk (the page opens in a 901x892 window, lock O): go full screen through the
  // DevTools window API; the calibration's 1920x1080 check still decides whether the run counts
  if (/^(chrome|chromium)$/.test(KEY)) {
    const cdp = await ctx.newCDPSession(pg), { windowId } = await cdp.send('Browser.getWindowForTarget');
    await cdp.send('Browser.setWindowBounds', { windowId, bounds: { windowState: 'fullscreen' } });
    await pg.waitForFunction(() => innerWidth === screen.width && innerHeight === screen.height, null, { timeout: 5000 }).catch(() => {});
  }
  const env = await pg.evaluate(() => ({ vw: innerWidth, vh: innerHeight, ua: navigator.userAgent }));
  const gpuName = () => pg.evaluate(() => { const R = WorldSpace.renderer; if (!R) return ''; const gl = R.getContext(), x = gl.getExtension('WEBGL_debug_renderer_info');
    return x ? gl.getParameter(x.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); });
  const measure = async (id, setup, { threeD = false, burn = 0, loop = null } = {}) => {
    await pg.evaluate(setup.fn, setup.arg);
    await pg.evaluate(([burn, loop]) => { window.__burn = burn; window.__loop = loop ? new Function(loop) : null; }, [burn, loop]);
    await pg.waitForTimeout(WARM * 1000);
    const t0 = await pg.evaluate(() => performance.now());
    await pg.waitForTimeout(MEAS * 1000);
    const r = await pg.evaluate(t0 => ({ ...__stats(t0, performance.now()), lit: __lit(), dark: matchMedia('(prefers-color-scheme: dark)').matches && getComputedStyle(document.body).backgroundColor === 'rgb(10, 14, 23)', bg: getComputedStyle(document.body).backgroundColor }), t0);
    await pg.evaluate(() => { window.__burn = 0; window.__loop = null; });
    const row = { browser: KEY, id, threeD, ...env, gpu: '', ...r, holds30: holds(r, 30) };
    rows.push(row);
    console.log(`${id.padEnd(22)} fps ${r.fps.toFixed(1).padStart(5)}  p95 ${r.p95.toFixed(1).padStart(6)} ms  lit ${(100 * r.lit).toFixed(2)} %  ${row.holds30 ? 'holds' : 'NO'}`);
  };
  const at = (i, k) => ({ fn: ([i, k]) => { if (si !== i) { show(i); } applyUpTo(k); }, arg: [i, k] });
  await measure('cal-empty', at(0, 0));
  await measure('cal-burn40', at(0, 0), { burn: 40 });
  // glide into the first rebuilt scene from its hosted neighbour, back and forth for the whole window
  const glideLoop = (a, b) => `if (glTimer === null) { show(si === ${a} ? ${b} : ${a}); }`;
  for (const n of SCENES) {
    const i = n - 1, nb = await pg.evaluate(i => scenes[i].beats.length, i);
    if (n === SCENES[0]) await measure(`glide-S${String(n - 1).padStart(2, '0')}-S${String(n).padStart(2, '0')}`, at(i - 1, 0), { threeD: true, loop: glideLoop(i - 1, i) });
    await pg.evaluate(i => { show(i); applyUpTo(0); }, i); await glideEnd(pg);
    for (let k = 0; k <= nb; k++) await measure(`S${String(n).padStart(2, '0')}b${k}-idle`, at(i, k), { threeD: true });
    for (let k = 1; k <= nb; k++) await measure(`S${String(n).padStart(2, '0')}b${k}-tween`, at(i, k - 1),
      { threeD: true, loop: `if (run === null) { applyUpTo(${k - 1}); next(); }` });
    await measure(`glide-S${String(n).padStart(2, '0')}-S${String(n + 1).padStart(2, '0')}`, at(i, nb), { threeD: true, loop: glideLoop(i, i + 1) });
  }
  const gpu = await gpuName();
  for (const r of rows) r.gpu = gpu;
  if (errors.length) rows.push({ browser: KEY, id: 'page', error: errors.join(' | ') });
  const errs = calibrationErrors(rows), fails = rows.filter(r => !r.error && !r.id.startsWith('cal-') && !r.holds30);
  fs.mkdirSync(OUT, { recursive: true });
  fs.writeFileSync(path.join(OUT, `results-${KEY}.json`), JSON.stringify(rows, null, 1));
  const md = [`### ${KEY}`, '', `GPU \`${gpu}\` · viewport ${env.vw}x${env.vh} · page ${PAGE} · dark · warm ${WARM} s, measured ${MEAS} s`, `UA \`${env.ua}\``, '',
    '| row | fps | p50 ms | p95 ms | p99 ms | max ms | frames > 34.8 ms | lit | holds 30 |', '|---|---|---|---|---|---|---|---|---|',
    ...rows.filter(r => !r.error).map(r => `| ${r.id} | ${r.fps.toFixed(1)} | ${r.p50.toFixed(1)} | ${r.p95.toFixed(1)} | ${r.p99.toFixed(1)} | ${r.max.toFixed(1)} | ${(100 * r.over34).toFixed(1)} % | ${(100 * r.lit).toFixed(2)} % | ${r.id.startsWith('cal-') ? '(calibration)' : r.holds30 ? 'yes' : 'NO'} |`),
    '', errs.length ? `**METER UNTRUSTED**: ${errs.join('; ')}` : 'calibration OK', fails.length ? `**FAILING ROWS**: ${fails.map(r => r.id).join(', ')}` : 'every row holds 30', ''].join('\n');
  fs.appendFileSync(path.join(OUT, 'results.md'), md + '\n');
  console.log(md);
  rc = errs.length || fails.length ? 1 : 0;
} catch (e) { console.log('SETUP_ERROR ' + (e.stack || e)); rc = 2; }
await browser.close();
process.exit(rc);
