// test_world_browser.mjs — browser tests for the World shell (ticket 04), one real browser per run.
//   BROWSER=chromium|firefox|chrome|firefox-stock  (keys of stage_browsers.json; chrome/firefox-stock are headed)
//   WORLD_DIR=<dir with world.html> (default: talk/world)   WORLD_PAGE=<page> (default world.html; the negative
//   controls point it at index.html or a broken copy)   REQUIRE_HW_GL=1 fails X3 on a software/masked renderer.
//   NODE_PATH=$(npm root -g) BROWSER=firefox node --test --test-reporter=tap talk/world/tools/test_world_browser.mjs
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { shot, pxDiff } from './world_test_lib.mjs';
const require = createRequire((process.env.NODE_PATH || '/usr/lib/node_modules') + '/');
const pw = require('playwright');
const here = path.dirname(new URL(import.meta.url).pathname);
const KEY = process.env.BROWSER || 'chromium';
const { type, launch } = JSON.parse(fs.readFileSync(path.join(here, 'stage_browsers.json'), 'utf8'))[KEY];
const WD = path.resolve(process.env.WORLD_DIR || path.join(here, '..'));
const PAGE = process.env.WORLD_PAGE || 'world.html';
const SOFTWARE_GL = /swiftshader|llvmpipe|softpipe|software|basic render|or similar/i; // ticket 03's speedtest.py
const url = (p, q = '') => 'file://' + path.join(WD, p) + q;
const wait = ms => new Promise(r => setTimeout(r, ms));

let browser;
before(async () => { browser = await pw[type].launch(launch); });
after(async () => { await browser?.close(); });

async function openWorld(q = '', page = PAGE, dir = WD) {
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const pg = await ctx.newPage(), errors = [], requests = [];
  pg.on('pageerror', e => errors.push(String(e)));
  pg.on('request', r => requests.push(r.url()));
  await pg.goto('file://' + path.join(dir, page) + q);
  await pg.waitForFunction(() => { try { return scenes.length === 15; } catch (e) { return false; } });
  return { ctx, pg, errors, requests };
}
const state = pg => pg.evaluate(() => ({ si, done, run: run && run.i, auto, notesOn, audPaused: AUD.paused }));
const settle = pg => pg.waitForFunction(() => run === null, null, { timeout: 15000 });

// shot / pxDiff: the 0 px recipes, shared with the ticket 06 tools (world_test_lib.mjs)

test('K1 idle: nothing auto-plays', async () => {
  const { ctx, pg, errors } = await openWorld();
  await wait(3000);
  assert.deepEqual(await state(pg), { si: 0, done: 0, run: null, auto: false, notesOn: false, audPaused: true });
  assert.deepEqual(errors, []); await ctx.close();
});

test('K2 A (narrated auto-run) and N (notes rail) do nothing in the World', async () => {
  const { ctx, pg } = await openWorld();
  // one press per key and a check after each: in the Deck a second press toggles back, which would hide the first
  for (const k of ['a', 'N']) {
    await pg.keyboard.press(k); await wait(300);
    assert.deepEqual(await state(pg), { si: 0, done: 0, run: null, auto: false, notesOn: false, audPaused: true }, `after ${k}`);
  }
  await ctx.close();
});

test('K3 each clicker key moves exactly one beat', async () => {
  const { ctx, pg } = await openWorld();
  await pg.evaluate(() => { window.MOTION = 'instant'; show(1); applyUpTo(0); }); // S02 has 5 beats
  const seq = [[' ', 1], ['ArrowRight', 2], ['PageDown', 3], ['ArrowLeft', 2], ['PageUp', 1]];
  for (const [k, want] of seq) {
    await pg.keyboard.press(k); await settle(pg);
    const s = await state(pg);
    assert.deepEqual([s.si, s.done], [1, want], `after ${JSON.stringify(k)}`);
  }
  await ctx.close();
});

test('O1 offline: every request is file:, data: or blob:', async () => {
  const { ctx, pg, requests } = await openWorld();
  await pg.keyboard.press('ArrowRight'); await wait(1000);
  // Playwright's Firefox sees no requests (nor resource timing) for file:// pages: there O1 audits every URL the page
  // references (src/href attributes, CSS url()s, the narration audio). Chromium checks both.
  requests.push(...await pg.evaluate(() => [
    ...[...document.querySelectorAll('[src],[href]')].map(e => new URL(e.getAttribute('src') ?? e.getAttribute('href'), location.href).href),
    AUD.src, ...performance.getEntriesByType('resource').map(e => e.name)]));
  // stylesheets are read from disk (a file:// page may not read its own cssRules in Chromium)
  for (const css of requests.filter(u => u.startsWith('file:') && u.endsWith('.css')))
    for (const m of fs.readFileSync(new URL(css), 'utf8').matchAll(/url\(\s*["']?([^"')]+)/g)) requests.push(new URL(m[1], css).href);
  const bad = requests.filter(u => !/^(file|data|blob):/.test(u));
  assert.ok(requests.length > 5, `saw ${requests.length} requests`);
  assert.deepEqual(bad, []); await ctx.close();
});

test('X1 three.js 0.186.1 and the bloom addons load as a classic script', async () => {
  const { ctx, pg } = await openWorld();
  assert.deepEqual(await pg.evaluate(() => [THREE.REVISION, typeof THREE.EffectComposer, typeof THREE.RenderPass, typeof THREE.UnrealBloomPass, typeof THREE.OutputPass]),
    ['186', 'function', 'function', 'function', 'function']);
  await ctx.close();
});

test('X2 the 3D layer is lazy: no WebGL context and a hidden canvas until started', async () => {
  const { ctx, pg } = await openWorld();
  assert.deepEqual(await pg.evaluate(() => [WorldSpace.renderer, getComputedStyle(document.getElementById('space')).display]), [null, 'none']);
  await ctx.close();
});

test('X3 the 3D layer draws (bloom, pixel ratio 1) and stop() leaves the page as it was', async () => {
  const { ctx, pg, errors } = await openWorld();
  const before = await shot(pg);
  const r = await pg.evaluate(() => {
    const sc = new THREE.Scene(), cam = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100); cam.position.z = 5;
    WorldSpace.start({ scene: sc, camera: cam });
    const R = WorldSpace.renderer, gl = R.getContext(), dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const lit = () => { const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, b = new Uint8Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, b); let n = 0;
      for (let i = 0; i < b.length; i += 4) if (b[i] + b[i + 1] + b[i + 2] > 30) n++; return n / (w * h); };
    WorldSpace.render(); const empty = lit();
    sc.add(new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshBasicMaterial({ color: 0xffcc66 })));
    WorldSpace.render(); const sphere = lit();
    const out = { empty, sphere, pr: R.getPixelRatio(), bloom: WorldSpace.bloom, samples: WorldSpace.samples,
      gpu: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER) };
    WorldSpace.stop(); return out;
  });
  assert.equal(r.empty, 0, 'empty scene must read 0 % lit (meter calibration)');
  assert.ok(r.sphere > 0.01, `sphere lit ${r.sphere}`);
  assert.deepEqual([r.pr, r.bloom, r.samples], [1, true, 0]);
  if (process.env.REQUIRE_HW_GL) assert.doesNotMatch(r.gpu, SOFTWARE_GL);
  assert.equal(await pxDiff(pg, before, await shot(pg)), 0);
  assert.deepEqual(errors, []); await ctx.close();
});

// ---- presenter view: presenter.html opens the World with a real click; the two talk by postMessage ----
async function openPresenter(q = '') {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
  const pres = await ctx.newPage(), errors = [];
  pres.on('pageerror', e => errors.push('presenter: ' + e));
  await pres.goto(url('presenter.html', q));
  const [world] = await Promise.all([pres.waitForEvent('popup', { timeout: 5000 }), pres.click('#open-world')]);
  world.on('pageerror', e => errors.push('world: ' + e));
  await world.waitForFunction(() => typeof __world !== 'undefined');
  await world.evaluate(() => { window.MOTION = 'instant'; });
  return { ctx, pres, world, errors };
}
const shows = (pres, sel, want, timeout = 1000) =>
  pres.waitForFunction(([s, w]) => document.querySelector(s)?.textContent === w, [sel, want], { timeout });
const done_ = world => world.evaluate(() => done);

test('X4 restarting the 3D layer leaks nothing: GPU textures flat over 5 stop/start cycles (ticket 06, R1: +11 per start)', async () => {
  const { ctx, pg, errors } = await openWorld();
  const counts = await pg.evaluate(() => {
    const sc = new THREE.Scene(), cam = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100); cam.position.z = 5;
    sc.add(new THREE.Mesh(new THREE.SphereGeometry(1, 16, 8), new THREE.MeshBasicMaterial({ color: 0xffcc66 })));
    const out = [];
    for (let i = 0; i < 6; i++) { WorldSpace.start({ scene: sc, camera: cam }); WorldSpace.render(); out.push(WorldSpace.renderer.info.memory.textures); WorldSpace.stop(); }
    return out;
  });
  assert.ok(counts[0] > 0, `the bloom allocates textures (control): ${counts}`);
  // flat from the second start on (the first restart adds one lazily-made composer texture, once); a leak grows every start
  assert.deepEqual(counts.slice(1), counts.slice(1).map(() => counts[1]), `textures per start: ${counts}`);
  assert.ok(counts[1] - counts[0] <= 1, `textures per start: ${counts}`);
  assert.deepEqual(errors, []); await ctx.close();
});

test('P1 the presenter opens the World and shows beat, next beat and its note within 1 s', async () => {
  const { ctx, pres, world, errors } = await openPresenter();
  await shows(pres, '#now-label', 'S01b0'); await shows(pres, '#next-label', 'S01b1');
  const note = await world.evaluate(() => scenes[0].notes[0]);
  assert.ok(await pres.evaluate(n => { const d = document.createElement('div'); d.innerHTML = n;
    return document.querySelector('#next-note').innerHTML === d.innerHTML; }, note), 'next note = scenes[0].notes[0]');
  assert.deepEqual(errors, []); await ctx.close();
});

test('P2 clicks in the World are mirrored: presenter now/next = the World\'s own beatView', async () => {
  const { ctx, pres, world } = await openPresenter();
  for (let i = 0; i < 3; i++) { await world.keyboard.press('ArrowRight'); await settle(world); }
  const st = await world.evaluate(() => __world.state());
  assert.equal(st.now.label, 'S02b0'); // S01 has 2 beats: clicks 1–2 reveal them, click 3 opens S02
  await shows(pres, '#now-label', st.now.label); await shows(pres, '#next-label', st.next.label);
  await ctx.close();
});

test('P3 the clicker works from the presenter window: PageDown / PageUp move the World', async () => {
  const { ctx, pres, world } = await openPresenter();
  await shows(pres, '#now-label', 'S01b0');
  await pres.keyboard.press('PageDown'); await world.waitForFunction(() => done === 1 && run === null);
  await pres.keyboard.press('PageUp'); await world.waitForFunction(() => done === 0 && run === null);
  await shows(pres, '#now-label', 'S01b0');
  await ctx.close();
});

test('P4 a reloaded presenter re-pairs within 2 s', async () => {
  const { ctx, pres, world } = await openPresenter();
  await world.keyboard.press('ArrowRight'); await settle(world);
  await pres.reload(); await shows(pres, '#now-label', 'S01b1', 2000);
  await pres.keyboard.press('PageDown'); await world.waitForFunction(() => done === 2 && run === null);
  await ctx.close();
});

test('P5 a reloaded World re-pairs within 2 s, back at the start', async () => {
  const { ctx, pres, world } = await openPresenter();
  await world.keyboard.press('ArrowRight'); await settle(world); await shows(pres, '#now-label', 'S01b1');
  await world.reload(); await world.waitForFunction(() => typeof __world !== 'undefined');
  await shows(pres, '#now-label', 'S01b0', 2000);
  await world.evaluate(() => { window.MOTION = 'instant'; });
  await pres.keyboard.press('PageDown'); await world.waitForFunction(() => done === 1 && run === null);
  await ctx.close();
});

test('P6 elapsed clock starts at the first advance; a click resets it', async () => {
  const { ctx, pres, world } = await openPresenter();
  await wait(2200); assert.equal(await pres.textContent('#elapsed'), '0:00');
  assert.match(await pres.textContent('#wall'), /^\d{2}:\d{2}$/);
  await world.keyboard.press('ArrowRight'); await wait(2300);
  assert.equal(await pres.textContent('#elapsed'), '0:02');
  await pres.click('#elapsed'); await wait(300);
  assert.equal(await pres.textContent('#elapsed'), '0:00');
  await ctx.close();
});

// ---- backup clips: a temp copy of the World whose scene 1 has a generated 6 s clip, red|green|blue 2 s each ----
let clipDir = null;
function clipWorld() {
  if (clipDir) return clipDir;
  clipDir = fs.mkdtempSync('/tmp/world-clip-');
  fs.cpSync(WD, clipDir, { recursive: true });
  const c = n => ['-f', 'lavfi', '-i', `color=c=${n}:s=480x270:r=30:d=2`];
  const r = require('node:child_process').spawnSync('ffmpeg', ['-y', '-loglevel', 'error', ...c('red'), ...c('green'), ...c('blue'),
    '-filter_complex', '[0][1][2]concat=n=3:v=1:a=0', '-c:v', 'libvpx-vp9', '-g', '15', '-b:v', '200k', path.join(clipDir, 'clips/test.webm')]);
  assert.equal(r.status, 0, 'ffmpeg: ' + r.stderr);
  fs.writeFileSync(path.join(clipDir, 'clips/manifest.js'), 'window.WORLD_CLIPS = {"1": {src: "clips/test.webm", marks: [1, 3, 5]}};\n');
  return clipDir;
}
after(() => { if (clipDir) fs.rmSync(clipDir, { recursive: true, force: true }); });
const vid = pg => pg.evaluate(() => { const v = document.querySelector('#backup video'); return v && { t: v.currentTime, paused: v.paused, rs: v.readyState, loop: v.loop, preload: v.preload }; });
// the clip is held at mark t: paused there, seek finished
const heldAt = (pg, t, timeout = 6000) => pg.waitForFunction(t => { const v = document.querySelector('#backup video.cur');
  return v && v.paused && !v.seeking && v.readyState >= 2 && Math.abs(v.currentTime - t) < 0.1; }, t, { timeout });
async function centre(pg) {
  const png = await pg.screenshot();
  return pg.evaluate(async s => { const bm = await createImageBitmap(await (await fetch('data:image/png;base64,' + s)).blob());
    const x = new OffscreenCanvas(bm.width, bm.height).getContext('2d'); x.drawImage(bm, 0, 0);
    const d = x.getImageData(bm.width >> 1, bm.height >> 1, 1, 1).data;
    return d[0] > 180 && d[1] < 80 && d[2] < 80 ? 'red' : d[1] > 90 && d[0] < 80 && d[2] < 80 ? 'green' : d[2] > 180 && d[0] < 80 && d[1] < 80 ? 'blue' : `other(${d[0]},${d[1]},${d[2]})`; },
  png.toString('base64'));
}
const backup = pg => pg.evaluate(() => __world.state().backup);

test('B1 the current scene\'s clip is preloaded before any key (warm swap)', async () => {
  const { ctx, pg } = await openWorld('', 'world.html', clipWorld());
  await pg.waitForFunction(() => document.querySelector('#backup video')?.readyState === 4, null, { timeout: 8000 });
  const v = await vid(pg);
  assert.deepEqual([v.preload, v.loop, v.paused], ['auto', false, true]);
  assert.deepEqual(await backup(pg), { forced: [], clip: 'ready', shown: false });
  await ctx.close();
});

test('B2 a clip that is not forced changes nothing on screen', async () => {
  const a = await openWorld('', 'world.html', clipWorld()), b = await openWorld();
  await a.pg.waitForFunction(() => document.querySelector('#backup video')?.readyState === 4, null, { timeout: 8000 });
  assert.equal(await pxDiff(a.pg, await shot(a.pg), await shot(b.pg)), 0);
  await a.ctx.close(); await b.ctx.close();
});

test('B3 ?backup=1 shows scene 1\'s clip held at its opening mark', async () => {
  const { ctx, pg } = await openWorld('?backup=1', 'world.html', clipWorld());
  await heldAt(pg, 1); assert.equal(await centre(pg), 'red');
  assert.deepEqual(await backup(pg), { forced: [1], clip: 'ready', shown: true });
  await ctx.close();
});

test('B4 each click plays the clip on to the next mark and holds; back seeks and holds', async () => {
  const { ctx, pg } = await openWorld('?backup=1', 'world.html', clipWorld());
  await pg.evaluate(() => { window.MOTION = 'instant'; });
  await heldAt(pg, 1);
  await pg.keyboard.press('ArrowRight'); await heldAt(pg, 3); assert.equal(await centre(pg), 'green');
  await pg.keyboard.press('ArrowRight'); await heldAt(pg, 5); assert.equal(await centre(pg), 'blue');
  await pg.keyboard.press('ArrowLeft'); await heldAt(pg, 3); assert.equal(await centre(pg), 'green');
  await ctx.close();
});

test('B5 V turns the clip off (live scene, 0 px from the no-clip World) and on again', async () => {
  const a = await openWorld('?backup=1', 'world.html', clipWorld()), b = await openWorld();
  for (const p of [a.pg, b.pg]) { await p.evaluate(() => { window.MOTION = 'instant'; }); await p.keyboard.press('ArrowRight'); await settle(p); }
  await heldAt(a.pg, 3);
  await a.pg.keyboard.press('v'); await a.pg.waitForFunction(() => !__world.state().backup.shown);
  assert.equal(await pxDiff(a.pg, await shot(a.pg), await shot(b.pg)), 0);
  await a.pg.keyboard.press('V'); await heldAt(a.pg, 3); assert.equal(await centre(a.pg), 'green');
  await a.ctx.close(); await b.ctx.close();
});

test('B6 V on a scene without a clip does nothing', async () => {
  const { ctx, pg } = await openWorld('', 'world.html', clipWorld());
  await pg.keyboard.press('ArrowDown'); await pg.waitForFunction(() => si === 1 && glTimer === null);
  const s0 = await shot(pg);
  await pg.keyboard.press('v'); await wait(300);
  assert.equal(await pxDiff(pg, s0, await shot(pg)), 0);
  assert.deepEqual(await backup(pg), { forced: [], clip: 'none', shown: false });
  await ctx.close();
});

test('B7 nothing auto-plays: a shown clip stays held while idle', async () => {
  const { ctx, pg } = await openWorld('?backup=1', 'world.html', clipWorld());
  await heldAt(pg, 1); await wait(3000); await heldAt(pg, 1, 100);
  await ctx.close();
});

test('B8 the 3D loop is held while a clip covers the scene and resumes after', async () => {
  const { ctx, pg } = await openWorld('', 'world.html', clipWorld());
  const frames = () => pg.evaluate(() => WorldSpace.frames);
  await pg.evaluate(() => { const c = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100); WorldSpace.start({ scene: new THREE.Scene(), camera: c }); });
  let f = await frames(); await wait(400); assert.ok(await frames() > f, '3D loop runs');
  await pg.keyboard.press('v'); await heldAt(pg, 1);
  f = await frames(); await wait(400); assert.equal(await frames(), f, '3D loop held under the clip');
  await pg.keyboard.press('v'); await pg.waitForFunction(() => !__world.state().backup.shown);
  f = await frames(); await wait(400); assert.ok(await frames() > f, '3D loop resumes');
  await ctx.close();
});

// ---- stock browsers only: prove the launch really is the installed browser with its popup blocker on ----
const STOCK = KEY === 'chrome' || KEY === 'firefox-stock';
test('N1 popup blocker is on: a window.open without a click is blocked', { skip: !STOCK && 'bundled browsers run with popups allowed' }, async () => {
  const dir = fs.mkdtempSync('/tmp/world-n1-'), f = path.join(dir, 'n1.html');
  fs.writeFileSync(f, '<meta charset="utf-8"><script>onload=()=>{const w=window.open("n1.html","_blank");window.__n1=w?"opened":"blocked";if(w)w.close();};</script>');
  const ctx = await browser.newContext(), pg = await ctx.newPage();
  await pg.goto('file://' + f); await pg.waitForFunction(() => window.__n1);
  assert.equal(await pg.evaluate(() => window.__n1), 'blocked');
  await ctx.close(); fs.rmSync(dir, { recursive: true });
});

test('N2 the launched browser is the installed one', { skip: !STOCK && 'bundled browser' }, () => {
  const bin = KEY === 'chrome' ? 'google-chrome' : launch.executablePath;
  const installed = require('node:child_process').execFileSync(bin, ['--version'], { encoding: 'utf8' });
  assert.ok(installed.includes(browser.version()), `launched ${browser.version()}, installed ${installed.trim()}`);
});
