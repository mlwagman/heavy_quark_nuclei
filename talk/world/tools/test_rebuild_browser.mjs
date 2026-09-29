// test_rebuild_browser.mjs — browser tests for the rebuilt S04/S05 (ticket 06), one real browser per run, dark scheme.
//   BROWSER=chromium|firefox|chrome|firefox-stock (stage_browsers.json)   WORLD_PAGE=<page> (default world.html)
//   WORLD_DIR=<dir>   REQUIRE_HW_GL=1 fails R4 on a software renderer.
//   BR=1 adds the backup-clip tests; outside Chromium they read the recorder's reference frames (<clip>-m<k>.png,
//   written beside each clip by tools/record_clip.mjs; CLIP_REFS=<dir> overrides): the clips are recorded in Chromium, and Firefox wraps some text
//   a few px differently, so there the held frame is compared with the clip's own frames, not with Firefox's live scene.
//   NODE_PATH=$(npm root -g) BROWSER=firefox node --test --test-reporter=tap talk/world/tools/test_rebuild_browser.mjs
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { launch, openWorld, assertDark, frames, glideEnd, shot, pxDiff, psnr, pixels, BG, WD, SOFTWARE_GL } from './world_test_lib.mjs';
// no top-level await below this line: tests registered after one run after the file's after() hook when a
// --test-name-pattern filters (the browser was closed mid-BR-run and the runner hung, lock M)
const RC = createRequire(import.meta.url)(path.join(WD, 'js/rebuild-core.js'));
const KEY = process.env.BROWSER || 'chromium', CHROMIUM = /^(chromium|chrome)$/.test(KEY), PAGE = process.env.WORLD_PAGE || 'world.html';
let browser;
before(async () => { browser = await launch(KEY); });
after(async () => { // bounded: a close that hangs must not keep the runner alive
  await Promise.race([browser?.close(), new Promise(r => setTimeout(r, 15000).unref())]);
});

const inventory = pg => pg.evaluate(() => ({ per_scene: scenes.map(s => s.beats.length), d: scenes.map(s => s.beats.map(b => b.d)),
  sum_d: scenes.reduce((a, s) => a + s.beats.reduce((t, b) => t + b.d, 0), 0), titles: scenes.map(s => (s.el.querySelector('h1') || s.el.querySelector('.eyebrow')).textContent) })); // world.js's rule: S01 has no h1
const space = pg => pg.evaluate(() => ({ display: getComputedStyle(document.getElementById('space')).display, running: WorldSpace.running }));
const go = (pg, fn, arg) => pg.evaluate(fn, arg);

test('R0 the run is dark: prefers-color-scheme dark and the dark --bg on body', async () => {
  const { ctx, pg, errors } = await openWorld(browser, { page: PAGE });
  await assertDark(pg); assert.deepEqual(errors, []); await ctx.close();
});

test('R1 same beats and durations as the Deck (inventory, sum_d 1014, titles); S04 and S05 are the rebuilt sections', async () => {
  const w = await openWorld(browser, { page: PAGE }), d = await openWorld(browser, { page: 'index.html' });
  const [iw, id] = [await inventory(w.pg), await inventory(d.pg)];
  assert.deepEqual(iw, id); assert.equal(iw.sum_d, 1014);
  assert.deepEqual(await go(w.pg, () => scenes.map(s => s.el.classList.contains('w-rebuilt'))), scenes15(i => i === 3 || i === 4));
  assert.equal(await go(w.pg, () => document.querySelectorAll('#deck > section.scene').length), 15, 'no leftover hosted S04/S05 section');
  await w.ctx.close(); await d.ctx.close();
});
const scenes15 = f => Array.from({ length: 15 }, (_, i) => f(i));

test('R2 3D lifecycle: lazy until S04, on through S04-S05, off once the glide to a hosted scene has ended', async () => {
  const { ctx, pg, errors, consoleErrors } = await openWorld(browser, { page: PAGE });
  assert.deepEqual([await go(pg, () => WorldSpace.renderer), (await space(pg)).display], [null, 'none'], 'lazy at S01');
  await go(pg, () => { show(2); applyUpTo(4); }); await glideEnd(pg); await frames(pg);
  assert.equal((await space(pg)).display, 'none', 'still off on hosted S03');
  await go(pg, () => show(3));
  await pg.waitForFunction(() => getComputedStyle(document.getElementById('space')).display === 'block', null, { timeout: 2000 });
  assert.ok(await go(pg, () => glTimer !== null), 'on already while the S03 → S04 glide runs');
  await glideEnd(pg); await frames(pg);
  assert.deepEqual(await space(pg), { display: 'block', running: true });
  await go(pg, () => show(4)); await frames(pg);
  assert.equal((await space(pg)).display, 'block', 'S04 → S05 glide: stays on'); await glideEnd(pg); await frames(pg);
  assert.deepEqual(await space(pg), { display: 'block', running: true });
  // sampled on every frame while the glide runs (a check after "a few frames" raced the 0.7 s glide under load)
  const during = await go(pg, () => new Promise(res => { const d = []; show(5);
    (function f() { if (glTimer === null) return res(d); d.push(getComputedStyle(document.getElementById('space')).display); requestAnimationFrame(f); })(); }));
  assert.ok(during.length >= 3 && during.every(x => x === 'block'), `S05 → S06: still on while the glide fades the floor out (${during})`);
  await frames(pg);
  assert.deepEqual(await space(pg), { display: 'none', running: false }, 'off on hosted S06');
  await go(pg, () => show(3)); await glideEnd(pg); await frames(pg); assert.equal((await space(pg)).display, 'block');
  await go(pg, () => show(2)); await glideEnd(pg); await frames(pg); assert.equal((await space(pg)).display, 'none', 'back to S03: off');
  assert.deepEqual([errors, consoleErrors], [[], []]); await ctx.close();
});

for (const scheme of ['dark', 'light']) test(`R3 no seam: on S04 the empty corners show the body --bg exactly; 3D on in dark, off in light (${scheme})`, async () => {
  // the light run gets its own browser: stock Firefox is launched with a forced-dark content pref (world_test_lib)
  const b = scheme === 'dark' ? browser : await launch(KEY, { scheme: 'light' });
  const { ctx, pg } = await openWorld(b, { page: PAGE, scheme });
  assert.equal(await go(pg, () => matchMedia('(prefers-color-scheme: dark)').matches), scheme === 'dark', 'the page renders the scheme under test');
  await go(pg, () => { show(3); applyUpTo(0); }); await glideEnd(pg); await frames(pg, 6);
  assert.equal((await space(pg)).display, scheme === 'dark' ? 'block' : 'none', 'light: the bloom would whiten the --bg, so no 3D');
  const px = await pixels(pg, await shot(pg), [[8, 8], [8, 1070], [1910, 1070]]); // corners away from the pair's glow
  assert.deepEqual(px.map(p => `rgb(${p.join(', ')})`), [BG[scheme], BG[scheme], BG[scheme]]);
  await go(pg, () => applyUpTo(4));
  assert.ok(await go(pg, () => [...scenes[3].el.querySelectorAll('.card')].every(c => +c.style.opacity > 0.3)), 'the Boards show');
  await ctx.close(); if (b !== browser) await b.close();
});

test('R4 the Borel branch point is crisp (no glow around the ring) while the bloomed quark pair glows (control, same run)', async () => {
  const { ctx, pg } = await openWorld(browser, { page: PAGE });
  await go(pg, () => { show(3); applyUpTo(0); });
  await glideEnd(pg); await frames(pg, 6);
  // lit fraction of an annulus 6-14 px outside a circle, skipping the ±4 px row of the u-axis / cut
  const lit = async (png, cx, cy, r) => {
    const pts = [];
    for (let a = 0; a < 360; a += 3) for (let d = r + 6; d <= r + 14; d += 2) {
      const x = cx + d * Math.cos(a * Math.PI / 180), y = cy + d * Math.sin(a * Math.PI / 180);
      if (Math.abs(y - cy) > 4) pts.push([x, y]);
    }
    const bg = [10, 14, 23], px = await pixels(pg, png, pts);
    return px.filter(p => p[0] + p[1] + p[2] - bg[0] - bg[1] - bg[2] > 36).length / px.length;
  };
  const cam0 = RC.cameraAt(4, [0, 0, 0, 0]), q = RC.project(RC.PAIR.map((v, i) => v + [0.4, 0, 0][i]), cam0, 1920, 1080); // right quark centre, radius 0.15
  const edge = RC.project(RC.PAIR.map((v, i) => v + [0.55, 0, 0][i]), cam0, 1920, 1080);
  const pairLit = await lit(await shot(pg), q.x, q.y, Math.hypot(edge.x - q.x, edge.y - q.y));
  await go(pg, () => applyUpTo(4)); await frames(pg, 6);
  const ring = await go(pg, () => { const c = document.querySelector('#w-floor-overlay .bp'); return [+c.getAttribute('cx'), +c.getAttribute('cy'), +c.getAttribute('r')]; });
  const ringLit = await lit(await shot(pg), ...ring);
  const gpuName = await go(pg, () => { const gl = WorldSpace.renderer.getContext(), x = gl.getExtension('WEBGL_debug_renderer_info'); return x ? gl.getParameter(x.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); });
  if (process.env.REQUIRE_HW_GL) assert.doesNotMatch(gpuName, SOFTWARE_GL);
  assert.ok(pairLit > 0.25, `control: the bloomed pair must light its annulus (lit ${pairLit.toFixed(3)}, gpu ${gpuName})`);
  assert.ok(ringLit < 0.05, `branch point annulus lit ${ringLit.toFixed(3)} (pair ${pairLit.toFixed(3)})`);
  await ctx.close();
});

test('R5 notes: the rebuilt S04/S05 carry the hosted notes (S04 through E1/E2 once its flag is on)', async () => {
  const w = await openWorld(browser, { page: PAGE }), d = await openWorld(browser, { page: 'index.html' });
  const [nw, nd, e12] = [await go(w.pg, () => [scenes[3].notes, scenes[4].notes]), await go(d.pg, () => [scenes[3].notes, scenes[4].notes]),
    await go(w.pg, () => W_REBUILT.find(r => r.sc === 4).E12)];
  assert.deepEqual(nw, [e12 ? RC.e12(nd[0]) : nd[0], nd[1]]);
  await w.ctx.close(); await d.ctx.close();
});

test('R6 reveal variants: the rv() counter after each scene opening equals the Deck\'s (so hosted scenes keep their look)', async () => {
  const seq = pg => go(pg, () => Array.from({ length: 15 }, (_, i) => { show(i); applyUpTo(0); return RVN; }));
  const w = await openWorld(browser, { page: PAGE }), d = await openWorld(browser, { page: 'index.html' });
  assert.deepEqual(await seq(w.pg), await seq(d.pg));
  await w.ctx.close(); await d.ctx.close();
});

test('R7 no WebGL: no page error, the 3D stays off, the Boards still show', async () => {
  const init = () => { const g = HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext = function (k, ...a) { return /webgl/.test(k) ? null : g.call(this, k, ...a); }; };
  const { ctx, pg, errors } = await openWorld(browser, { page: PAGE, init });
  await go(pg, () => { show(3); applyUpTo(4); }); await glideEnd(pg); await frames(pg, 4);
  assert.deepEqual(await go(pg, () => [__rebuiltSpace.failed, __rebuiltSpace.on]), [true, false]);
  const ops = await go(pg, () => [...scenes[3].el.querySelectorAll('.card')].map(c => +c.style.opacity));
  assert.ok(ops.length >= 7 && ops.every(o => o > 0.3), `cards ${ops}`);
  assert.deepEqual(errors, []); await ctx.close();
});

test('R9 rebuilt content stays inside its section box (overflow would enlarge the Deck stage layer and move hosted antialiasing, lock E: S02b5)', async () => {
  const { ctx, pg } = await openWorld(browser, { page: PAGE });
  for (const [i, k] of [[1, 5], [3, 0], [3, 4], [4, 0], [4, 5]]) {
    await go(pg, ([i, k]) => { show(i); applyUpTo(k); }, [i, k]); await glideEnd(pg); await frames(pg, 3);
    const out = await go(pg, () => scenes.filter(s => s.el.classList.contains('w-rebuilt')).map(s => {
      const o = getComputedStyle(s.el).overflow, r = s.el.getBoundingClientRect(), bad = [];
      for (const e of s.el.querySelectorAll('*')) { const q = e.getBoundingClientRect(); if (!q.width && !q.height) continue;
        if (q.left < r.left - 1 || q.top < r.top - 1 || q.right > r.right + 1 || q.bottom > r.bottom + 1) bad.push(e.className.baseVal ?? e.className); }
      return { o, clipped: o === 'clip' || o === 'hidden', outside: bad.length, sample: bad.slice(0, 3) };
    }));
    for (const x of out) assert.ok(x.clipped || x.outside === 0, `S0${i + 1}b${k}: overflow ${x.o}, ${x.outside} elements outside (${x.sample})`);
  }
  await ctx.close();
});

test('R10 a rebuilt scene off screen is not rendered at all (hidden S04 content changed Chromium\'s raster of S02b5, lock I); on during its glides', async () => {
  const { ctx, pg } = await openWorld(browser, { page: PAGE });
  const disp = () => go(pg, () => [3, 4].map(i => getComputedStyle(scenes[i].el).display));
  await go(pg, () => { show(1); applyUpTo(5); }); await glideEnd(pg);
  assert.deepEqual(await disp(), ['none', 'none'], 'on hosted S02');
  await go(pg, () => { show(2); applyUpTo(4); }); await glideEnd(pg);
  await go(pg, () => show(3));
  assert.deepEqual(await disp(), ['flex', 'none'], 'S03 -> S04 glide: S04 renders from its first frame');
  await glideEnd(pg); await go(pg, () => show(4));
  assert.deepEqual(await disp(), ['flex', 'flex'], 'S04 -> S05 glide: both');
  await glideEnd(pg); assert.deepEqual(await disp(), ['none', 'flex'], 'after the glide');
  await ctx.close();
});

test('R11 S05 b3: the band flies from the cut to Λ_R without crossing box 1, the 𝒥 box or the "} +" (physics review round 2, R2-1)', async () => {
  const { ctx, pg } = await openWorld(browser, { page: PAGE });
  await go(pg, () => { show(4); applyUpTo(2); }); await glideEnd(pg); await frames(pg, 3);
  const r = await go(pg, () => {
    const s = scenes[4].el, band = s.querySelector('.w-band'), out = [];
    const box = (e, m = 0) => { const q = e.getBoundingClientRect(); return [q.left - m, q.top - m, q.right + m, q.bottom + m]; };
    const keep = { 'box 1': s.querySelector('.w-b1'), '𝒥 box': s.querySelector('.w-b2'), '} +': s.querySelectorAll('.w-eqrow > .w-t')[2] };
    let seen = 0;
    for (let i = 0; i <= 100; i++) {
      scenes[4].beats[2].f(i / 100); __rebuiltSpace.update();
      if (+getComputedStyle(band).opacity < 0.02) continue;
      seen++;
      const [l, t, rt, b] = box(band, 3); // plus its 3 px blur
      for (const [n, e] of Object.entries(keep)) { const [L, T, R, B] = box(e); if (l < R && rt > L && t < B && b > T) out.push(`p=${i / 100} ${n}`); }
    }
    return { seen, out };
  });
  assert.ok(r.seen > 20, `the band is on screen during its flight (${r.seen} of 101 samples)`);
  assert.deepEqual(r.out, [], 'the band overlaps');
  await ctx.close();
});

test('R8 dark hosted scenes are untouched: S01b0, every S03 and S06 frame, S15b1 are 0 px vs the Deck (index.html)', async () => {
  const init = () => { window.__RENDER = true; };
  const w = await openWorld(browser, { page: PAGE, init }), d = await openWorld(browser, { page: 'index.html', init });
  const both = (fn, arg) => Promise.all([w.pg.evaluate(fn, arg), d.pg.evaluate(fn, arg)]);
  const bad = [];
  const cmp = async id => { const [a, b] = await Promise.all([shot(w.pg), shot(d.pg)]); const n = await pxDiff(w.pg, a, b); if (n) bad.push(`${id}:${n}`); };
  for (let i = 0; i < 15; i++) {
    const n = await go(w.pg, i => scenes[i].beats.length, i), S = `S${String(i + 1).padStart(2, '0')}`;
    await both(i => { show(i); applyUpTo(0); }, i); await Promise.all([glideEnd(w.pg), glideEnd(d.pg)]);
    if (i === 0) await cmp('S01b0');
    if (i === 2 || i === 5) {
      await cmp(`${S}b0`);
      for (let k = 1; k <= n; k++) { await both(k => applyUpTo(k), k); await cmp(`${S}b${k}`); }
      for (let k = 1; k <= n; k++) { await both(k => { applyUpTo(k - 1); scenes[si].beats[k - 1].f(0.5); }, k); await cmp(`${S}b${k}-mid`); }
    }
    if (i === 14) { await both(() => applyUpTo(1)); await cmp('S15b1'); }
  }
  assert.deepEqual(bad, []);
  assert.deepEqual([w.errors, d.errors], [[], []]);
  assert.deepEqual(w.consoleErrors, [], 'no console errors across all 15 scenes (deck_diff fails on them: CONSOLE_ERROR)');
  await w.ctx.close(); await d.ctx.close();
});

// ---- backup clips of the rebuilt scenes (BR=1; the World at WORLD_DIR must list scenes 4 and 5 in clips/manifest.js) ----
if (process.env.BR) {
  const heldAt = (pg, t) => pg.waitForFunction(t => { const v = document.querySelector('#backup video.cur');
    return !!v && v.paused && !v.seeking && Math.abs(v.currentTime - t) < 0.02; }, t, { timeout: 25000 });
  // Chromium: each held frame against that beat's live frame (the clip shows the scene's own code). Elsewhere: against
  // the recorder's reference frames (≥ 30 dB own, ≥ 6 dB over both neighbours, as record_clip.mjs --verify).
  for (const n of [4, 5]) test(`BR1 S0${n}: ?backup=${n} holds the clip at every mark, on that beat's frame (PSNR; ${CHROMIUM ? 'vs live' : 'vs clip references'})`, { timeout: 240000 }, async () => {
    const { ctx, pg, errors } = await openWorld(browser, { page: PAGE, q: `?backup=${n}` });
    const { marks, src } = await go(pg, n => WORLD_CLIPS[String(n)], n), nb = marks.length - 1, L = [];
    if (CHROMIUM) {
      const live = await openWorld(browser, { page: PAGE, init: () => { window.__RENDER = true; } });
      await go(live.pg, i => { show(i); applyUpTo(0); }, n - 1); await glideEnd(live.pg); await frames(live.pg, 4);
      for (let k = 0; k <= nb; k++) { await go(live.pg, k => applyUpTo(k), k); await frames(live.pg, 3); L.push(await shot(live.pg)); }
      await live.ctx.close();
    } else {
      const base = path.basename(src, '.webm'), refs = process.env.CLIP_REFS || path.join(WD, path.dirname(src));
      for (let k = 0; k <= nb; k++) L.push(fs.readFileSync(path.join(refs, `${base}-m${k}.png`)));
    }
    assert.equal(nb, await go(pg, i => scenes[i].beats.length, n - 1), 'manifest marks = beats + 1');
    const [minOwn, minGap] = CHROMIUM ? [25, 3] : [30, 6];
    await go(pg, i => { show(i); }, n - 1); await glideEnd(pg);
    const rows = [];
    for (let k = 0; k <= nb; k++) {
      if (k) await pg.keyboard.press('ArrowRight');
      await heldAt(pg, marks[k]); await frames(pg, 3);
      const c = await pg.screenshot({ caret: 'hide' });
      const own = await psnr(pg, c, L[k]), nbrs = [];
      for (const j of [k - 1, k + 1]) if (j >= 0 && j <= nb) nbrs.push(await psnr(pg, c, L[j]));
      rows.push(`b${k}: own ${own.toFixed(1)} dB, neighbours ${nbrs.map(x => x.toFixed(1)).join('/')}`);
      assert.ok(own >= minOwn && nbrs.every(x => own >= x + minGap), `S0${n} ${rows.at(-1)}`);
    }
    assert.deepEqual(await go(pg, () => __world.state().backup.shown), true);
    assert.deepEqual(errors, []); await ctx.close();
    console.log(`# BR1 S0${n} ` + rows.join(' | '));
  });
  test('BR2 V toggles the S04 clip off and on; BR3 the 3D is held (no frames) while the clip covers it', { timeout: 60000 }, async () => {
    const { ctx, pg } = await openWorld(browser, { page: PAGE });
    await go(pg, () => { show(3); applyUpTo(0); }); await glideEnd(pg); await frames(pg, 4);
    const st = () => go(pg, () => __world.state().backup.shown);
    assert.equal(await st(), false);
    await pg.keyboard.press('v'); await frames(pg, 4); assert.equal(await st(), true, 'V: clip on');
    const f0 = await go(pg, () => WorldSpace.frames); await pg.waitForTimeout(600);
    assert.equal(await go(pg, () => WorldSpace.frames), f0, 'no 3D frames under the clip');
    await pg.keyboard.press('v'); await frames(pg, 4); assert.equal(await st(), false, 'V again: live');
    const f1 = await go(pg, () => WorldSpace.frames); await pg.waitForTimeout(600);
    assert.ok(await go(pg, () => WorldSpace.frames) > f1, '3D runs again');
    await ctx.close();
  });
  test('BR4 on S04 the S05 clip is already loaded (warm swap)', { timeout: 60000 }, async () => {
    const { ctx, pg } = await openWorld(browser, { page: PAGE });
    await go(pg, () => { show(3); applyUpTo(0); }); await glideEnd(pg);
    await pg.waitForFunction(() => { const v = [...document.querySelectorAll('#backup video')].find(x => /s05/.test(x.src)); return v && v.readyState >= 4; }, null, { timeout: 20000 });
    await ctx.close();
  });
}
