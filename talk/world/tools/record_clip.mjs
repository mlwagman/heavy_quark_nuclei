#!/usr/bin/env node
// record_clip.mjs — backup clips and review stills of rebuilt World scenes (ticket 06), always in the dark stage scheme.
//   NODE_PATH=$(npm root -g) node talk/world/tools/record_clip.mjs --scene 4 --page world.html --out clips/s04-renormalon.webm --evidence DIR
//       Records one scene on the Deck's virtual clock (window.__frame, 30 fps): the whole page (DOM text + 3D), VP9 CRF 24,
//       the opening and every settled beat held 0.2 s; marks = the first settled frame of each beat (clips/manifest.js
//       contract, RebuildCore.marksFrom). Writes DIR/<clip>.json {src, marks, frames} and DIR/<clip>-m<k>.png (references).
//   ... --verify clips/s04-renormalon.webm --evidence DIR          (no browser: ffmpeg only)
//       At each mark and mark + 0.1 s the decoded clip frame must be ≥ 30 dB PSNR against its own reference and beat
//       both neighbouring references by ≥ 6 dB (so a clip that holds on the wrong beat fails).
//   ... --stills --scenes 3-6 --page world.html --out DIR [--browser chromium|firefox]
//       Review frames: the neighbours' edge beats, each rebuilt beat at 25/50/75 % and settled, the Deck glides frozen at
//       25/50/75 %; contact sheets DIR/sheet-*.png (montage). Browser jobs except --verify. Exit 0 ok, 1 check failed, 2 setup.
import fs from 'node:fs';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { launch, openWorld, assertDark, glideEnd, frames, shot, WD } from './world_test_lib.mjs';
const RC = createRequire(import.meta.url)(path.join(WD, 'js/rebuild-core.js'));
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const PAGE = opt('--page', 'world.html'), OUT = opt('--out'), EV = opt('--evidence'), FPS = 30, HOLD = 6;
const pad = n => String(n).padStart(2, '0');
const psnrFile = (a, b) => { const r = spawnSync('ffmpeg', ['-hide_banner', '-i', a, '-i', b, '-lavfi', 'psnr', '-f', 'null', '-'], { encoding: 'utf8' });
  const m = /average:(inf|[\d.]+)/.exec(r.stderr); return m ? (m[1] === 'inf' ? Infinity : +m[1]) : NaN; };

if (args.includes('--verify')) {
  const clip = opt('--verify'), base = path.basename(clip, '.webm'), meta = JSON.parse(fs.readFileSync(path.join(EV, base + '.json'), 'utf8'));
  const tmp = fs.mkdtempSync('/tmp/t06verify-'), ref = k => path.join(EV, `${base}-m${k}.png`), lines = [];
  let ok = true;
  meta.marks.forEach((m, k) => [0, 0.1].forEach(dt => {
    const png = path.join(tmp, `${k}-${dt}.png`), t = m + dt + 0.5 / FPS; // mid-frame: never the frame before the mark
    spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', String(t), '-i', clip, '-frames:v', '1', png]);
    const own = psnrFile(png, ref(k)), nb = [k - 1, k + 1].filter(j => j >= 0 && j < meta.marks.length).map(j => psnrFile(png, ref(j)));
    const good = own >= 30 && nb.every(x => own - x >= 6);
    ok &&= good;
    lines.push(`${good ? 'ok  ' : 'BAD '} mark ${k} t=${(m + dt).toFixed(3)} own=${own.toFixed(2)} dB neighbours=${nb.map(x => x.toFixed(2)).join(',')}`);
  }));
  fs.rmSync(tmp, { recursive: true, force: true });
  lines.forEach(l => console.log(l));
  console.log(`${ok ? 'VERIFY PASS' : 'VERIFY FAIL'} ${clip} marks=${meta.marks.length}`);
  process.exit(ok ? 0 : 1);
}

const browser = await launch(opt('--browser', 'chromium'));
let rc = 0;
try {
  const { pg, errors } = await openWorld(browser, { page: PAGE, init: () => { window.__RENDER = true; } });
  await assertDark(pg);
  if (args.includes('--stills')) {
    const [a, z] = opt('--scenes', '3-6').split('-').map(Number), dir = OUT, shots = [];
    fs.mkdirSync(dir, { recursive: true });
    let n = 0;
    // glide frames: a plain screenshot, because animations:'disabled' (in shot()) would fast-forward the paused glide
    const save = async (id, still = true) => { const f = path.join(dir, `${pad(n++)}-${id}.png`);
      fs.writeFileSync(f, still ? await shot(pg) : (await pg.evaluate(() => { tAmb = 0; }), await pg.screenshot({ caret: 'hide' }))); shots.push(f); };
    const beats = await pg.evaluate(() => scenes.map(s => s.beats.length));
    // a Deck glide frozen at 25/50/75 %: the engine's 720 ms clean-up timer is held back until the frames are taken
    const glide = async (to, id) => {
      await pg.evaluate(to => { window.__st = window.setTimeout; window.setTimeout = (f, ms, ...r) => (ms === 720 ? (window.__gcb = f, -1) : window.__st(f, ms, ...r)); show(to); applyUpTo(0); }, to);
      for (const f of [25, 50, 75]) {
        await pg.evaluate(f => { for (const an of document.getAnimations()) { an.pause(); an.currentTime = 7 * f; } }, f);
        await frames(pg, 3); await save(`${id}-${f}`, false);
      }
      await pg.evaluate(() => { for (const an of document.getAnimations()) an.finish(); window.setTimeout = window.__st; window.__gcb && window.__gcb(); });
      await glideEnd(pg); await frames(pg, 3);
    };
    await pg.evaluate(i => { show(i); applyUpTo(scenes[i].beats.length); }, a - 1); await glideEnd(pg); await frames(pg, 3);
    await save(`S${pad(a)}b${beats[a - 1]}-settled`);
    for (let i = a; i < z - 1; i++) { // the rebuilt scenes between the two neighbours
      await glide(i, `glide-S${pad(i)}-S${pad(i + 1)}`);
      await save(`S${pad(i + 1)}b0-settled`);
      for (let k = 1; k <= beats[i]; k++) {
        for (const f of [25, 50, 75]) { await pg.evaluate(([k, f]) => { applyUpTo(k - 1); scenes[si].beats[k - 1].f(f / 100); }, [k, f]); await frames(pg, 2); await save(`S${pad(i + 1)}b${k}-p${f}`); }
        await pg.evaluate(k => applyUpTo(k), k); await frames(pg, 2); await save(`S${pad(i + 1)}b${k}-settled`);
      }
    }
    await glide(z - 1, `glide-S${pad(z - 1)}-S${pad(z)}`);
    await save(`S${pad(z)}b0-settled`);
    for (const [name, pat] of [['S04', /S04b|S03b/], ['S05', /S05b|S06b/], ['glides', /glide-/]]) {
      const files = shots.filter(f => pat.test(path.basename(f)));
      const r = spawnSync('montage', ['-label', '%t', ...files, '-tile', '4x', '-geometry', '640x360+6+6', '-pointsize', '18', '-background', '#1b1b1b', '-fill', '#dddddd', path.join(dir, `sheet-${name}.png`)]);
      if (r.status) throw new Error('montage: ' + r.stderr);
    }
    fs.writeFileSync(path.join(dir, 'stills.json'), JSON.stringify({ page: PAGE, dark: true, frames: shots.map(f => path.basename(f)), errors }, null, 1));
    console.log(`STILLS ${shots.length} frames, 3 sheets in ${dir}; page errors ${errors.length}`);
    if (errors.length) rc = 1;
  } else {
    const N = +opt('--scene'), clip = OUT, base = path.basename(clip, '.webm');
    fs.mkdirSync(EV, { recursive: true });
    const n = await pg.evaluate(i => { evIdx = NARR.sched.length; window.MOTION = 'dynamic'; show(i); applyUpTo(0); return scenes[i].beats.length; }, N - 1);
    await glideEnd(pg); await frames(pg, 4);
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
      '-c:v', 'libvpx-vp9', '-crf', '24', '-b:v', '0', '-g', '15', '-pix_fmt', 'yuv420p', '-row-mt', '1', clip], { stdio: ['pipe', 'inherit', 'inherit'] });
    const done = new Promise(r => ff.on('close', r));
    const log = [], refs = new Map();
    let i = 0;
    const frame = async () => {
      const st = await pg.evaluate(vt => { window.__frame(vt); __rebuiltSpace.update(); if (__rebuiltSpace.on) WorldSpace.render();
        return { done, run: run ? run.i : null }; }, i / FPS);
      const png = await pg.screenshot({ caret: 'hide' });
      if (!ff.stdin.write(png)) await new Promise(r => ff.stdin.once('drain', r));
      const t = +(i / FPS).toFixed(4); log.push({ t, ...st }); i++;
      if (st.run === null && !refs.has(st.done)) refs.set(st.done, png);
    };
    for (let h = 0; h <= HOLD; h++) await frame();
    for (let k = 1; k <= n; k++) {
      await pg.evaluate(() => next());
      do await frame(); while (log.at(-1).run !== null);
      for (let h = 0; h < HOLD; h++) await frame();
    }
    ff.stdin.end();
    if (await done) throw new Error('ffmpeg failed');
    const marks = RC.marksFrom(log, n);
    // reference frames: in the evidence dir and beside the clip (gitignored), where the tests look for them
    for (const [k, png] of refs) for (const d of [EV, path.dirname(clip)]) fs.writeFileSync(path.join(d, `${base}-m${k}.png`), png);
    fs.writeFileSync(path.join(EV, base + '.json'), JSON.stringify({ src: `clips/${base}.webm`, marks, frames: i, page: PAGE, dark: true, errors }, null, 1));
    const bytes = fs.statSync(clip).size;
    console.log(`CLIP ${clip} scene=${N} frames=${i} duration=${(i / FPS).toFixed(2)}s bytes=${bytes} marks=${JSON.stringify(marks)} page_errors=${errors.length}`);
    if (errors.length) rc = 1;
  }
} catch (e) { console.log('SETUP_ERROR ' + (e.stack || e)); rc = 2; }
await browser.close();
process.exit(rc);
