#!/usr/bin/env node
// contact.mjs: drive every beat of the World through the capture shim and save one PNG per settled beat.
//
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/contact.mjs              # all beats → talk/world3d/contact/SNNbM.png
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/contact.mjs --only S07   # one scene
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/contact.mjs --fps S07b2 S10b2   # headed, GPU: prints fps per beat
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/contact.mjs --mid        # also save each beat at p = 0.5 (SNNbM-mid.png)
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/contact.mjs --overview S15b1   # the journey overview from that beat → contact/overview-S15b1.png
//   ... --theme dark                                                         # any mode: render with ?theme=dark into contact/theme-dark/
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire((process.env.NODE_PATH || '/usr/lib/node_modules') + '/');
const { chromium } = require('playwright');
import { installDeckShim } from '/home/gofer/mrs_talk_video/render/deck_shim.mjs';

const HERE = path.dirname(new URL(import.meta.url).pathname);
const ROOT = path.resolve(HERE, '..');
const INDEX = path.join(ROOT, 'index.html');
const theme = (() => { const i = process.argv.indexOf('--theme'); return i >= 0 ? process.argv[i + 1] : null; })();
const OUT = path.join(ROOT, 'contact', theme ? 'theme-' + theme : '');
const url = q => { const qs = [theme ? 'theme=' + theme : '', q].filter(Boolean).join('&'); return 'file://' + INDEX + (qs ? '?' + qs : ''); };
const argv = process.argv.slice(2);
const flag = (n) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : null; };
const only = flag('--only');
const fpsBeats = argv.includes('--fps') ? argv.slice(argv.indexOf('--fps') + 1).filter(a => /^S\d+b\d+$/.test(a)) : null;
const mid = argv.includes('--mid');
const ovBeat = flag('--overview');
const atList = flag('--at');                                  /* --at S03b1:0.26,S05b2:0.68  frames inside a beat → contact/S03b1-p26.png */
const layoutMode = argv.includes('--layout');                /* true-space check: which boards of different scenes share a place in the World */
const frameMode = argv.includes('--frame');                   /* framing check: boards cut by the stage edge, per beat (--mid: also inside the beats) */
const mapMode = argv.includes('--map');                     /* the journey map: S14 b4 lift-off, settled map, S15 b1 descent → contact/map-*.png */

const gpuArgs = ['--use-gl=angle', '--use-angle=gl', '--enable-gpu-rasterization', '--ignore-gpu-blocklist'];

if (fpsBeats) {
  /* fps: headed Chromium on the real GPU (DISPLAY must be set), the World's own ?fps counter */
  const browser = await chromium.launch({ headless: false, args: [...gpuArgs, '--window-size=1920,1080', '--window-position=0,0'] });
  for (const b of fpsBeats) {
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
    await page.goto(url(`fps&beat=${b}`));
    await installDeckShim(page);
    await page.waitForTimeout(3000);                          /* warm up */
    const samples = [];
    for (let i = 0; i < 8; i++) { await page.waitForTimeout(600); samples.push(await page.evaluate(() => window.__fps)); }
    const gl = await page.evaluate(() => { const c = document.createElement('canvas'); const g = c.getContext('webgl2') || c.getContext('webgl'); const d = g && g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'n/a'; });
    const med = samples.slice().sort((a, b) => a - b)[Math.floor(samples.length / 2)];
    console.log(`${b}: fps median ${med.toFixed(1)} (samples ${samples.map(s => s.toFixed(0)).join(' ')}) renderer: ${gl}`);
    await page.close();
  }
  await browser.close();
  process.exit(0);
}

/* --play S05: click through one scene in real time (the engine's own tween pace), reporting page errors and the beat clock;
   --firefox: open the World in Firefox and screenshot the S05 b2 settled beat (the offline double-click case) */
const play = flag('--play');
if (play || argv.includes('--firefox')) {
  const bt = argv.includes('--firefox') ? require('playwright').firefox : chromium;
  const browser = await bt.launch({ headless: true, args: argv.includes('--firefox') ? [] : gpuArgs });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const errs = []; page.on('pageerror', e => errs.push(String(e)));
  await page.goto(url(play ? '' : 'beat=S05b2'));
  await installDeckShim(page);
  if (play) {
    const s = Number(play.replace(/^S0?/, '')) - 1;
    await page.evaluate(s => { window.__deck.show(s); window.__deck.applyUpTo(0); }, s);
    const nb = await page.evaluate(() => window.__deck.scenes[window.__deck.si].beats.length);
    for (let k = 0; k < nb; k++) {
      await page.evaluate(() => window.__deck.next());
      const t0 = Date.now();
      await page.waitForFunction(() => window.__deck.run === null, null, { timeout: 30000 });
      const st = await page.evaluate(() => ({ si: window.__deck.si, done: window.__deck.done }));
      console.log(`${play} b${k + 1}: tween ${((Date.now() - t0) / 1000).toFixed(1)} s → scene ${st.si + 1} done ${st.done}`);
    }
    await page.evaluate(() => window.__deck.next());           /* the click at the scene end moves on and starts the next beat */
    await page.waitForTimeout(600);
    console.log('after the scene-end click:', JSON.stringify(await page.evaluate(() => ({ si: window.__deck.si, run: window.__deck.run && window.__deck.run.i }))));
  } else {
    await page.waitForTimeout(1500);
    fs.mkdirSync(OUT, { recursive: true }); await page.screenshot({ path: path.join(OUT, 'firefox-S05b2.png') });
    console.log('firefox: saved ' + path.relative(ROOT, path.join(OUT, 'firefox-S05b2.png')));
  }
  console.log(errs.length ? 'ERRORS:\n' + errs.join('\n') : 'no page errors');
  await browser.close();
  process.exit(errs.length ? 1 : 0);
}

fs.mkdirSync(OUT, { recursive: true });
if (layoutMode) {
  const browser = await chromium.launch({ headless: true, args: gpuArgs });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto(url('')); await installDeckShim(page); await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => { window.__deck.show(scenes.length - 1); window.__deck.applyUpTo(99); }); await page.waitForTimeout(500);
  const out = await page.evaluate(() => {
    const R = []; for (const b of BOARDS.all) { if (b.a <= 0.01 || b.hidden || b.noLayout) continue; const hw = b.w / 2, hh = (b.h || 10) * b.w / b.px / 2;
      R.push({ sc: b.scene == null ? 'P' : 'S' + String(b.scene + 1).padStart(2, '0'), t: (b.el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 28), l: b.anchor.x - hw, r: b.anchor.x + hw, b: b.anchor.y - hh - (b.hung ? hh : 0), t2: b.anchor.y + hh - (b.hung ? hh : 0) }); }
    return R; });
  const box = {}; for (const r of out) { const o = box[r.sc] || (box[r.sc] = { l: 1e9, r: -1e9, b: 1e9, t: -1e9, n: 0 }); o.l = Math.min(o.l, r.l); o.r = Math.max(o.r, r.r); o.b = Math.min(o.b, r.b); o.t = Math.max(o.t, r.t2); o.n++; }
  for (const k of Object.keys(box).sort()) { const o = box[k]; console.log(`${k}: ${o.n} boards, x ${o.l.toFixed(0)}..${o.r.toFixed(0)}  y ${o.b.toFixed(0)}..${o.t.toFixed(0)}`); }
  const hits = []; for (let i = 0; i < out.length; i++) for (let j = i + 1; j < out.length; j++) { const A = out[i], B = out[j]; if (A.sc === B.sc) continue;
    const ox = Math.min(A.r, B.r) - Math.max(A.l, B.l), oy = Math.min(A.t2, B.t2) - Math.max(A.b, B.b); if (ox > 0.5 && oy > 0.5) hits.push(`${A.sc} "${A.t}" x ${A.l.toFixed(0)}..${A.r.toFixed(0)} y ${A.b.toFixed(0)}..${A.t2.toFixed(0)}  ×  ${B.sc} "${B.t}" x ${B.l.toFixed(0)}..${B.r.toFixed(0)} y ${B.b.toFixed(0)}..${B.t2.toFixed(0)}`); }
  console.log(hits.length ? `\n${hits.length} overlapping board pairs:\n` + hits.join('\n') : '\nno two scenes share a place');
  /* colour charge: every hadron glyph must be colour-neutral (r, g, b each cancelled by its anticolour, or one of each) */
  const bad = await page.evaluate(() => {
    const net = qs => { const n = { r: 0, g: 0, b: 0 }; for (const q of qs) { const c = q.userData.c; if (c.length === 1) n[c]++; else n[c[1]]--; } return n.r === n.g && n.g === n.b; };
    const G = Object.entries(SHELF.glyphs).map(([id, x]) => [id, x.g.userData.quarks]); G.push(['hero', window.__S01.hero.userData.quarks]);
    window.__S07.insets.forEach((g, i) => G.push(['inset' + i, g.children.filter(c => c.userData.c)]));
    return { n: G.length, bad: G.filter(([, qs]) => !net(qs)).map(([id, qs]) => id + ':' + qs.map(q => q.userData.c).join(',')) }; });
  console.log(bad.bad.length ? `colour check: NOT neutral: ${bad.bad.join(' ')}` : `colour check: ${bad.n} hadron glyphs, all colour-neutral`);
  await browser.close(); process.exit(hits.length || bad.bad.length ? 1 : 0);   /* round 18: a shared place fails the run */
}
if (frameMode) {
  /* framing check: at the end of every beat (and at p = .25, .5, .75 with --mid) no board may be cut by the stage edge.
     FOCUS = a board the beat is showing (opacity >= .8), DIM = the rest of the scene and the persistent boards (>= .3).
     A FOCUS board cut by the edge fails the run; DIM ones and FOCUS boards left off stage are listed. Ghosts of the previous scene are not checked. */
  const browser = await chromium.launch({ headless: true, args: gpuArgs });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto(url('')); const inv = await installDeckShim(page); await page.evaluate(() => document.fonts.ready);
  const probe = () => page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(() => {
    const R = []; for (const b of BOARDS.all) { const e = b.el; if (e.style.visibility !== 'visible' || b.noLayout) continue; const op = parseFloat(e.style.opacity) || 0; if (op < 0.3) continue;
      const r = e.getBoundingClientRect(); if (r.width < 2 || r.height < 2) continue;
      const cut = Math.max(-r.left, r.right - 1920, -r.top, r.bottom - 1080); if (cut <= 2) continue;
      const off = r.right < 0 || r.left > 1920 || r.bottom < 0 || r.top > 1080;
      R.push({ kind: op >= 0.8 ? 'FOCUS' : 'DIM', off, cut: Math.round(cut), t: (e.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 40) }); }
    done(R); }))));
  let nFocus = 0, nDim = 0, nOff = 0;
  for (let s = 0; s < inv.scenes; s++) { const id = 'S' + String(s + 1).padStart(2, '0'); if (only && only !== id) continue;
    for (let k = 1; k <= inv.per_scene[s]; k++) for (const p of mid ? [0.25, 0.5, 0.75, null] : [null]) {
      await page.evaluate(([s, k, p]) => { window.__deck.show(s); if (p == null) window.__deck.applyUpTo(k); else { window.__deck.applyUpTo(k - 1); applyState(s, k - 1, p); } }, [s, k, p]);
      for (const r of await probe()) {
        if (r.off) { if (r.kind === 'FOCUS' && p == null) { nOff++; console.log(`${id}b${k} FOCUS off stage "${r.t}"`); } continue; }   /* a dim board wholly off stage is simply not in this frame */
        if (r.kind === 'FOCUS') nFocus++; else nDim++;
        console.log(`${id}b${k}${p == null ? '' : ' p=' + p} ${r.kind} cut by ${r.cut}px "${r.t}"`); } } }
  console.log(`\n${nFocus} focus boards cut by the stage edge, ${nOff} focus boards off stage at the end of their beat, ${nDim} dim boards cut`);
  await browser.close(); process.exit(nFocus ? 1 : 0);       /* off stage is a note, not a failure: S03 b1's camera leaves the QCD floor behind on purpose */
}
if (atList) {
  const browser = await chromium.launch({ headless: true, args: gpuArgs });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const errs = []; page.on('pageerror', e => errs.push(String(e))); page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  await page.goto(url('')); await installDeckShim(page); await page.evaluate(() => document.fonts.ready);
  for (const item of atList.split(',')) {
    const m = /^S(\d+)b(\d+):([\d.]+)$/.exec(item); if (!m) { console.log('skip ' + item); continue; }
    const s = Number(m[1]) - 1, k = Number(m[2]), p = Number(m[3]);
    await page.evaluate(([s, k, p]) => { window.__deck.show(s); window.__deck.applyUpTo(k - 1); applyState(s, k - 1, p); }, [s, k, p]);
    await page.waitForTimeout(450); const name = `S${m[1].padStart(2, '0')}b${k}-p${String(Math.round(p * 100)).padStart(2, '0')}.png`;
    await page.screenshot({ path: path.join(OUT, name) }); process.stdout.write(name + ' ');
  }
  console.log('\n' + (errs.length ? 'ERRORS: ' + errs.join(' | ') : 'no page errors'));
  await browser.close(); process.exit(errs.length ? 1 : 0);
}
if (mapMode) {
  const browser = await chromium.launch({ headless: true, args: gpuArgs });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const errs = []; page.on('pageerror', e => errs.push(String(e)));
  await page.goto(url('')); await installDeckShim(page); await page.evaluate(() => document.fonts.ready);
  for (const [s, k, p, name] of [[13, 5, 0.5, 'S14b5-p50'], [13, 5, null, 'S14b5-end'], [14, 1, null, 'S15b1-end'], [14, 2, 0.25, 'S15b2-p25'], [14, 2, null, 'S15b2-end']]) {   /* S14 b5: the lift-off (round 18) */   /* S15 b1: a stop on the map (round 18), b2: down onto the references */
    await page.evaluate(([s, k, p]) => { window.__deck.show(s); if (p == null) window.__deck.applyUpTo(k); else { window.__deck.applyUpTo(k - 1); applyState(s, k - 1, p); } }, [s, k, p]);
    await page.waitForTimeout(400); await page.screenshot({ path: path.join(OUT, `map-${name}.png`) }); process.stdout.write(`map-${name} `);
  }
  console.log('\n' + (errs.length ? 'ERRORS: ' + errs.join(' | ') : 'no page errors'));
  await browser.close(); process.exit(errs.length ? 1 : 0);
}
if (ovBeat) {
  const browser = await chromium.launch({ headless: true, args: gpuArgs });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  const errs = []; page.on('pageerror', e => errs.push(String(e)));
  await page.goto(url(`beat=${ovBeat}&overview`));
  await installDeckShim(page); await page.evaluate(() => document.fonts.ready); await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(OUT, `overview-${ovBeat}.png`) });
  console.log(`saved contact/overview-${ovBeat}.png` + (errs.length ? '\nERRORS: ' + errs.join(' | ') : ''));
  await browser.close(); process.exit(errs.length ? 1 : 0);
}
const browser = await chromium.launch({ headless: true, args: gpuArgs });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
const errs = []; page.on('pageerror', e => errs.push(String(e))); page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
await page.goto(url(''));
const inv = await installDeckShim(page);
console.log('shim', JSON.stringify(inv));
await page.evaluate(() => document.fonts.ready);
const nScenes = inv.scenes;
for (let s = 0; s < nScenes; s++) {
  const id = 'S' + String(s + 1).padStart(2, '0');
  if (only && only !== id) continue;
  const nb = inv.per_scene[s];
  for (let k = 1; k <= nb; k++) {
    if (mid) {
      await page.evaluate(([s, k]) => { window.__deck.show(s); window.__deck.applyUpTo(k - 1); const b = window.__deck.scenes[s].beats[k - 1]; applyState(s, k - 1, 0.5); }, [s, k]);
      await page.waitForTimeout(250);
      await page.screenshot({ path: path.join(OUT, `${id}b${k}-mid.png`) });
    }
    await page.evaluate(([s, k]) => { window.__deck.show(s); window.__deck.applyUpTo(k); }, [s, k]);
    await page.waitForTimeout(250);
    await page.screenshot({ path: path.join(OUT, `${id}b${k}.png`) });
    process.stdout.write(`${id}b${k} `);
  }
}
console.log('\ndone' + (errs.length ? '\nERRORS:\n' + errs.join('\n') : ''));
await browser.close();
