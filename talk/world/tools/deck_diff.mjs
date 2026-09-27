#!/usr/bin/env node
// deck_diff.mjs — screenshot + notes diff between two builds of the MRS Deck (ticket 02's gate; reused
// by ticket 08 "only the named beats may change" and ticket 04 "a hosted scene looks exactly as in the Deck").
//
//   NODE_PATH=$(npm root -g) node talk/world/tools/deck_diff.mjs --a <ref.html> --b <cand.html> --out <dir>
//        [--allow S14b4,S07b2,...] [--browser chromium|firefox]
//
// Both pages open over file:// in the SAME browser at 1920×1080, get render/deck_shim.mjs, and are driven
// through the identical sequence (the rv() reveal variants depend on call order). Per scene: the opening frame
// (SNNb0), every settled beat (SNNbK) and every beat half-way (SNNbK, kind "mid": applyUpTo(K-1); f(0.5)) —
// 129 frames for the 57-beat Deck. A frame differs iff any pixel differs (threshold 0: a Deck-vs-Deck run is 0,
// so any tolerance would only hide real changes). Notes strings are compared per beat (notes[K-1] ↔ SNNbK).
// Writes <out>/report.json and <id>-<kind>.{a,b}.png for each differing frame.
// Exit 0: nothing outside --allow differs. 1: unexpected difference or INVENTORY_MISMATCH.
// 2: setup/page error (FONT_NOT_LOADED, IMG_NOT_LOADED, AUDIO_NOT_LOADED, PAGE_ERROR, CONSOLE_ERROR, shim timeout).
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { installDeckShim } from '/home/gofer/mrs_talk_video/render/deck_shim.mjs';
const require = createRequire((process.env.NODE_PATH || '/usr/lib/node_modules') + '/');
const pw = require('playwright');

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const [A, B, OUT, BROWSER] = [opt('--a'), opt('--b'), opt('--out'), opt('--browser', 'chromium')];
const allowed = opt('--allow', '').split(',').filter(Boolean);
if (!A || !B || !OUT || !['chromium', 'firefox'].includes(BROWSER)) {
  console.error('usage: deck_diff.mjs --a <ref.html> --b <cand.html> --out <dir> [--allow SNNbK,...] [--browser chromium|firefox]');
  process.exit(2);
}
fs.mkdirSync(OUT, { recursive: true });
const RESTORE = 'python3 talk/world/tools/extract_deck.py --only assets/narration.mp3';
const report = { pass: false, browser: BROWSER, browser_version: null, a: path.resolve(A), b: path.resolve(B), frames: 0,
  threshold_px: 0, differing_beats: [], differing_frames: [], notes_differing: [], allowed, unexpected: [],
  inventory: {}, audio: {}, errors: {}, console: {} };
class SetupError extends Error { constructor(code, detail) { super(`${code} ${detail}`); this.code = code; } }
const pad = n => String(n).padStart(2, '0');

async function open(browser, file, side) {
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  // Park the rAF loop before the Deck's script runs: if the ambient orbiters (S01/S02) move on the wall clock
  // even briefly, the repaint residue they leave differs between the two pages by a few pixels (T0 caught it).
  await ctx.addInitScript(() => { window.__RENDER = true; });
  const page = await ctx.newPage();
  const errors = report.errors[side] = [], cons = report.console[side] = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') cons.push(m.text()); });
  await page.goto('file://' + path.resolve(file), { timeout: 60000 });
  try { await installDeckShim(page, { timeout: 60000 }); } catch (e) { throw new SetupError('SHIM_TIMEOUT', `${side}: ${e.message}`); }
  const fonts = await page.evaluate(async () => {
    await Promise.all(['400 24px P052', 'italic 400 24px P052'].map(f => document.fonts.load(f).catch(() => null)));
    return [...document.fonts].filter(f => f.family.replace(/["']/g, '') === 'P052').map(f => `${f.style}:${f.status}`);
  });
  if (fonts.length !== 2 || fonts.some(s => !s.endsWith(':loaded'))) throw new SetupError('FONT_NOT_LOADED', `${side}: ${fonts.join(' ')}`);
  const imgs = await page.evaluate(() => Promise.all([...document.images].map(i => i.decode().then(() => null, () => (i.getAttribute('src') || '').slice(0, 60)))));
  if (imgs.some(Boolean)) throw new SetupError('IMG_NOT_LOADED', `${side}: ${imgs.filter(Boolean).join(' ')}`);
  // AUD is built at load with preload=auto, so a missing narration.mp3 is a real load failure, not a lazy one.
  report.audio[side] = await page.evaluate(() => new Promise(res => {
    if (AUD.error) return res('error:' + AUD.error.code);
    if (AUD.readyState >= 1) return res('ok');
    const t = setTimeout(() => res('timeout'), 15000);
    AUD.addEventListener('loadedmetadata', () => { clearTimeout(t); res('ok'); }, { once: true });
    AUD.addEventListener('error', () => { clearTimeout(t); res('error:' + AUD.error.code); }, { once: true });
  }));
  report.inventory[side] = await page.evaluate(() => ({ scenes: scenes.length, beats: scenes.reduce((a, s) => a + s.beats.length, 0),
    sum_d: scenes.reduce((a, s) => a + s.beats.reduce((t, b) => t + b.d, 0), 0), per_scene: scenes.map(s => s.beats.length),
    d: scenes.map(s => s.beats.map(b => b.d)) }));
  return page;
}

// Decode both PNGs in the browser under test and count pixels whose RGBA differs anywhere.
async function pixelDiff(helper, a, b) {
  if (a.equals(b)) return { pixels: 0, bbox: null };
  return helper.evaluate(async ([a, b]) => {
    const dec = async s => { const bm = await createImageBitmap(await (await fetch('data:image/png;base64,' + s)).blob());
      const x = new OffscreenCanvas(bm.width, bm.height).getContext('2d'); x.drawImage(bm, 0, 0); return x.getImageData(0, 0, bm.width, bm.height); };
    const [P, Q] = [await dec(a), await dec(b)];
    if (P.width !== Q.width || P.height !== Q.height) return { pixels: P.width * P.height, bbox: null, size: [P.width, P.height, Q.width, Q.height] };
    let n = 0, x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
    for (let p = 0; p < P.data.length; p += 4) {
      if (P.data[p] === Q.data[p] && P.data[p + 1] === Q.data[p + 1] && P.data[p + 2] === Q.data[p + 2] && P.data[p + 3] === Q.data[p + 3]) continue;
      const q = p / 4, x = q % P.width, y = (q - x) / P.width; n++;
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
    return { pixels: n, bbox: n ? [x0, y0, x1, y1] : null };
  }, [a.toString('base64'), b.toString('base64')]);
}

let rc = 0;
const browser = await pw[BROWSER].launch();
report.browser_version = browser.version();
try {
  const pa = await open(browser, A, 'a'), pb = await open(browser, B, 'b');
  const both = (fn, arg) => Promise.all([pa.evaluate(fn, arg), pb.evaluate(fn, arg)]);
  const audioOk = report.audio.a === report.audio.b;
  if (!audioOk) throw new SetupError('AUDIO_NOT_LOADED', `a=${report.audio.a} b=${report.audio.b}; restore: ${RESTORE}`);
  const [ia, ib] = [report.inventory.a, report.inventory.b];
  if (JSON.stringify(ia) !== JSON.stringify(ib)) {
    rc = 1; console.log(`DIFF FAIL INVENTORY_MISMATCH a=${JSON.stringify(ia.per_scene)} b=${JSON.stringify(ib.per_scene)}`);
  } else {
    const helper = await (await browser.newContext()).newPage();
    const shoot = async (id, kind) => {
      // Pin the ambient orbiters (S01/S02) to one pose, then raster #stage afresh on a new layer: partial repaints
      // leave antialiasing residue that depends on paint timing, so identical DOMs differed by a few edge pixels.
      await both(async () => {
        tAmb = 0; window.__frame(0);
        const st = document.getElementById('stage'), frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        st.style.willChange = ''; await frame(); st.style.willChange = 'transform'; await frame();
      });
      const [sa, sb] = await Promise.all([pa, pb].map(p => p.screenshot({ animations: 'disabled', caret: 'hide' })));
      report.frames++;
      const d = await pixelDiff(helper, sa, sb);
      if (d.pixels > 0) {
        report.differing_frames.push({ id, kind, ...d });
        fs.writeFileSync(path.join(OUT, `${id}-${kind}.a.png`), sa); fs.writeFileSync(path.join(OUT, `${id}-${kind}.b.png`), sb);
      }
    };
    for (let i = 0; i < ia.scenes; i++) {
      const n = ia.per_scene[i], S = `S${pad(i + 1)}`;
      await both(i => { show(i); applyUpTo(0); }, i);
      await Promise.all([pa, pb].map(p => p.waitForFunction(() => glTimer === null)));
      await shoot(`${S}b0`, 'open');
      for (let k = 1; k <= n; k++) { await both(k => applyUpTo(k), k); await shoot(`${S}b${k}`, 'settled'); }
      for (let k = 1; k <= n; k++) { await both(k => { applyUpTo(k - 1); scenes[si].beats[k - 1].f(0.5); }, k); await shoot(`${S}b${k}`, 'mid'); }
    }
    const [na, nb] = await both(() => scenes.map(s => s.notes || []));
    na.forEach((N, i) => { for (let j = 0; j < Math.max(N.length, nb[i].length); j++) if (N[j] !== nb[i][j]) report.notes_differing.push(`S${pad(i + 1)}b${j + 1}`); });
    report.differing_beats = [...new Set(report.differing_frames.map(f => f.id))];
    report.unexpected = [...new Set([...report.differing_beats, ...report.notes_differing])].filter(id => !allowed.includes(id));
    rc = report.unexpected.length ? 1 : 0;
  }
  if (report.errors.a.length || report.errors.b.length) throw new SetupError('PAGE_ERROR', JSON.stringify(report.errors).slice(0, 300));
  if (report.console.b.length > report.console.a.length) throw new SetupError('CONSOLE_ERROR', report.console.b.slice(0, 3).join(' | '));
  report.pass = rc === 0;
  const counts = `browser=${BROWSER} frames=${report.frames} differing=${report.differing_beats.length} notes_differing=${report.notes_differing.length}`;
  if (rc === 0) console.log(`DIFF PASS ${counts}`);
  else if (report.frames) console.log(`DIFF FAIL ${counts} unexpected=${JSON.stringify(report.unexpected)}`);
} catch (e) { // SetupError carries its code; anything else (goto timeout, crash) is still a setup failure, never "differs"
  report.error = e.message; report.pass = false; rc = 2;
  console.log(`DIFF SETUP_ERROR ${e.message}`);
} finally {
  fs.writeFileSync(path.join(OUT, 'report.json'), JSON.stringify(report, null, 1));
  await browser.close();
}
process.exit(rc);
