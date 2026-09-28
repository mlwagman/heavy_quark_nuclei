// world_test_lib.mjs — browser helpers shared by the World's test and capture tools (ticket 06; the shot/pxDiff
// recipes are ticket 04's, from test_world_browser.mjs). Browsers are the keys of stage_browsers.json.
//   scheme: 'dark' (default: the stage runs dark, ticket 06 decision 1), 'light', or null (no emulation: the OS decides).
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire((process.env.NODE_PATH || '/usr/lib/node_modules') + '/');
export const pw = require('playwright');
const here = path.dirname(new URL(import.meta.url).pathname);
export const WD = path.resolve(process.env.WORLD_DIR || path.join(here, '..'));
export const BG = { dark: 'rgb(10, 14, 23)', light: 'rgb(246, 247, 244)' }; // deck.css --bg
export const SOFTWARE_GL = /swiftshader|llvmpipe|softpipe|software|basic render|or similar/i; // ticket 03's speedtest.py

export async function launch(key = 'chromium', { scheme = 'dark', args = [] } = {}) {
  const { type, launch: opts } = JSON.parse(fs.readFileSync(path.join(here, 'stage_browsers.json'), 'utf8'))[key];
  const o = structuredClone(opts);
  // stock Firefox: context emulation alone may not reach content, so the content pref says dark too (in memory only)
  if (key === 'firefox-stock' && scheme === 'dark') o.firefoxUserPrefs = { ...o.firefoxUserPrefs, 'layout.css.prefers-color-scheme.content-override': 0 };
  if (args.length) o.args = [...(o.args || []), ...args];
  return pw[type].launch(o);
}
export async function openWorld(browser, { q = '', page = 'world.html', dir = WD, scheme = 'dark', init = null } = {}) {
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1, colorScheme: scheme });
  if (init) await ctx.addInitScript(init);
  const pg = await ctx.newPage(), errors = [], consoleErrors = [];
  pg.on('pageerror', e => errors.push(String(e)));
  pg.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  await pg.goto('file://' + path.join(dir, page) + q);
  await pg.waitForFunction(() => { try { return scenes.length === 15; } catch (e) { return false; } });
  await pg.evaluate(() => document.fonts.ready);
  return { ctx, pg, errors, consoleErrors };
}
/* the page really renders dark: media query and the body background (a failed check voids the run) */
export async function assertDark(pg) {
  const r = await pg.evaluate(() => [matchMedia('(prefers-color-scheme: dark)').matches, getComputedStyle(document.body).backgroundColor]);
  if (!(r[0] === true && r[1] === BG.dark)) throw new Error(`NOT_DARK prefers-dark=${r[0]} body=${r[1]}`);
  return r;
}
export const frames = (pg, n = 3) => pg.evaluate(n => new Promise(r => { const f = k => k ? requestAnimationFrame(() => f(k - 1)) : r(); f(n); }), n);
export const glideEnd = pg => pg.waitForFunction(() => glTimer === null);
// pin the ambient clock, repaint #stage afresh (deck_diff.mjs's recipe for a 0 px noise floor), screenshot
export async function shot(pg) {
  await pg.evaluate(async () => {
    window.__RENDER = true; tAmb = 0; window.__frame(0);
    const st = document.getElementById('stage'), fr = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    st.style.willChange = ''; await fr(); st.style.willChange = 'transform'; await fr();
  });
  return pg.screenshot({ animations: 'disabled', caret: 'hide' });
}
/* one page-side analyser for PNG screenshots, decoded in the browser under test:
   diff = differing pixels · psnr = dB over RGB (Infinity if equal) · pixels = RGB at [[x,y],...] */
async function analyse(pg, pngs, kind, arg = null) {
  return pg.evaluate(async ([pngs, kind, arg]) => {
    const dec = async s => { const bm = await createImageBitmap(await (await fetch('data:image/png;base64,' + s)).blob());
      const x = new OffscreenCanvas(bm.width, bm.height).getContext('2d'); x.drawImage(bm, 0, 0); return x.getImageData(0, 0, bm.width, bm.height); };
    const [P, Q] = await Promise.all(pngs.map(dec));
    if (kind === 'pixels') return arg.map(([x, y]) => { const i = (Math.round(y) * P.width + Math.round(x)) * 4; return [P.data[i], P.data[i + 1], P.data[i + 2]]; });
    let n = 0, se = 0;
    for (let i = 0; i < P.data.length; i += 4) {
      const d = [0, 1, 2].map(c => P.data[i + c] - Q.data[i + c]);
      if (d.some(Boolean) || P.data[i + 3] !== Q.data[i + 3]) n++;
      se += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
    return kind === 'diff' ? n : se ? 10 * Math.log10(255 * 255 / (se / (P.data.length / 4 * 3))) : Infinity;
  }, [pngs.map(b => b.toString('base64')), kind, arg]);
}
export const pxDiff = (pg, a, b) => (a.equals(b) ? 0 : analyse(pg, [a, b], 'diff'));
export const psnr = (pg, a, b) => (a.equals(b) ? Infinity : analyse(pg, [a, b], 'psnr'));
export const pixels = (pg, png, pts) => analyse(pg, [png], 'pixels', pts);
