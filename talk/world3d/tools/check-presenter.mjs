// check-presenter.mjs: the presenter view has no content of its own; it must show what the World sends it. For three scenes it is
// asked to open (through its own channel to the World), it has to show that scene's beat count and the keynotes of its last beat.
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-presenter.mjs
import fs from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH + '/');
const { chromium } = require('playwright');
const N = '/home/gofer/mrs_talk_video/narration_world3d/';
const beats = n => fs.readFileSync(`${N}scene_${String(n).padStart(2, '0')}.md`, 'utf8').split(/^## Beat .*$/m).slice(1).map(x => x.trim().replace(/\s+/g, ' '));
/* round 20: the presenter shows keynotes (js/keys.js), not the narration: a few words per beat, and what to point at without reading */
const src = new URL('../js/keys.js', import.meta.url); let KEYS = [], KEY_T = [];
try { ({ KEYS, KEY_T } = new Function(fs.readFileSync(src, 'utf8') + '; return { KEYS, KEY_T };')()); } catch (e) { console.log('FAIL  js/keys.js: ' + e.message); process.exit(1); }
let bad = 0; const chk = (ok, msg) => { if (!ok) bad++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`); };
const words = t => t.split(/\s+/).filter(Boolean).length, all = KEYS.flat();
const counts = Array.from({ length: 14 }, (_, i) => beats(i + 1).length);
chk(KEYS.length === 14 && KEYS.every((sc, i) => sc.length === counts[i]), `keynotes: one entry per beat (${KEYS.map(sc => sc.length).join(',')} against ${counts.join(',')})`);
const long = all.flatMap(b => b.k).filter(t => words(t) > 10), many = all.filter(b => b.k.length < 1 || b.k.length > 3).length;
chk(long.length === 0 && many === 0, `keynotes: 1 to 3 per beat, 10 words or fewer each (${many} beats out of range; too long: ${JSON.stringify(long.slice(0, 3))})`);
const kw = all.reduce((n, b) => n + b.k.reduce((m, t) => m + words(t), 0), 0), nw = counts.reduce((n, _, i) => n + beats(i + 1).reduce((m, t) => m + words(t), 0), 0);
chk(kw < 0.4 * nw, `keynotes: ${kw} words against ${nw} of narration (under 40 %)`);
const named = all.flatMap(b => b.k).filter(t => /Komijani|Kronfeld|Hylleraas|Assi|Wagman|Detmold|Meinel|[—–]/.test(t));
chk(named.length === 0, `keynotes: no name of a person to say out loud, no dash (${JSON.stringify(named)})`);
chk(KEY_T.length === 14 && KEY_T[0] === 0 && KEY_T.every((t, i) => i === 0 || t > KEY_T[i - 1]) && KEY_T[13] < 900, `target start times: 14, rising, inside 15 minutes (${KEY_T.join(',')})`);
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
const ctx = await b.newContext({ viewport: { width: 1280, height: 720 } }), page = await ctx.newPage();
await page.goto(new URL('../presenter.html', import.meta.url).href);
const [world] = await Promise.all([ctx.waitForEvent('page'), page.click('#open')]);
await world.waitForLoadState('load'); await page.waitForTimeout(2500);
const norm = t => t.replace(/\s+/g, ' ').trim();
for (const sc of [4, 5, 12]) {
  await page.evaluate(si => w.postMessage({ type: 'show', si, end: true }, '*'), sc - 1); await page.waitForTimeout(700);
  const got = await page.evaluate(() => ({ where: document.getElementById('where').textContent, keys: [...document.querySelectorAll('#notes li')].map(e => e.textContent), show: (document.querySelector('#notes .show') || {}).textContent || '', all: document.getElementById('notes').textContent }));
  const K = KEYS[sc - 1], want = K[K.length - 1], nar = beats(sc)[K.length - 1];
  chk(got.where.startsWith(`scene ${sc} / 14 · beat ${K.length} / ${K.length}`) && JSON.stringify(got.keys.map(norm)) === JSON.stringify(want.k.map(norm)) && (!want.show || got.show.includes(want.show)) && !norm(got.all).includes(nar.slice(0, 40)),
    `presenter, scene ${sc}: "${got.where}", shows the keynotes of its last beat ${JSON.stringify(got.keys)}, not the narration`);
  chk(new RegExp(`target ${Math.floor(KEY_T[sc - 1] / 60)}:${String(KEY_T[sc - 1] % 60).padStart(2, '0')}`).test(got.where), `presenter, scene ${sc}: shows the time this panel should start ("${got.where}")`);
}
await page.click('#next_'); await page.waitForTimeout(600);
const nx = await page.evaluate(() => document.getElementById('where').textContent);
chk(/^scene 13 \/ 14 · beat 1 /.test(nx), `presenter: "next" after the end of scene 12 starts scene 13 ("${nx}")`);
await b.close(); process.exit(bad ? 1 : 0);
