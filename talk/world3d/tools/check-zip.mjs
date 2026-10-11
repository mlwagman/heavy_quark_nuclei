// check-zip.mjs: the talk zip (talk/mrs-talk-world3d.zip, built by tools/build-zip.py) unzips into one folder that works on
// Windows: legal file names, no two names that differ only by case, short paths, a README that Notepad reads (CRLF), every file
// the World and the presenter load, byte-identical to the sources; unzipped, both pages open from file:// with no page error and
// no request outside the folder.
//   python3 talk/world3d/tools/build-zip.py && NODE_PATH=$(npm root -g) node talk/world3d/tools/check-zip.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH + '/');
const { chromium } = require('playwright');
const W = new URL('../', import.meta.url).pathname, ZIP = path.join(W, '../mrs-talk-world3d.zip'), TOP = 'mrs-talk-world3d/';
const R = []; const ok = (name, pass, got) => { R.push(pass); console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}  (${got})`); };
if (!fs.existsSync(ZIP)) { ok('talk/mrs-talk-world3d.zip exists', false, 'missing'); process.exit(1); }

const names = execFileSync('unzip', ['-Z1', ZIP], { encoding: 'utf8' }).split('\n').filter(Boolean), files = names.filter(n => !n.endsWith('/'));
ok('everything sits in one folder, mrs-talk-world3d/', names.every(n => n.startsWith(TOP)), names.filter(n => !n.startsWith(TOP)).slice(0, 3).join(', ') || 'yes');
const RESERVED = /^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$/i;
const bad = names.flatMap(n => n.split('/').filter(Boolean)).filter(c => !/^[A-Za-z0-9._ -]+$/.test(c) || /[. ]$/.test(c) || RESERVED.test(c));
ok('every name is legal on Windows (letters, digits, . _ - space; no trailing dot or space; no CON, AUX, …)', bad.length === 0, [...new Set(bad)].slice(0, 5).join(', ') || 'all legal');
const low = new Map(); names.forEach(n => { const k = n.toLowerCase(); low.set(k, (low.get(k) || 0) + 1); });
const dup = [...low].filter(([, c]) => c > 1).map(([k]) => k);
ok('no two names differ only by case', dup.length === 0, dup.slice(0, 3).join(', ') || 'none');
const longest = Math.max(...names.map(n => n.length));
ok('paths stay short (≤ 120 characters inside the zip, room for C:\\Users\\…\\Downloads\\ under 260)', longest <= 120, `${longest}`);

/* what must be inside, byte for byte: the pages, css, js, vendor, fonts, assets of talk/world3d, and the phone file */
const walk = d => fs.readdirSync(path.join(W, d), { withFileTypes: true }).flatMap(e => e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]);
const want = ['index.html', 'presenter.html', ...['css', 'js', 'vendor', 'fonts', 'assets'].flatMap(walk)];
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'zipcheck-')); execFileSync('unzip', ['-q', ZIP, '-d', tmp]);
const root = path.join(tmp, TOP), same = (a, b) => fs.existsSync(b) && fs.readFileSync(a).equals(fs.readFileSync(b));
const miss = want.filter(f => !same(path.join(W, f), path.join(root, f)));
ok(`the ${want.length} source files are inside and identical to talk/world3d`, miss.length === 0, miss.slice(0, 4).join(', ') || 'all identical');
ok('the phone file is inside and identical to talk/mrs-talk-practice.html', same(path.join(W, '../mrs-talk-practice.html'), path.join(root, 'mrs-talk-practice.html')), 'compared');
const extra = files.map(f => f.slice(TOP.length)).filter(f => !want.includes(f) && !['mrs-talk-practice.html', 'README.txt'].includes(f));
ok('nothing else (no tools, notes, frames)', extra.length === 0, extra.slice(0, 4).join(', ') || 'none');
const readme = fs.existsSync(path.join(root, 'README.txt')) ? fs.readFileSync(path.join(root, 'README.txt'), 'utf8') : '';
ok('README.txt is there with Windows line ends (CRLF) and says to unzip before opening', readme.length > 0 && /\r\n/.test(readme) && !/[^\r]\n/.test(readme) && /Extract all/i.test(readme), readme ? `${readme.split('\r\n').length} lines` : 'missing');

/* unzipped, the pages work from file:// */
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
for (const pg of ['index.html', 'presenter.html']) {
  const page = await b.newPage({ viewport: { width: 1920, height: 1080 } }), out = [], errs = [];
  page.on('request', r => { const u = r.url(); if (!u.startsWith('data:') && !u.startsWith('blob:') && !u.startsWith('file://' + root) && !u.startsWith('about:')) out.push(u); });
  page.on('requestfailed', r => errs.push('failed ' + r.url())); page.on('pageerror', e => errs.push(String(e)));
  await page.goto('file://' + path.join(root, pg)); await page.waitForTimeout(2500);
  const n = pg === 'index.html' ? await page.evaluate(() => typeof scenes !== 'undefined' ? scenes.length : 0) : 14;
  ok(`unzipped ${pg} opens: no page error, nothing loaded from outside the folder${pg === 'index.html' ? ', 14 scenes' : ''}`, errs.length === 0 && out.length === 0 && n === 14, [...errs, ...out].slice(0, 3).join(' | ') || `${n} scenes`);
  await page.close();
}
await b.close(); fs.rmSync(tmp, { recursive: true, force: true });
console.log(`${R.filter(Boolean).length} pass, ${R.filter(x => !x).length} fail`); process.exit(R.every(Boolean) ? 0 : 1);
