// check-jumps.mjs: the engine's invariant, tested. A beat's f(0) must equal the state at the end of the previous beat;
// if it does not, the World jumps on the click. For every beat, the settled state before it is compared with the beat at p = 0:
// boards (opacity, strength as drawn, place, width, content while visible), every 3D object's opacity and place, the camera, the ghost levels.
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-jumps.mjs
import { createRequire } from 'node:module';
const require = createRequire((process.env.NODE_PATH || '/usr/lib/node_modules') + '/');
const { chromium } = require('playwright');
import { installDeckShim } from '/home/gofer/mrs_talk_video/render/deck_shim.mjs';
const browser = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
await page.goto(new URL('../index.html', import.meta.url).href); const inv = await installDeckShim(page); await page.evaluate(() => document.fonts.ready);
await page.evaluate(() => { window.__snap = () => { const S = {};
  BOARDS.all.forEach((bd, i) => { if (bd.noLayout) return; const k = 'board ' + i; S[k + ' opacity'] = [bd.a, 0.02];
    S[k + ' strength'] = [bd.el.style.visibility === 'visible' ? +bd.el.style.opacity : 0, 0.05];   /* as drawn: a board the last beat showed must not dip when the next one shows it again */
    if (bd.a > 0.01) { S[k + ' place'] = [[bd.anchor.x, bd.anchor.y, bd.anchor.z], 1.5]; S[k + ' width'] = [bd.w, 0.05 * bd.w];
      S[k + ' content'] = bd.el.innerHTML.replace(/-?\d+\.\d+/g, m => (+m).toFixed(0)); S[k + ' text'] = (bd.el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 30); } });
  const walk = (o, path, top) => { const k = path + ' ' + (o.type || '') + (o.userData && o.userData.id ? ':' + o.userData.id : '');
    if (o.material && o.material.opacity != null) S[k + ' opacity'] = [o.visible === false ? 0 : o.material.opacity, 0.02];
    if (o.material && o.material.uniforms) for (const u of ['uOp', 'uSpread', 'uTau', 'uWeight', 'uRelax']) if (o.material.uniforms[u]) S[k + ' ' + u] = [+o.material.uniforms[u].value, 0.02];
    if (top && o.type !== 'Points') S[k + ' place'] = [[o.position.x, o.position.y, o.position.z, o.scale.x], 1.5];   /* 1.5 units: the ambient jitter of the pair is not a jump */
    o.children.forEach((c, i) => walk(c, k + '.' + i, false)); };
  W.scene.children.forEach((c, i) => walk(c, 'lit ' + i, true)); W.crisp.children.forEach((c, i) => walk(c, 'crisp ' + i, true));
  S.camera = [[beatCam.pos.x, beatCam.pos.y, beatCam.pos.z, beatCam.look.x, beatCam.look.y, beatCam.look.z], 0.2];
  scenes.forEach((sc, i) => { S['ghost of scene ' + (i + 1)] = [sc.ghost ?? 1, 0.02]; }); S.overview = [OV.beat, 0.02]; return S; }; });
const raf2 = () => page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
const num = v => Array.isArray(v[0]) ? v[0] : [v[0]];
const differ = (a, b) => { if (a === undefined || b === undefined) return false;          /* a board hidden on one side: its opacity line reports it */
  if (typeof a === 'string') return a !== b; const x = num(a), y = num(b); return x.some((v, i) => Math.abs(v - y[i]) > a[1]); };
const show = v => v === undefined ? '-' : typeof v === 'string' ? v.slice(0, 60) : num(v).map(n => +n.toFixed(2)).join(', ');
let bad = 0;
for (let s = 0; s < inv.scenes; s++) for (let k = 0; k < inv.per_scene[s]; k++) {
  await page.evaluate(([s, k]) => { window.__deck.show(s); window.__deck.applyUpTo(k); }, [s, k]); await raf2(); const A = await page.evaluate(() => window.__snap());
  await page.evaluate(([s, k]) => applyState(s, k, 0), [s, k]); await raf2(); const B = await page.evaluate(() => window.__snap());
  const diff = Object.keys(A).filter(key => !key.endsWith(' text') && differ(A[key], B[key]));
  if (diff.length) { bad++; console.log(`S${String(s + 1).padStart(2, '0')} b${k + 1} jumps on its click:`);
    for (const key of diff.slice(0, 6)) console.log(`    ${key}${A[key.replace(/ \w+$/, ' text')] ? ' ("' + A[key.replace(/ \w+$/, ' text')] + '")' : ''}: ${show(A[key])}  →  ${show(B[key])}`); if (diff.length > 6) console.log(`    and ${diff.length - 6} more`); }
}
console.log(`\n${bad} of ${inv.beats} beats jump on their click`);
await browser.close(); process.exit(bad ? 1 : 0);
