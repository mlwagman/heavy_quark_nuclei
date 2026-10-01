// check-text.mjs: is any text drawn on other text, under a panel, outside its own panel, or cut by the stage edge?
// Works on the ink (the line boxes of every text node and the pill of every chip), not on the board boxes.
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-text.mjs            # the end of every beat
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-text.mjs --mid      # also p = .25 .5 .75 inside every beat
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-text.mjs --only S03
//   NODE_PATH=$(npm root -g) node talk/world3d/tools/check-text.mjs --only S14 --grid 50   # p every 1/50 inside each beat
// STRONG = both sides drawn at >= .3 (fails the run at the end of a beat); FAINT = one side is a dim board or a ghost (listed).
// Text drawn under 5 px (line box) is not read (the overview's thumbnails, round 18): it is skipped, its boxes are not its letters.
import { createRequire } from 'node:module';
const require = createRequire(process.env.NODE_PATH + '/');
const { chromium } = require('playwright');
import { installDeckShim } from '/home/gofer/mrs_talk_video/render/deck_shim.mjs';
const argv = process.argv.slice(2), only = argv.includes('--only') ? argv[argv.indexOf('--only') + 1] : null;
const grid = argv.includes('--grid') ? Number(argv[argv.indexOf('--grid') + 1]) : 0, mid = argv.includes('--mid') || grid > 0;   /* --grid N: p = 1/N ... (N-1)/N inside every beat (round 18: a 0.3 s overlap fell between .25, .5 and .75) */
const PS = grid > 0 ? [...Array(grid - 1).keys()].map(i => +((i + 1) / grid).toFixed(3)) : [0.25, 0.5, 0.75];
const b = await chromium.launch({ headless: true, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'] });
const page = await b.newPage({ viewport: { width: 1920, height: 1080 } });
await page.goto(new URL('../index.html', import.meta.url).href); const inv = await installDeckShim(page); await page.evaluate(() => document.fonts.ready);

/* in the page: every piece of ink on the stage, with the opacity it is drawn at */
const probe = () => page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(() => {
  const stage = document.getElementById('stage'), ink = [], panels = [], ops = new Map();
  const op = el => { if (ops.has(el)) return ops.get(el); let o = 1; const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') o = 0; else { o = parseFloat(cs.opacity); if (el.parentElement && el !== stage) o *= op(el.parentElement); } ops.set(el, o); return o; };
  const label = el => (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 34);
  const unitOf = el => { const chip = el.closest('.chip'); if (chip) return chip; if (el.ownerSVGElement) return el.closest('text') || el;
    for (let e = el; e; e = e.parentElement) if (!getComputedStyle(e).display.startsWith('inline') || e.classList.contains('board')) return e; return el; };
  const seenChip = new Set(), tw = document.createTreeWalker(stage, NodeFilter.SHOW_TEXT), rg = document.createRange();
  for (let n = tw.nextNode(); n; n = tw.nextNode()) { if (!n.nodeValue.trim()) continue; const el = n.parentElement, o = op(el); if (o < 0.1 || el.closest('.badge')) continue;   /* a badge is a marker set on its scene on purpose */
    const unit = unitOf(el), board = el.closest('.board') || el.closest('.legend') || unit;
    if (unit.classList && unit.classList.contains('chip')) { if (seenChip.has(unit)) continue; seenChip.add(unit); const r = unit.getBoundingClientRect(); if (r.height < 5) continue; ink.push({ u: unit, bd: board, o, l: r.left, r: r.right, t: r.top, b: r.bottom, s: label(unit) }); continue; }
    rg.selectNodeContents(n); for (const r of rg.getClientRects()) { if (r.width < 2 || r.height < 5) continue; const tr = r.height * 0.18;   /* the line box is taller than the letters */
      ink.push({ u: unit, bd: board, o, l: r.left, r: r.right, t: r.top + tr, b: r.bottom - tr, s: label(unit) }); } }
  /* drawings: a shape of a panel's plot outside the panel, a stroke through a label of its own plot */
  const gfx = [];
  for (const svg of stage.querySelectorAll('.board svg')) { if (op(svg) < 0.1) continue; const bd = svg.closest('.board'), P = bd.getBoundingClientRect(), isPanel = bd.classList.contains('panel');
    const texts = [...svg.querySelectorAll('text')].filter(t => op(t) >= 0.3 && t.textContent.trim() && t.getBoundingClientRect().height >= 5).map(t => { const r = t.getBoundingClientRect(), tr = r.height * 0.22; return { o: op(t), l: r.left + 2, r: r.right - 2, t: r.top + tr, b: r.bottom - tr, s: label(t) }; });
    for (const e of svg.querySelectorAll('path,line,polyline,polygon,rect,circle,ellipse')) { const o = op(e); if (o < 0.3 || e.dataset.out) continue;   /* data-out: drawn outside on purpose */ const r = e.getBoundingClientRect(); if (r.width < 1 && r.height < 1) continue;
      const cs = getComputedStyle(e), name = { s: `<${e.tagName}${e.id ? '#' + e.id : ''}> of "${label(bd).slice(0, 22)}"`, o };
      if (isPanel) { const out = Math.max(P.left - r.left, r.right - P.right, P.top - r.top, r.bottom - P.bottom); if (out > 3) gfx.push(['DRAWING outside its own panel', name, null, out, 0]); }
      if (!/^(path|line|polyline)$/.test(e.tagName) || cs.stroke === 'none' || parseFloat(cs.strokeOpacity) < 0.3 || !texts.length || !e.getTotalLength) continue;
      const L = e.getTotalLength(), M = e.getScreenCTM(); if (!L || !M) continue; const hit = new Map();
      for (let i = 0; i <= 240; i++) { const q = e.getPointAtLength(L * i / 240).matrixTransform(M); for (const t of texts) if (q.x > t.l && q.x < t.r && q.y > t.t && q.y < t.b) hit.set(t, (hit.get(t) || 0) + 1); }
      for (const [t, n] of hit) if (n >= 2) gfx.push(['LINE through text', t, name, n, 0]); } }
  for (const e of stage.querySelectorAll('.board.panel')) { const o = op(e); if (o < 0.1) continue; const r = e.getBoundingClientRect(); panels.push({ bd: e, o, l: r.left, r: r.right, t: r.top, b: r.bottom, s: label(e) }); }
  const R = [], key = new Set(), add = (kind, a, c, w, h) => { const k = kind + '|' + a.s + '|' + (c ? c.s : ''); if (key.has(k)) return; key.add(k); R.push({ kind, a: a.s, ao: +a.o.toFixed(2), c: c ? c.s : '', co: c ? +c.o.toFixed(2) : 1, w: Math.round(w), h: Math.round(h) }); };
  const on = q => q.r > 0 && q.l < 1920 && q.b > 0 && q.t < 1080;
  for (let i = 0; i < ink.length; i++) { const A = ink[i]; if (!on(A)) continue;
    const cut = Math.max(-A.l, A.r - 1920, -A.t, A.b - 1080); if (cut > 3) add('CUT by the stage edge', A, null, cut, 0);
    if (A.bd.classList && A.bd.classList.contains('panel')) { const p = A.bd.getBoundingClientRect(), out = Math.max(p.left - A.l, A.r - p.right, p.top - A.t, A.b - p.bottom); if (out > 3) add('OUTSIDE its own panel', A, null, out, 0); }
    for (let j = i + 1; j < ink.length; j++) { const B = ink[j]; if (B.u === A.u) continue; const w = Math.min(A.r, B.r) - Math.max(A.l, B.l), h = Math.min(A.b, B.b) - Math.max(A.t, B.t); if (w > 3 && h > 3) add('TEXT on text', A, B, w, h); }
    for (const P of panels) { if (P.bd === A.bd) continue; const w = Math.min(A.r, P.r) - Math.max(A.l, P.l), h = Math.min(A.b, P.b) - Math.max(A.t, P.t); if (w > 3 && h > 3) add('TEXT under or over a panel', A, P, w, h); } }
  /* two boards of words set in one place: their lines may miss each other and still read as one jumble (a caption between the two
     lines of a name). The ink box of each plain board against the others': a third of the smaller one covered is a hit */
  const box = new Map(); for (const q of ink) { if (!on(q) || !q.bd.classList || q.bd.classList.contains('panel')) continue; const h = (q.b - q.t) * 0.18 / 0.64;   /* back to the full line box */
    const B = box.get(q.bd) || box.set(q.bd, { l: 1e9, r: -1e9, t: 1e9, b: -1e9, o: q.o, s: label(q.bd) }).get(q.bd); B.l = Math.min(B.l, q.l); B.r = Math.max(B.r, q.r); B.t = Math.min(B.t, q.t - h); B.b = Math.max(B.b, q.b + h); }
  const BX = [...box.values()]; for (let i = 0; i < BX.length; i++) for (let j = i + 1; j < BX.length; j++) { const A = BX[i], B = BX[j], w = Math.min(A.r, B.r) - Math.max(A.l, B.l), h = Math.min(A.b, B.b) - Math.max(A.t, B.t);
    if (w > 3 && h > 3 && w * h > 0.33 * Math.min((A.r - A.l) * (A.b - A.t), (B.r - B.l) * (B.b - B.t))) add('BOARD in board', A, B, w, h); }
  for (const [kind, a, c, w, h] of gfx) add(kind, a, c, w, h);
  done(R); }))));

let strong = 0, faint = 0, midN = 0; const seen = new Map();
for (let s = 0; s < inv.scenes; s++) { const id = 'S' + String(s + 1).padStart(2, '0'); if (only && only !== id) continue;
  for (let k = 1; k <= inv.per_scene[s]; k++) for (const p of mid ? [...PS, null] : [null]) {
    await page.evaluate(([s, k, p]) => { window.__deck.show(s); if (p == null) window.__deck.applyUpTo(k); else { window.__deck.applyUpTo(k - 1); applyState(s, k - 1, p); } }, [s, k, p]);
    for (const r of await probe()) { const lvl = Math.min(r.ao, r.co) >= 0.3 ? 'STRONG' : 'FAINT';
      const line = `${lvl} ${r.kind}: "${r.a}" (${r.ao})` + (r.c ? ` × "${r.c}" (${r.co}), ${r.w}×${r.h} px` : `, ${r.w} px`).replace('×0 px', ' samples');
      const where = `${id}b${k}${p == null ? '' : '@' + p}`; if (!seen.has(line)) seen.set(line, []); seen.get(line).push(where);
      if (p != null) midN++; else if (lvl === 'STRONG') strong++; else faint++; } } }
for (const [line, where] of seen) console.log(`${line}\n      ${where.join(' ')}`);
console.log(`\nend of beats: ${strong} strong, ${faint} faint` + (mid ? `; inside beats: ${midN}` : ''));
await b.close(); process.exit(strong ? 1 : 0);
