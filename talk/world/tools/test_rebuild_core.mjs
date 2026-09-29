// test_rebuild_core.mjs — unit tests for js/rebuild-core.js (ticket 06, the rebuilt S04/S05). No browser.
//   node --test talk/world/tools/test_rebuild_core.mjs      (REBUILD_CORE=<path> tests a mutated copy)
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { createRequire } from 'node:module';
const here = path.dirname(new URL(import.meta.url).pathname);
const C = createRequire(import.meta.url)(process.env.REBUILD_CORE || path.join(here, '../js/rebuild-core.js'));

test('C1 |term| dots shrink, bottom out within the first four, then grow out of the frame; dots 0-2 are the used ones', () => {
  const { h, used } = C.termDots();
  const m = h.indexOf(Math.min(...h));
  assert.ok(m >= 1 && m <= 3, `minimum at dot ${m}`); // storyboard #15: within the first 3-4 dots
  for (let i = 1; i <= m; i++) assert.ok(h[i] < h[i - 1], `shrinks at ${i}`);
  for (let i = m + 1; i < h.length; i++) assert.ok(h[i] > h[i - 1], `grows at ${i}`);
  assert.ok(h.at(-1) > 1, 'the last dot bursts out of the frame (height > 1)');
  assert.ok(h.slice(0, -1).every(x => x > 0), 'every dot above the axis');
  assert.deepEqual(used, [0, 1, 2]); // "used (L = 3)": NNLO truncation
});

test('C2 S05 split: only l = 0, 1, 2 carry a remainder v_l - V_l, signed and to scale (physics review #21: +23, -14, +1 %)', () => {
  const { h } = C.termDots();
  h.forEach((V, l) => {
    const f = C.remainder(l);
    if (l > 2) return assert.equal(f, null, `l=${l} must have no remainder (catalogue S05-F4)`);
    assert.ok(Math.abs(f) < 0.5, `l=${l}: the remainder stays small next to V_l`);
  });
  assert.deepEqual([0, 1, 2].map(l => Math.sign(C.remainder(l))), [1, -1, 1], 'l = 1 lies below: V_1 > v_1');
  assert.ok(C.remainder(2) < C.remainder(0) / 10, 'l = 2 is tiny');
});

const XS = Array.from({ length: 21 }, (_, i) => i / 20);
test('C3 fixed-order bands vs m_Q: LO sits apart below NLO and NNLO everywhere; NLO and NNLO overlap everywhere', () => {
  const B = C.bandCurves(XS);
  XS.forEach((x, i) => {
    const [lo, n1, n2] = ['LO', 'NLO', 'NNLO'].map(k => [B[k].lo[i], B[k].hi[i]]);
    for (const [a, b] of [lo, n1, n2]) assert.ok(a < b && a >= 0 && b <= 1, `band inside the frame at x=${x}`);
    assert.ok(lo[1] < n1[0] && lo[1] < n2[0], `LO apart at x=${x}`);          // Figs. 1-2 left; catalogue S04-F7
    assert.ok(Math.max(n1[0], n2[0]) < Math.min(n1[1], n2[1]), `NLO/NNLO overlap at x=${x}`);
  });
});

// every settled or running state of a scene with n beats: beats < k done, beat k at p, the rest not started
function* sweep(n) {
  for (let k = 0; k < n; k++) for (let i = 0; i <= 20; i++) yield Array.from({ length: n }, (_, j) => j < k ? 1 : j > k ? 0 : i / 20);
}
test('C4 haze: Λ_R hazy from S04 b1 on; the S05 band only fades after it sits in the Λ_R box, then both cancel; 𝒥 never hazy', () => {
  assert.equal(C.haze(4, [0, 0, 0, 0]).lambdaR, 0, 'no Λ_R before S04 b1');
  for (const prog of sweep(4)) {
    const h = C.haze(4, prog);
    assert.equal(h.J, 0);
    if (prog[0] === 1) assert.equal(h.lambdaR, 1, `S04 ${prog}`);
    assert.equal(h.band, 0, 'the S05 band does not exist in S04');
  }
  let arrived = false;
  for (const prog of sweep(5)) {
    const h = C.haze(5, prog);
    assert.equal(h.J, 0, `𝒥 never hazy ${prog}`);
    if (prog[2] === 0) assert.equal(h.lambdaR, 1, `Λ_R still hazy before b3 ${prog}`);
    if (prog[1] === 1 && (h.lambdaR < 1 || h.band < 1)) assert.equal(h.bandPos, 1, `fades only in the box ${prog}`);
    if (prog[1] === 1 && h.bandPos < 1) assert.ok(h.band > 0.5 && h.lambdaR === 1, `band waits, visible ${prog}`);
    if (h.bandPos === 1) arrived = true;
  }
  assert.ok(arrived, 'the band reaches the Λ_R box');
  assert.deepEqual([C.haze(5, [1, 1, 1, 0, 0]).lambdaR, C.haze(5, [1, 1, 1, 0, 0]).band], [0, 0], 'after b3 both hazes are gone');
  assert.equal(C.haze(5, [1, 0, 0, 0, 0]).band, 0, 'no band before S05 b2');
});

test('C13 S05 b3 band flight: from the cut to Λ_R without crossing box 1, the 𝒥 box or the "} +" (physics review round 2, R2-1)', () => {
  // layout of the settled S05 b3 in Chromium at 1920x1080 (evidence/06/stills-chromium/33-S05b3-p25.png): [left, top, right, bottom]
  const a = [998, 872], b = [1580, 224], keepOut = { 'box 1': [240, 184, 838, 356], '𝒥 box': [874, 210, 1498, 324], '} +': [1510, 250, 1538, 282] };
  const W = 170 / 2 + 3, H = 26 / 2 + 3; // band half-size plus its 3 px blur
  let prev = null;
  for (let i = 0; i <= 400; i++) {
    const t = i / 400, f = C.bandFlight(a, b, t), hw = W * f.scale, hh = H * f.scale;
    for (const [name, [l, tp, r, bt]] of Object.entries(keepOut))
      assert.ok(f.x + hw < l || f.x - hw > r || f.y + hh < tp || f.y - hh > bt, `t=${t}: band [${(f.x - hw) | 0}, ${(f.y - hh) | 0}, ${(f.x + hw) | 0}, ${(f.y + hh) | 0}] inside ${name}`);
    if (prev) assert.ok(Math.hypot(f.x - prev.x, f.y - prev.y) < 12, `t=${t}: no jump`);
    prev = f;
  }
  const [s, e] = [C.bandFlight(a, b, 0), C.bandFlight(a, b, 1)];
  assert.deepEqual([s.x, s.y, s.scale], [a[0], a[1], 1], 'starts over the cut, full size');
  assert.deepEqual([e.x, e.y], b, 'ends on Λ_R');
  assert.ok(Math.abs(170 * e.scale - 60) < 2, `lands at the size of Λ_R's own haze (~60 px), got ${170 * e.scale}`);
  // the live "} +" box is wider than its glyphs (lock W, R11 in Chromium): rise in a column right of it, then step over
  const close = [1505, 240, 1556, 292], xc = close[2] + 45;
  prev = null;
  for (let i = 0; i <= 400; i++) {
    const t = i / 400, f = C.bandFlight(a, b, t, xc), hw = W * f.scale, hh = H * f.scale;
    for (const [name, [l, tp, r, bt]] of Object.entries({ ...keepOut, '} +': close }))
      assert.ok(f.x + hw < l || f.x - hw > r || f.y + hh < tp || f.y - hh > bt, `xc: t=${t}: band inside ${name}`);
    if (prev) assert.ok(Math.hypot(f.x - prev.x, f.y - prev.y) < 12, `xc: t=${t}: no jump`);
    prev = f;
  }
  const ex = C.bandFlight(a, b, 1, xc);
  assert.deepEqual([ex.x, ex.y], b, 'xc: still ends on Λ_R');
});

test('C4b the V_R(r) haze band has the same width at every r (a constant, never fanning out)', () => {
  const ys = [-2.1, -1.3, -0.7, -0.4, -0.25, -0.2];
  const { lo, hi } = C.hazeBand(ys, 0.3);
  ys.forEach((y, i) => { assert.ok(Math.abs(hi[i] - lo[i] - 0.3) < 1e-12, `width at ${i}`); assert.ok(lo[i] < y && y < hi[i]); });
});

const [W, H] = [1920, 1080], BP = [4, 0, 0]; // the Borel branch point u = ½ on the floor (x = 8u)
const onScreen = (q, x0 = 0.05, x1 = 0.95, y0 = 0.55, y1 = 0.95) => q.front && q.x >= x0 * W && q.x <= x1 * W && q.y >= y0 * H && q.y <= y1 * H;
const dist = (a, b) => Math.hypot(...a.pos.map((v, i) => v - b.pos[i])) + Math.hypot(...a.target.map((v, i) => v - b.target[i]));

test('C5a project: the camera target lands at the screen centre; a point behind the camera is not in front', () => {
  const cam = C.cameraAt(4, [0, 0, 0, 0]), q = C.project(cam.target, cam, W, H);
  assert.ok(Math.abs(q.x - W / 2) < 1e-6 && Math.abs(q.y - H / 2) < 1e-6 && q.front);
  const back = cam.pos.map((v, i) => 2 * v - cam.target[i]);
  assert.equal(C.project(back, cam, W, H).front, false);
});

test('C5b S04 camera: level with the pair in the top band, then tilts down in b4 to the Borel floor', () => {
  const level = C.cameraAt(4, [0, 0, 0, 0]), end = C.cameraAt(4, [1, 1, 1, 1]);
  assert.equal(dist(C.cameraAt(4, [1, 1, 1, 0]), level), 0, 'no camera move before b4');
  const pair = C.project(C.PAIR, level, W, H);
  // top band, right of the title and left of where the |term| panel's last dot bursts out (animation review round 1, #1)
  assert.ok(pair.front && pair.x > 0.55 * W && pair.x < 0.72 * W && pair.y < 0.12 * H, `pair in the top band ${JSON.stringify(pair)}`);
  assert.ok(C.project(C.PAIR, end, W, H).y < 0, 'the pair leaves the frame upward as the camera tilts down');
  assert.ok(onScreen(C.project(BP, end, W, H), 0.55, 0.85), 'branch point in view, right of centre, lower half');
  assert.equal(C.boardShift(level), 0);
  assert.ok(C.boardShift(end) < -250 && C.boardShift(end) > -400, `board slides up ${C.boardShift(end)} stage px`);
  let prev = level; // continuous: a 0.2 % step of any beat never moves the camera more than 0.25 unit (smooth max ≈ 0.17)
  for (let k = 0; k < 4; k++) for (let i = 0; i <= 500; i++) {
    const c = C.cameraAt(4, [0, 1, 2, 3].map(j => j < k ? 1 : j > k ? 0 : i / 500));
    assert.ok(dist(c, prev) < 0.25, `jump at b${k + 1} p=${i / 500}`); prev = c;
  }
});

test('C5c S04 -> S05 glide: the camera blends from the S04 end to the S05 view; the branch point never leaves the screen', () => {
  const a = C.cameraAt(4, [1, 1, 1, 1]), b = C.cameraAt(5, [0, 0, 0, 0, 0]);
  for (let i = 0; i <= 20; i++) {
    const w = i / 20, v = C.blend([{ w: 1 - w, cam: a }, { w, cam: b }]);
    assert.equal(v.presence, 1);
    assert.ok(onScreen(C.project(BP, v.cam, W, H)), `branch point at w=${w}`);
    if (i === 0) assert.equal(dist(v.cam, a), 0); if (i === 20) assert.equal(dist(v.cam, b), 0);
  }
  for (const prog of sweep(5)) {
    const c = C.cameraAt(5, prog);
    assert.ok(onScreen(C.project(BP, c, W, H), 0.3, 0.65), `S05 branch point at ${prog}`);
    assert.ok(onScreen(C.project([0, 0, 0], c, W, H), 0.05, 0.3), `S05 contour start u = 0 at ${prog}`);
  }
  assert.deepEqual(C.blend([{ w: 0.3, cam: a }]).presence, 0.3, 'leaving to a hosted scene fades the 3D out');
  assert.equal(C.blend([]).presence, 0);
});

const HOSTED = ['<b>b1.</b> Now the problem.', '<b>b2.</b> Here is the picture.',
  '<b>b3.</b> The origin is infrared. It is the same renormalon.', '<b>b4.</b> And here is the symptom. You cannot quote it.'];
test('C6 e12: E2 swaps the origin and symptom bodies (prefixes stay in slot order), E1 closes the origin beat', () => {
  assert.deepEqual(C.e12(HOSTED), ['<b>b1.</b> Now the problem.', '<b>b2.</b> Here is the picture.',
    '<b>b3.</b> And here is the symptom. You cannot quote it.',
    '<b>b4.</b> The origin is infrared. It is the same renormalon. In the Borel plane, it shows up as a branch point at u equals one half.']);
  assert.equal(C.E1, 'In the Borel plane, it shows up as a branch point at u equals one half.'); // story-edits.md E1, verbatim
  assert.throws(() => C.e12(C.e12(HOSTED)), /origin/, 'never applied twice');
  assert.throws(() => C.e12([HOSTED[0], HOSTED[1], '<b>b3.</b> Something else.', HOSTED[3]]), /origin/, 'hosted notes changed: stop');
  assert.deepEqual(HOSTED[2], '<b>b3.</b> The origin is infrared. It is the same renormalon.', 'input untouched');
});

test('C7 spacePlan: start on a rebuilt scene; keep running through the glide away; stop only once it ends; never start under a clip', () => {
  const P = s => C.spacePlan({ want: false, on: false, gliding: false, held: false, ...s });
  assert.equal(P({ want: true }), 'start');
  assert.equal(P({ want: true, gliding: true }), 'start', 'S03 → S04: start as the glide begins, so the pair fades in with it');
  assert.equal(P({ want: true, on: true }), 'none');
  assert.equal(P({ want: true, held: true }), 'none', 'a backup clip covers the scene: no WebGL work');
  assert.equal(P({ on: true, gliding: true }), 'none', 'S05 → S06: the floor fades out with the glide first');
  assert.equal(P({ on: true }), 'stop');
  assert.equal(P({}), 'none');
});

test('C8 once: the hosted-scene shadow replay runs on the first call only, whichever beat function calls first', () => {
  let n = 0; const go = C.once(() => { n++; return 'ran'; });
  assert.equal(n, 0, 'nothing runs at build time');
  go(); go(); go();
  assert.equal(n, 1);
});

const WC = createRequire(import.meta.url)(path.join(here, '../js/world-core.js'));
// a 30 fps recording of a 2-beat scene: opening held 0.2 s, beat 1 runs, held, beat 2 runs, held
const fr = (from, to, done, run) => Array.from({ length: Math.round((to - from) * 30) }, (_, i) => ({ t: +(from + i / 30).toFixed(4), done, run }));
const LOG = [...fr(0, 0.2, 0, null), ...fr(0.2, 1, 0, 0), ...fr(1, 1.2, 1, null), ...fr(1.2, 2, 1, 1), ...fr(2, 2.2, 2, null)];
test('C9 marksFrom: one mark per settled beat plus the opening, at the first settled frame; fits the manifest contract', () => {
  const m = C.marksFrom(LOG, 2);
  assert.deepEqual(m, [0, 1, 2]);
  assert.ok(WC.validClip({ src: 'clips/x.webm', marks: m }, 2));
  assert.throws(() => C.marksFrom(LOG, 3), /beat 3/, 'a beat never settled');
  const short = LOG.filter(f => !(f.done === 1 && f.run === null && f.t > 1.05)); // beat 1 held < 0.1 s
  assert.throws(() => C.marksFrom(short, 2), /hold/);
  assert.throws(() => C.marksFrom(LOG.filter(f => f.t < 2.05), 2), /hold/, 'the last mark needs its hold too');
});

test('C11 rebuild.css touches only rebuilt content: every selector starts .w-rebuilt, #w-floor-overlay or #space.w-', async () => {
  const fs = await import('node:fs');
  const css = fs.readFileSync(process.env.REBUILD_CSS || path.join(here, '../rebuild.css'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
  const sels = [...css.matchAll(/([^{}]+)\{[^{}]*\}/g)].map(m => m[1].trim()).filter(s => !s.startsWith('@') && !/^(from|to|\d+%)$/.test(s));
  assert.ok(sels.length > 0, 'no rules found');
  for (const s of sels.flatMap(x => x.split(',')).map(x => x.trim()))
    assert.match(s, /^(\.w-rebuilt|#w-floor-overlay|#space\.w-)/, `unscoped selector: ${s}`);
});

test('C12 the easing copies in rebuild-core.js equal the Deck originals in js/00-helpers.js', async () => {
  const fs = await import('node:fs'), vm = await import('node:vm');
  const src = fs.readFileSync(path.join(here, '../js/00-helpers.js'), 'utf8') + '\n;({ clamp, seg, eio });';
  const D = vm.runInNewContext(src, { window: {}, document: { getElementById: () => null } });
  const xs = Array.from({ length: 41 }, (_, i) => -0.25 + i * 0.0375);
  for (const x of xs) {
    assert.equal(C._ease.clamp(x), D.clamp(x)); assert.equal(C._ease.eio(x), D.eio(x));
    assert.equal(C._ease.seg(x, 0.2, 0.7), D.seg(x, 0.2, 0.7));
  }
});
