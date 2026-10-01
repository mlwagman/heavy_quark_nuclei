'use strict';
/* util.js: shared helpers. `scenes` lives here so every scene file can push into it
   and the shim can read it as a bare identifier. */
const scenes = [];
const TICKS = [];                                             /* per-frame followers (boards that ride on 3D objects) */

/* theme: every colour lives in css/world.css. ?theme=space|paper adds a class on <html> before any script reads a colour. */
{ const t = new URLSearchParams(location.search).get('theme'); if (t) document.documentElement.classList.add('theme-' + t.replace(/[^a-z]/g, '')); }
const _cs = getComputedStyle(document.documentElement);
const TV = n => _cs.getPropertyValue('--' + n).trim();                                /* raw CSS value */
const TC = n => parseInt(TV(n).slice(1), 16);                                         /* '#rrggbb' → three.js hex */
const TN = n => parseFloat(TV(n));
const TBLEND = () => TV('blend') === 'add' ? THREE.AdditiveBlending : THREE.NormalBlending;

const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const seg = (p, a, b) => clamp((p - a) / (b - a));
const eo = u => 1 - Math.pow(1 - u, 3);                 /* ease out */
const eio = u => u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; /* ease in-out */
const back = u => { if (u <= 0) return 0; if (u >= 1) return 1; const c = 1.70158, s = c + 1; return 1 + s * Math.pow(u - 1, 3) + c * Math.pow(u - 1, 2); };
const lerp = (a, b, u) => a + (b - a) * u;
const fmt = (x, n) => x.toFixed(n);

/* one beat played as several that each stop: ds = the pieces' durations (summing to the beat's d). Every piece plays the same f
   over its share of p, so each piece starts where the one before stopped. The camera: by default each piece flies its share of
   the beat's own path (cut where that path holds still); cams[i] gives a piece its own {cam, win, camU}, or null to hold */
function cut(B, ds, cams = []) {
  let a = 0;
  return ds.map((d, i) => { const c0 = a / B.d, c1 = (a += d) / B.d, m = q => c0 + (c1 - c0) * q, P = { d, slow: B.slow, f: q => B.f(m(q)) };
    if (cams[i] !== undefined) { if (cams[i]) Object.assign(P, cams[i]); }
    else P.camAt = (q, idx) => pathCam(B, m(q), idx - i > 0 ? endCam(idx - i - 1) : CAM0);
    return P; });
}

/* DOM */
const D = (tag, cls, parent, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; if (parent) parent.appendChild(e); return e; };
const NS = 'http://www.w3.org/2000/svg';
const S = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs || {}) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const rv = (el, u, dy = 10) => { el.style.opacity = u; el.style.transform = `translateY(${((1 - u) * dy).toFixed(2)}px)`; };
const op = (el, u) => { el.style.opacity = u; };
const dl = (path, u) => { /* draw an svg stroke from 0 to u */
  if (!path.__len) { try { path.__len = path.getTotalLength() || 1; } catch (e) { path.__len = 1000; } }
  path.style.strokeDasharray = path.__len; path.style.strokeDashoffset = path.__len * (1 - clamp(u)); };

/* math markup (HTML strings) */
const frac = (n, d) => `<span class="frac"><span>${n}</span><span class="den">${d}</span></span>`;
const sum = (lo, hi) => `<span class="sum"><span class="lim">${hi || ''}</span><span class="sg">Σ</span><span class="lim">${lo || ''}</span></span>`;
const sub = s => `<sub>${s}</sub>`;
const sup = s => `<sup>${s}</sup>`;
const ov = s => `<span class="ov">${s}</span>`;
const I = s => `<i>${s}</i>`;
/* hadron names as HTML */
const H = {
  cc: `c${ov('c')}`, bb: `b${ov('b')}`, bc: `b${ov('c')}`, tc: `t${ov('c')}`, tb: `t${ov('b')}`,
  et: `η${sub('t')}`, Bc: `B${sub('c')}`,
  Occc: `Ω${sub('ccc')}`, Occb: `Ω${sub('ccb')}`, Ocbb: `Ω${sub('cbb')}`, Obbb: `Ω${sub('bbb')}`,
  Otcc: `Ω${sub('tcc')}`, Otcb: `Ω${sub('tcb')}`, Otbb: `Ω${sub('tbb')}`, Ottc: `Ω${sub('ttc')}`, Ottb: `Ω${sub('ttb')}`, Ottt: `Ω${sub('ttt')}`,
  T4: `QQ${ov('Q′')}${ov('Q′')}`, bbtt: `bb${ov('t')}${ov('t')}`, bctt: `bc${ov('t')}${ov('t')}`, cctt: `cc${ov('t')}${ov('t')}`,
  bbbb: `bb${ov('b')}${ov('b')}`, ccbb: `cc${ov('b')}${ov('b')}`,
  mQ: `${I('m')}${sub('Q')}`, mc: `${I('m')}${sub('c')}`, mb: `${I('m')}${sub('b')}`, mt: `${I('m')}${sub('t')}`,
  as: `${I('α')}${sub('s')}`, LQCD: `Λ${sub('QCD')}`, LR: `Λ${sub('R')}`, mup: `${I('μ′')}`, mucut: `${I('μ')}${sub('cut')}`,
  VR: `${I('V')}${sub('R')}`, CR: `${I('C')}${sub('R')}`, CF: `${I('C')}${sub('F')}`,
};

/* deterministic pseudo-random (so scrubbing a beat is reproducible) */
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
