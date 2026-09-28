#!/usr/bin/env node
// check_rebuilt_text.mjs — the rebuilt S04/S05 show exactly the approved words (ticket 06). Browser job.
//   NODE_PATH=$(npm root -g) node talk/world/tools/check_rebuilt_text.mjs [--page world.html] [--narration DIR] [--out report.json]
// T1 every approved on-screen string (evidence/06/screen-text.md after step 1e; written here in DOM text order, i.e.
//    a fraction reads numerator then denominator, a sum reads upper limit, Σ, lower limit) is VISIBLE in its scene's
//    section in at least one state: every settled beat, and every beat at p = 0.5 and 0.6 (text nodes whose element has
//    an effective opacity, the product over its ancestors, of at least 0.3); compared with all whitespace removed. T2 no FORBIDDEN phrase. T3 every number on screen is on the scene's list
//    (one token per text node, so a stacked fraction's 1 over 2β0 is two tokens). With --narration:
// T4 each rebuilt scene's notes == narration_video/scene_NN.md, word for word; no em dash in a beat body (Piper).
// Settles each scene on its last beat first. Exit 0 PASS, 1 FAIL, 2 setup error.
import fs from 'node:fs';
import path from 'node:path';
import { launch, openWorld, glideEnd, frames } from './world_test_lib.mjs';
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const PAGE = opt('--page', 'world.html'), NARR = opt('--narration', null), OUT = opt('--out', null);

const ROWS = {
  4: ['Scene 4 / 15 — Background II', 'The renormalon problem', 'The series', 'VR(r) =−CRrΣlvl αsl+1(μ)', '+ ΛR → 𝒱R (Eq. A.1)',
    'Factorial growth', 'vl ~ V0 (2β0)lΓ(l+1+b)Γ(1+b)', 'the series is asymptotic, not convergent', 'colour singlet — schematic',
    '|term| vs order', 'ℓ (order)', '|vℓ αsℓ+1| (schematic)', 'schematic — terms shrink, reach a minimum, grow', 'minimal term ⇒ δV ~ ΛQCD',
    'used (L = 3)', 'Symptom',
    'Varying μ by 2× moves Ωbbb by up to ~700 MeV at NNLO (Tab. III); in binding energy at fixed mQ, LO does not even overlap NLO/NNLO (Figs. 1–2)',
    'target precision', '~10 MeV', 'fixed-order scale dependence', '~700 MeV (Ωbbb mass, NNLO, Tab. III)', 'Bands at successive orders — fixed order',
    '|ΔE| (schematic)', 'binding energy vs mQ (Figs. 1–2 left, schematic)', 'NNLO', 'LO apart — no meaningful uncertainty', 'Origin — infrared',
    'Infrared sensitivity (momenta ~ ΛQCD) ⇒ leading ambiguity δV ~ ΛQCD — a constant, same at every r', 'singlet: the same renormalon as the pole mass',
    'perturbative sum VR(r) (Eq. A.1)', 'F(r) = dV/dr (Eq. A.3)', 'no constant (leading-renormalon) ambiguity', 'taking a derivative removes the constant', 'constantschematic',
    'Borel plane (textbook) — leading IR singularity at u = ½, a branch point'],
  5: ['Scene 5 / 15 — Background III', 'Minimal renormalon subtraction (MRS)',
    'Isolate the leading renormalon into an explicit power correction, and resum the growth (§II B).', 'Vl', 'vl − Vl', 'schematic',
    '𝒱R(r) = −CRr {', '2Σl=0(vl − Vl) αsl+1(μ′(r))', 'V02β0𝒥(b, 12β0αs(μ′(r)))', '} +', 'ΛR',
    'leading renormalon subtracted — no factorial growth at the leading rate. Vl = the renormalon part of vl; the vl are known to lmax = 2 (NNLO)',
    'the full renormalon series, Borel-summed in closed form (§II B)', 'one power correction (Eq. A.15)', 'free of the leading-renormalon ambiguity',
    'Vl series (all l ≥ 0) Borel-summed up to the branch point → 𝒥 (Eq. 4, A.11–A.13)', 'exponent 1 + b', 'at t = 1/(2β₀) (u = ½)',
    '𝒥(c,y) = e−y Γ(−c) γ*(−c,−y) — an incomplete-gamma closed form', 'Where does the renormalon go?',
    'ΛR comes out free of the leading-renormalon ambiguity. For the singlet it is absorbed into the quark mass, tuned so that 2mQ + E matches the spin-averaged 1S states. For non-singlet channels (baryons, tetraquarks) it is neglected — extending MRS there is a scheme choice, left to future work.',
    'combined with ΛR: ΛR now free of the leading-renormalon ambiguity (§II B step 3)', 'Normalization has converged',
    'V0 = 0.771 (NNLO) → 0.772 (N³LO) ✓ (singlet, nf = 4; §II B)', '“Minimal” matters',
    'Same MS inputs — the series is only reorganized: the renormalon tail is isolated and resummed (Komijani ’17; Kronfeld ’23).',
    'What is new', 'Used for quark masses — all flavours except top (Fermilab/MILC/TUMQCD). First use in a pNRQCD bound-state problem.',
    'Borel plane (textbook) — leading IR singularity at u = ½, a branch point'],
};
const FORBIDDEN = [/MSR/, /one explicit constant/i, /make the ambiguity/i, /renormalon-free/i, /resummed to ∞/, /proven for quark/i,
  /\bpole\b(?! mass)/i, /μ → 2μ/, /series now converges/i, /(collects|carries) the (whole )?ambiguity/i, /contaminat/i, /flux tube/i,
  /instanton/i, /gluon cloud/i, /Λ₁/, /Λ1 absorbed/, /non-singlet: ΛR neglected/];
const NUMBERS = { 4: ['0', '1', '2', '3', '4', '10', '15', '700'], 5: ['0', '1', '2', '3', '4', '5', '11', '13', '15', '17', '23', '0.771', '0.772'] };
const key = s => s.replace(/\s+/g, '');

const report = { page: PAGE, pass: false, fails: [], rows: 0 };
let rc = 0;
const browser = await launch('chromium');
try {
  const { pg, errors } = await openWorld(browser, { page: PAGE, init: () => { window.__RENDER = true; } });
  for (const sc of [4, 5]) {
    await pg.evaluate(i => { show(i); applyUpTo(scenes[i].beats.length); }, sc - 1);
    await glideEnd(pg); await frames(pg, 3);
    const got = await pg.evaluate(i => {
      const el = scenes[i].el, w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT), nodes = [];
      for (let n = w.nextNode(); n; n = w.nextNode()) nodes.push(n.textContent);
      return { text: el.textContent, nodes, notes: scenes[i].notes };
    }, sc - 1);
    // the visible text of each state (the rebuilt scenes are pure functions of their beat progress)
    const visible = async () => { await frames(pg, 2); return pg.evaluate(i => {
      const el = scenes[i].el, w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      const op = e => { let o = 1; for (; e && e !== el.parentElement; e = e.parentElement) { const c = getComputedStyle(e);
        if (c.display === 'none' || c.visibility === 'hidden') return 0; o *= parseFloat(c.opacity); } return o; };
      let t = '';
      for (let n = w.nextNode(); n; n = w.nextNode()) if (op(n.parentElement) >= 0.3) t += n.textContent;
      return t;
    }, sc - 1); };
    const nb = await pg.evaluate(i => scenes[i].beats.length, sc - 1), states = [];
    for (let k = 0; k <= nb; k++) {
      await pg.evaluate(k => applyUpTo(k), k); states.push(key(await visible()));
      if (k) for (const p of [0.5, 0.6]) { await pg.evaluate(([k, p]) => { applyUpTo(k - 1); scenes[si].beats[k - 1].f(p); }, [k, p]); states.push(key(await visible())); }
    }
    await pg.evaluate(i => applyUpTo(scenes[i].beats.length), sc - 1);
    const S = `S0${sc}`;
    for (const r of ROWS[sc]) { report.rows++; if (!states.some(v => v.includes(key(r)))) report.fails.push(`T1 ${S} not visible in any state: ${r}`); }
    for (const f of FORBIDDEN) if (f.test(got.text)) report.fails.push(`T2 ${S} forbidden: ${f}`);
    const toks = got.nodes.flatMap(t => t.match(/\d+(?:\.\d+)?/g) || []);
    for (const t of new Set(toks)) if (!NUMBERS[sc].includes(t)) report.fails.push(`T3 ${S} number not on the list: ${t}`);
    if (NARR) {
      const md = fs.readFileSync(path.join(NARR, `scene_0${sc}.md`), 'utf8').split('\n'), beats = [];
      for (const line of md) {
        if (/^##\s+Beat\s+\d+\s+·/.test(line)) beats.push([]); else if (beats.length && !line.startsWith('# ')) beats.at(-1).push(line.trim());
      }
      const want = beats.map(b => b.join(' ').split(/\s+/).filter(Boolean).join(' '));
      const have = got.notes.map(h => h.replace(/^<b>b\d+\.<\/b>\s*/, '').replace(/<[^>]+>/g, '').split(/\s+/).filter(Boolean).join(' '));
      if (want.length !== have.length) report.fails.push(`T4 ${S} beats: narration ${want.length} vs notes ${have.length}`);
      have.forEach((h, j) => { if (h !== want[j]) report.fails.push(`T4 ${S}b${j + 1} notes≠narration`); if (/—/.test(want[j] || '')) report.fails.push(`T4 ${S}b${j + 1} em dash in narration`); });
    }
  }
  if (errors.length) { report.fails.push(`page errors: ${errors.join(' | ')}`); }
  rc = report.fails.length ? 1 : 0;
} catch (e) { report.error = String(e); rc = 2; }
await browser.close();
report.pass = rc === 0;
if (OUT) fs.writeFileSync(OUT, JSON.stringify(report, null, 1));
for (const f of report.fails) console.log(f);
console.log(rc === 0 ? `TEXT PASS rows=${report.rows} page=${PAGE}${NARR ? ' narration=ok' : ''}` : rc === 1 ? `TEXT FAIL ${report.fails.length} page=${PAGE}` : `TEXT SETUP_ERROR ${report.error}`);
process.exit(rc);
