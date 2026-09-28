/* world-core.js — the World's pure logic (no DOM): keys, beat labels, presenter view, backup-clip plan.
   Classic script in the page (global WorldCore), CommonJS in Node for tools/test_world_core.mjs. */
(function (root, factory) {
  const m = factory();
  if (typeof module === 'object' && module.exports) module.exports = m; else root.WorldCore = m;
})(this, function () {
  'use strict';
  const ACTION = { ' ': 'next', ArrowRight: 'next', PageDown: 'next', ArrowLeft: 'prev', PageUp: 'prev',
    a: 'block', A: 'block', n: 'block', N: 'block', v: 'backup', V: 'backup' };
  /* 'next'/'prev' pass through to the Deck engine; 'block' is swallowed (no auto-run, no rail on the projector) */
  function keyAction(key) { return ACTION[key] || null; }
  const FORWARD = new Set([' ', 'ArrowRight', 'PageDown', 'ArrowLeft', 'PageUp', 'ArrowUp', 'ArrowDown', 'v', 'V']);
  /* keys the presenter window sends on to the World (so the clicker works whichever window has focus) */
  const forwards = key => FORWARD.has(key);
  const label = (si, k) => 'S' + String(si + 1).padStart(2, '0') + 'b' + k;
  /* beat the speaker is on: a running beat counts (it was clicked); else the settled count (Deck `run`/`done`) */
  const beatK = (done, running) => running == null ? done : running + 1;
  /* S = [{title, n, notes}]; now/next for the presenter. note null = scene opening (no beat revealed yet) */
  function beatView(S, si, done, running) {
    const k = beatK(done, running), sc = S[si];
    const now = { label: label(si, k), note: k > 0 ? sc.notes[k - 1] ?? '' : null };
    const next = k < sc.n ? { label: label(si, k + 1), note: sc.notes[k] ?? '', title: sc.title }
      : si + 1 < S.length ? { label: label(si + 1, 0), note: null, title: S[si + 1].title } : null;
    return { now, next };
  }
  /* ?backup=4,5 (scene numbers as the speaker says them) → [3,4]; junk ignored */
  function parseBackupParam(search, nScenes) {
    const v = new URLSearchParams(search).get('backup') || '';
    return [...new Set(v.split(',').map(Number).filter(x => Number.isInteger(x) && x >= 1 && x <= nScenes))]
      .map(x => x - 1).sort((a, b) => a - b);
  }
  /* manifest entry {src, marks:[t0..tn]}: tk = clip time (s) at which beat k is settled, t0 = scene opening */
  function validClip(e, nBeats) {
    const m = e && e.marks;
    return !!(e && typeof e.src === 'string' && e.src && Array.isArray(m) && m.length === nBeats + 1
      && m.every((t, i) => Number.isFinite(t) && t >= 0 && (i === 0 || t > m[i - 1])));
  }
  /* what the clip does when the World moves from beat `prev` to `cur` ({si,k}; prev null = clip just shown):
     one beat forward plays on to the next mark (the caller stops it there); anything else seeks and holds */
  function clipPlan(prev, cur, marks) {
    if (prev && prev.si === cur.si) {
      if (cur.k === prev.k) return { op: 'none' };
      if (cur.k === prev.k + 1) return { op: 'play', to: marks[cur.k] };
    }
    return { op: 'seek', t: marks[cur.k] };
  }
  function fmtClock(ms) {
    const s = Math.max(0, Math.floor(ms / 1000)), h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, ss = String(s % 60).padStart(2, '0');
    return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
  }
  /* World → presenter {type:'world-state', v:1, …}; presenter → World {type:'world-key', key} */
  const isWorldMsg = d => !!d && typeof d === 'object'
    && ((d.type === 'world-state' && d.v === 1) || (d.type === 'world-key' && typeof d.key === 'string'));
  return { keyAction, forwards, label, beatK, beatView, parseBackupParam, validClip, clipPlan, fmtClock, isWorldMsg };
});
