/* world.js — the World shell (ticket 04), on top of the Deck engine's bare globals (scenes, si, done, run, show, next…),
   which it only reads: swallows A (narrated auto-run) and N (notes rail), shows a scene's backup clip when forced
   (?backup=4,5 or V), keeps the presenter window (window.opener, see presenter.js) in sync, and exposes window.__world.
   Wrapped in an IIFE: a top-level name here that matched a Deck global would kill this script. */
(function () {
  'use strict';
  const C = WorldCore;
  const S = scenes.map(sc => ({ title: (sc.el.querySelector('h1') || sc.el.querySelector('.eyebrow')).textContent.trim(),
    n: sc.beats.length, notes: sc.notes || [] }));
  let t0 = null; // first advance (ms), for the presenter's elapsed clock

  /* ---- backup clips (contract in clips/manifest.js). The Deck keeps running underneath: its beat drives the clip ---- */
  const CLIPS = window.WORLD_CLIPS || {}, box = document.getElementById('backup'), vids = new Map();
  const forced = new Set(C.parseBackupParam(location.search, S.length));
  let shownAt = null, stopAt = null; // {si,k} the clip shows (null = overlay off); clip time to hold at after 'play'
  const clipOf = i => { const e = CLIPS[String(i + 1)]; return i < S.length && C.validClip(e, S[i].n) ? e : null; };
  // ponytail: a visited scene's <video> is kept; drop the ones > 1 scene away if memory ever matters
  function video(i) {
    if (!clipOf(i)) return null;
    if (!vids.has(i)) {
      const v = document.createElement('video'); // muted: a key forwarded by the presenter is no user gesture here
      Object.assign(v, { muted: true, playsInline: true, preload: 'auto', src: clipOf(i).src });
      box.appendChild(v); vids.set(i, v);
    }
    return vids.get(i);
  }
  const clipStatus = v => !v ? 'none' : v.error ? 'error' : v.readyState >= 4 ? 'ready' : 'loading';
  function syncBackup() {
    const v = video(si); video(si + 1); // preload the current and next scene's clip (ticket 03: warm swap ≤ 2 frames)
    if (!(forced.has(si) && v && !v.error)) {
      if (shownAt) { box.classList.remove('on'); for (const x of vids.values()) { x.pause(); x.classList.remove('cur'); } }
      shownAt = stopAt = null; WorldSpace.hold(false); return;
    }
    if (!shownAt || shownAt.si !== si) {
      for (const [i, x] of vids) { x.classList.toggle('cur', i === si); if (i !== si) x.pause(); }
      box.classList.add('on'); WorldSpace.hold(true);
    }
    const cur = { si, k: C.beatK(done, run ? run.i : null) }, p = C.clipPlan(shownAt, cur, clipOf(si).marks);
    shownAt = cur;
    if (p.op === 'seek') { v.pause(); v.currentTime = p.t; stopAt = null; }
    else if (p.op === 'play') { stopAt = p.to; v.play().catch(() => {}); }
    if (stopAt !== null && v.currentTime >= stopAt) { v.pause(); v.currentTime = stopAt; stopAt = null; } // snap to the mark
  }

  function state() {
    const running = run ? run.i : null, v = C.beatView(S, si, done, running);
    return { type: 'world-state', v: 1, si, done, running, n: S.length, title: S[si].title, now: v.now, next: v.next, t0,
      backup: { forced: [...forced].sort((a, b) => a - b).map(i => i + 1), clip: clipStatus(vids.get(si)), shown: !!shownAt } };
  }
  /* capture phase on window runs before the Deck's own keydown listener (bubble phase on window) */
  addEventListener('keydown', ev => {
    const a = C.keyAction(ev.key);
    if (a === 'block') ev.stopImmediatePropagation();
    else if (a === 'backup') { ev.stopImmediatePropagation(); if (clipOf(si)) forced.has(si) ? forced.delete(si) : forced.add(si); }
  }, true);
  /* presenter link: state on every change + a 1 s heartbeat (re-pairs after either window reloads); keys it forwards
     are replayed as keydown on document, so they take the same path as the clicker's. targetOrigin '*' because
     file:// origins are opaque; only the presenter holds a handle to this window, and only its keys are accepted. */
  function send() { const o = window.opener; if (o && !o.closed) o.postMessage(state(), '*'); }
  addEventListener('message', ev => {
    const d = ev.data;
    if (ev.source !== window.opener || !C.isWorldMsg(d) || d.type !== 'world-key' || !C.forwards(d.key)) return;
    document.dispatchEvent(new KeyboardEvent('keydown', { key: d.key, bubbles: true, cancelable: true }));
  });
  setInterval(send, 1000);
  let last = '';
  (function poll() {
    if (t0 === null && (si || done || run)) t0 = Date.now();
    syncBackup();
    const b = state().backup, key = [si, done, run ? run.i : -1, b.forced, b.clip, b.shown].join();
    if (key !== last) { last = key; send(); }
    requestAnimationFrame(poll);
  })();
  window.__world = { state };
})();
