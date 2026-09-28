/* presenter.js — the presenter view (ticket 04). Opens the World as a popup (a click-opened popup passes every
   popup blocker) and talks to it by postMessage, the one channel that works between two file:// pages in stock
   Firefox (BroadcastChannel and storage events do not). The World's state arrives on every change and every 1 s,
   so either window can be reloaded and they re-pair. */
(function () {
  'use strict';
  const C = WorldCore, $ = id => document.getElementById(id);
  let world = null, st = null, seen = 0, reset = null;

  $('open-world').onclick = () => { world = window.open('world.html' + location.search, 'mrs-world', 'popup,width=1280,height=720'); };
  addEventListener('message', ev => {
    if (!C.isWorldMsg(ev.data) || ev.data.type !== 'world-state') return;
    world = ev.source; st = ev.data; seen = Date.now(); render();
  });
  addEventListener('keydown', ev => {
    if (!C.forwards(ev.key) || !world) return;
    ev.preventDefault(); world.postMessage({ type: 'world-key', key: ev.key }, '*');
  });
  $('elapsed').onclick = () => { reset = Date.now(); tick(); };

  function render() {
    $('title').textContent = st.title;
    $('now-label').textContent = st.now.label;
    $('now-note').innerHTML = st.now.note ?? '<i>scene opening</i>';
    $('next-label').textContent = st.next ? st.next.label : 'end';
    $('next-note').innerHTML = !st.next ? '' : st.next.note ?? '<i>next scene: ' + st.next.title + '</i>';
    const b = st.backup;
    $('backup').textContent = !b ? '' : 'backup: ' + (b.forced.length ? 'scenes ' + b.forced.join(',') : 'none forced')
      + (b.clip === 'none' ? '' : ' · this scene ' + b.clip + (b.shown ? ' (SHOWN)' : ''));
  }
  function tick() {
    const now = Date.now(), d = new Date(now), base = reset ?? (st && st.t0);
    $('wall').textContent = String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
    $('elapsed').textContent = C.fmtClock(base ? now - base : 0);
    const ok = now - seen < 2500; // 1 s heartbeat + slack for a busy World frame
    $('status').textContent = ok ? 'World connected' : 'no World'; $('status').className = ok ? 'ok' : 'lost';
  }
  setInterval(tick, 250); tick();
})();
