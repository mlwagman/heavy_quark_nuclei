'use strict';
/* boards.js: crisp DOM boards anchored to 3D points and re-projected every frame.
   A board is a billboard: it always faces the camera; its size follows perspective.
   Focus: the boards the current beat touched are full size and opacity; the rest of the scene is dimmed
   and slightly smaller; the previous scene's boards stay visible in the distance; older scenes are hidden,
   except in the overview, where every scene up to the current one is shown at its exploded station. */
const GHOST = 0.10;                                         /* opacity of the previous scene's boards: tune for the room */
const EDGE = 90;                                            /* stage px over which a board the beat is not showing fades once the stage edge would cut it */
const BOARDS = (() => {
  const layer = document.getElementById('boards');
  const all = [];
  let APPLY = -1;                                             /* index of the beat whose f() is running (set by the engine) */
  const sceneEls = [];
  const sceneEl = i => { while (sceneEls.length <= i) { const e = D('div', 'scene', layer); e.style.cssText = 'position:absolute;left:0;top:0;width:0;height:0;overflow:visible'; sceneEls.push(e); } return sceneEls[i]; };
  class Board {
    /* o: {html, px (css width), w (world width), at [x,y,z], cls, scene (owner index or null = persistent), z (bias),
       keep (the scene's main board: it stays at full strength while its scene is the current one)} */
    constructor(o) {
      this.el = D('div', 'board' + (o.cls ? ' ' + o.cls : ''), o.scene != null ? sceneEl(o.scene) : layer, o.html || '');
      this.el.style.width = (o.px || 600) + 'px';
      this.px = o.px || 600; this.w = o.w || 20; this.a = o.a ?? 0; this.scene = o.scene ?? null;
      this.anchor = W.V3(...(o.at || [0, 0, 0])); this.off = [o.ox || 0, o.oy || 0];
      this.h = 0; this.zb = o.z || 0; this.hidden = false; this.touched = -1; this.prev = -1; this.fk = 0; this.fixed = !!o.at; this.keep = !!o.keep;   /* fixed: placed by `at`, counts for the scene's overview thumbnail box */
      all.push(this); this.measure();
    }
    measure() { this.h = this.el.offsetHeight; return this; }
    set(x, y, z) { this.anchor.set(x, y, z); return this; }
    setV(v) { this.anchor.copy(v); return this; }
    lerpTo(a, b, u) { this.anchor.lerpVectors(a, b, u); return this; }
    show(a) { a = clamp(a); this.a = a; if (APPLY >= 0 && a > 0) this.touch(); return this; }
    touch() { if (APPLY !== this.touched) { this.prev = this.touched; this.touched = APPLY; } return this; }   /* prev: the beat that showed it before this one */
    html(s) { this.el.innerHTML = s; this.measure(); return this; }
    q(sel) { return this.el.querySelector(sel); }
    qa(sel) { return [...this.el.querySelectorAll(sel)]; }
    /* project to the stage. st: {tier, focus, ov, gk} from the engine */
    update(st) {
      const cur = st.tier === 'cur' || st.tier === 'pers';
      /* a scene does not turn into a ghost on the click: over the first part of the next scene's first beat its boards go from what
         they were (st.was) to the ghost level, and the ghosts of the scene before go to nothing (st.enter runs 0 → 1) */
      let mult = cur ? (0.42 + 0.58 * st.focus) : st.tier === 'prev' ? lerp(0.42 + 0.58 * st.was, GHOST * st.gk, st.enter) : st.tier === 'old2' ? lerp(GHOST * st.gk, 0, st.enter) : 0;
      let sc = cur ? (0.9 + 0.1 * st.focus) : st.tier === 'prev' ? lerp(0.9 + 0.1 * st.was, 0.86, st.enter) : 0.86;
      const ov = st.ov;
      if (ov > 0) { const show = st.tier !== 'future' ? 0.9 : 0; mult = lerp(mult, show, ov); sc = lerp(sc, 1, ov); }   /* overview: every scene so far, at its real place and size */
      let op = this.a * mult;
      if (op <= 0.003 || this.hidden) { if (this.el.style.visibility !== 'hidden') this.el.style.visibility = 'hidden'; return; }
      const P = W.project(this.anchor);
      if (P.behind || P.depth < 1) { this.el.style.visibility = 'hidden'; return; }
      const s = W.ppu(P.depth) * this.w / this.px * sc;
      if (this.h === 0) this.measure();
      const x = P.x - (this.px / 2 - this.off[0]) * s, y = P.y - ((this.hung ? 0 : this.h / 2) - this.off[1]) * s;   /* hung: the anchor is the middle of the top edge */
      /* never shown cut: a board the beat is not showing fades as it crosses the stage edge (a focus board stays: --frame watches those) */
      const out = Math.max(-x, x + this.px * s - 1920, -y, y + this.h * s - 1080);
      if (out > 0) { op *= lerp(clamp(1 - out / EDGE), 1, cur ? st.focus : 0); if (op <= 0.003) { this.el.style.visibility = 'hidden'; return; } }
      this.el.style.visibility = 'visible';
      this.el.style.opacity = op.toFixed(3);
      this.el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px) scale(${s.toFixed(4)})`;
      if (ov > 0) this.el.style.setProperty('--inv', (1 / s).toFixed(1));
      this.el.style.zIndex = String(Math.max(1, Math.round(9000 - P.depth * 10 + this.zb + (cur ? 200 * st.focus : 0))));
    }
  }
  const _t = new THREE.Vector3();
  function board(o) { const b = new Board(o); b.qa('img').forEach(im => im.addEventListener('load', () => b.measure())); return b; }
  /* a row of boards hangs from one top edge: their heights differ with their words, and centred on one line their tops were ragged.
     The anchor of a hung board is the middle of its top edge, so the tops stay level at any scale (a board far off the axis is drawn
     smaller). The tallest keeps its place; re-done whenever the boards are measured again (fonts, images) */
  const rows = [];
  const hang = ([list, y]) => { const top = y + Math.max(...list.map(b => b.h * b.w / b.px)) / 2; list.forEach(b => { b.hung = true; b.anchor.y = top; }); };
  function topAlign(...list) { const r = [list, list[0].anchor.y]; rows.push(r); hang(r); }
  const remeasure = () => { all.forEach(b => b.measure()); rows.forEach(hang); };
  addEventListener('load', remeasure); if (document.fonts) document.fonts.ready.then(remeasure);
  /* per-frame update; stFn(board) gives {tier, focus, ov, gk} */
  function update(stFn) { for (const b of all) b.update(stFn(b)); }
  const T = (b, sel, u, dy) => { for (const e of b.qa(sel)) rv(e, u, dy); };
  return { board, update, all, T, layer, sceneEl, topAlign, setApply: i => { APPLY = i; }, touch: b => b.touch() };
})();
