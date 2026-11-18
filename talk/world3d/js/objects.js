'use strict';
/* objects.js: the persistent objects of the World, each created once and owned here.
   Family shelf · static pair · EFT tower · Borel floor + branch point · |term| dots ·
   band gauge · number line · walker cloud · |ψ|² bump, rings, red zone · lattice wall. */

/* ---------- World layout (units are arbitrary; the camera lives 60-150 units away) ---------- */
const LAY = {
  shelfY: 70, topY: 83, shelfZ: 0,
  towerX: -36, floorsY: [1, -10, -21], pairY: -17, pairZ: 0,
  bottomY: -22, borelY: -52,
  resultsY: 104,                                             /* boards above the shelf, S08 onwards */
  /* round 18: every scene has its own place, in talk order along one winding path that the closing overview shows (title top left,
     down the left side, along the bottom, up the right side, in to the tetraquarks, up to the conclusions beside the title and the
     references above it). A scene's code keeps its own coordinates; T[sc] carries them into the World */
  T: [[-2, 185], [-250, 139], [-210, 92], [-233, 62], [-292, -47], [-81, -193], [15, -216], [160, -312], [228, -203], [158, -90],
      [299, -119], [-92, -105], [61, -72], [92, 238], [-170, 41]],
};
const TP = (sc, x, y, z = 0) => [x + LAY.T[sc][0], y + LAY.T[sc][1], z];   /* a point of scene sc in the World, as [x, y, z] */
const TW = (sc, x, y, z = 0) => W.V3(...TP(sc, x, y, z));                  /* ... as a vector */

/* ======================= Family shelf ======================= */
const SHELF = (() => {
  /* the shelf travels with the story: its origin `at` is the place of the scene that uses it (S02, S08, S09, S10, S12, S14; T of that
     scene). Glyph positions are local to it; wp(id) is a glyph's World position */
  const g = new THREE.Group(); W.scene.add(g); const at = W.V3(...TP(1, 0, 0, 0)); g.position.copy(at);
  const mkLine = (x0, x1, y, col, op) => { const l = W.cline([W.V3(x0, y, LAY.shelfZ), W.V3(x1, y, LAY.shelfZ)], col, op); W.crisp.add(l); return l; };
  const lineMain = mkLine(-64, 64, LAY.shelfY - 2.6, TC('wire-dim'), 0.0);   /* under the glyphs, above every name: no line runs through a name or its overbar */
  const lineTop = mkLine(-58, 58, LAY.topY - 1.9, TC('wire-dim'), 0.0);
  lineMain.position.copy(at); lineTop.position.copy(at);
  const glyphs = {};
  const order = { main: [], top: [] };
  const rowY = { main: LAY.shelfY, top: LAY.topY };
  const rowScale = { main: 1, top: 0.62 };
  /* custom geometry for a bound tetraquark: t̄t̄ compact at the centre, lighter quarks outside (S12-I1) */
  const tetraTick = (gg, t) => {
    const u = gg.userData, q = u.quarks, sp = u.split || 0;    /* split > 0: the state falls apart into two mesons (S12 b3, unbound) */
    q[0].position.set(0.55 * Math.cos(t * 1.1) - 3.2 * sp, 0, 0.55 * Math.sin(t * 1.1)); q[1].position.set(-0.55 * Math.cos(t * 1.1) + 3.2 * sp, 0, -0.55 * Math.sin(t * 1.1));
    q[0].scale.setScalar(0.5 + 0.3 * sp); q[1].scale.setScalar(0.5 + 0.3 * sp);
    const a = u.phase + t * u.spin;
    q[2].position.set(2.6 * Math.cos(a) * (1 - sp) - 3.2 * sp, 1.2 * Math.sin(a) * (1 - sp) + 1.4 * sp, 2.6 * Math.sin(a) * 0.5 * (1 - sp));
    q[3].position.set(-2.6 * Math.cos(a) * (1 - sp) + 3.2 * sp, -1.2 * Math.sin(a) * (1 - sp) + 1.4 * sp, -2.6 * Math.sin(a) * 0.5 * (1 - sp));
    for (const l of u.links) { const [i, j] = l.userData.ab; l.set(q[i].position, q[j].position); const cross = !((i === 0 && j === 2) || (i === 1 && j === 3)); l.material.opacity = 0.6 * (u.op ?? 1) * (cross ? 1 - sp : 1); }
  };
  function mk(id, n, o = {}) {
    const gl = W.glyph(n, { R: o.R ?? (n === 2 ? 2.0 : n === 3 ? 2.2 : 2.5), r: o.r ?? 0.8, spin: 0.3 + (Object.keys(glyphs).length % 5) * 0.03, custom: o.tetra ? tetraTick : null });
    gl.userData.id = id; gl.userData.row = o.row || 'main'; gl.userData.sc = rowScale[gl.userData.row] * (o.scale ?? 1);
    gl.scale.setScalar(gl.userData.sc); gl.setOpacity(0); g.add(gl);
    const top = gl.userData.row === 'top';
    const lab = BOARDS.board({ html: `<div style="text-align:center">${o.label}</div>`, px: o.side ? 220 : top ? 200 : 300, w: o.side ? 11 : top ? 10 : 17, cls: top ? 'tiny' : '', a: 0 });   /* top row: 0.05 units per px, in boxes no wider than their names */
    lab.el.style.textAlign = 'center';
    const chip = BOARDS.board({ html: '', px: 220, w: top ? 13.6 : 15, a: 0, cls: top ? 'tiny' : 'small' }); chip.el.style.textAlign = 'center';
    if (o.side) lab.el.firstElementChild.style.cssText = 'text-align:right;font-size:22px';
    glyphs[id] = { g: gl, lab, chip, x: 0, stag: 0, top, side: !!o.side };   /* side: a long name stands left of its glyph (the first of a row), so the chips of its neighbours keep their levels */
    return glyphs[id];
  }
  /* named layouts: which ids sit on which row, in order */
  const layouts = {};
  function layout(name, main, top) { layouts[name] = { main, top }; }
  const xs = (ids, row) => { const n = ids.length; const span = Math.min((row === 'main' ? 12.5 : 8.6) * (n - 1), 112); const m = {}; ids.forEach((id, i) => m[id] = n === 1 ? 0 : -span / 2 + span * i / (n - 1)); return m; };
  /* place every glyph between layouts a and b at fraction u */
  function place(a, b, u) {
    const A = layouts[a], B = layouts[b] || A;
    for (const row of ['main', 'top']) {
      const xa = xs(A[row], row), xb = xs(B[row], row);
      const ids = new Set([...A[row], ...B[row]]);
      for (const id of ids) { const G = glyphs[id]; if (!G) continue;
        const x0 = xa[id] ?? xb[id], x1 = xb[id] ?? xa[id];
        G.x = lerp(x0, x1, u); G.g.position.set(G.x, rowY[row], LAY.shelfZ);
        const ib = B[row].indexOf(id); G.stag = (ib >= 0 ? ib : A[row].indexOf(id)) % 2; }
    }
  }
  /* right-aligned under the line's right end: inside the S02 frame (half-width 66), clear of the line and of the T4 label */
  const END = [57, LAY.shelfY - 3.9, 0];                      /* clear of the last name of the full row (S10) */
  const endLabel = BOARDS.board({ html: `<span class="lbl">cartoon: not a computed wavefunction</span>`, px: 330, w: 17, at: [END[0] + at.x, END[1] + at.y, END[2]], a: 0 });
  endLabel.el.style.textAlign = 'right';
  const hw = b => b.h * b.w / b.px;                            /* a board's height in World units */
  function showLines(a) { lineMain.material.opacity = 0.55 * a; }
  function showTop(a) { lineTop.material.opacity = 0.4 * a; }
  const _w = new THREE.Vector3();
  const wp = id => glyphs[id].g.position.clone().add(at);
  function moveTo(v) { at.copy(v); g.position.copy(v); lineMain.position.copy(v); lineTop.position.copy(v); endLabel.set(END[0] + v.x, END[1] + v.y, END[2] + v.z); }
  const O = sc => W.V3(...TP(sc, 0, 0, 0));                    /* the shelf's place in scene sc */
  const travel = (a, b, u) => moveTo(O(a).lerp(O(b), u));
  const layoutPos = (name, id) => { const L = layouts[name], row = L.main.includes(id) ? 'main' : 'top'; return W.V3(xs(L[row], row)[id], rowY[row], LAY.shelfZ); };   /* where id sits in a named layout, shelf coordinates */
  function tick(t) {
    for (const id in glyphs) { const G = glyphs[id]; if (!G.g.visible) { G.lab.hidden = G.chip.hidden = true; continue; }
      G.lab.hidden = G.chip.hidden = false;
      G.g.userData.live = Math.max(liveOf(id), boardState(G.lab).focus, boardState(G.chip).focus, Math.min(1, 4 * (G.g.userData.pulse || 0)));   /* live: the beat shows its name or its number, or flashes it */
      G.g.tick(t); const p = _w.copy(G.g.position).add(at);
      /* names and chips hang from their top edge, so a second line grows downward: every first line of a row is level, and a
         two-line chip never climbs onto the name above it. The top row's chips alternate between two levels (they are wider than the spacing) */
      if (G.side) G.lab.set(p.x - 2.6 - G.lab.w / 2, p.y + 0.2, p.z); else G.lab.set(p.x, p.y - (G.top ? 2.25 : 3.05) - hw(G.lab) / 2, p.z);
      G.chip.set(p.x, p.y - (G.top ? 4.4 + 2.7 * G.stag : 6.85) - hw(G.chip) / 2, p.z); }
  }
  return { g, at, glyphs, mk, layout, place, tick, showLines, showTop, endLabel, rowY, wp, moveTo, travel, O, layoutPos };
})();

/* ======================= Static pair (S03 → S07) ======================= */
const PAIR = (() => {
  const g = new THREE.Group(); g.position.set(...TP(2, LAY.towerX, LAY.pairY, LAY.pairZ)); W.scene.add(g);
  const A = W.quark(1.15, { glow: 1.9, c: 'r' }), B = W.quark(1.15, { glow: 1.9, c: 'ar' }); g.add(A, B);   /* Q red, Q̄ anti-red: a colour singlet */
  /* meson → baryon (S03 b3): anti-red is green plus blue, so the antiquark splits into a green and a blue quark. The colours change
     while the two spheres still overlap: what is drawn is colour-neutral at every moment (tools/check-colour.mjs) */
  const B_ANTI = new THREE.Color(W.QCOL.ar), B_QUARK = new THREE.Color(W.QCOL.g), C_QUARK = new THREE.Color(W.QCOL.b);
  const lk = W.link({ th: 0.16, opacity: 0.9 }); g.add(lk);
  const C = W.quark(1.15, { glow: 1.9, c: 'b' }); C.material.opacity = 0; g.add(C);           /* the third quark of S03 b3 */
  const lkAC = W.link({ th: 0.16, opacity: 0 }), lkBC = W.link({ th: 0.16, opacity: 0 }); g.add(lkAC, lkBC);
  const tri = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: TC('soft'), transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false })); g.add(tri);
  /* the pair jitters only in the beats that set its state (they are about it): tb is touched like a board, for the same focus */
  const tb = { touched: -1, prev: -1 };
  const st = new Proxy({ sep: 7, cy: 0, jit: 1, fq: 1, lkOp: 1, tri: 0, third: 0, opacity: 1, scale: 1, bOp: 1 }, { set(o, k, v) { o[k] = v; const i = BOARDS.apply(); if (i >= 0 && i !== tb.touched) { tb.prev = tb.touched; tb.touched = i; } return true; } });   /* jit, fq: size and speed of the quarks' own motion; lkOp: the V_R link */
  let ph = 0, lt = 0;
  const base = { A: new THREE.Vector3(), B: new THREE.Vector3(), C: new THREE.Vector3() };
  function tick(tt) {
    ph += clamp(tt - lt, 0, 0.1) * st.fq; lt = tt; const t = ph;   /* phase integrated so that a changing speed never jumps */
    base.A.set(-st.sep / 2, 0, 0); base.B.set(st.sep / 2, 0, 0); base.C.set(0, st.sep * 0.82, 0);
    const j = 0.14 * st.jit * Math.max(liveOf('pair'), boardState(tb).focus);                               /* ambient jitter: static = leading-order potential, the quarks still move */
    A.position.set(base.A.x + j * Math.sin(t * 1.7), base.A.y + j * Math.cos(t * 1.3), j * Math.sin(t * 0.9));
    B.position.set(base.B.x + j * Math.sin(t * 1.5 + 2), base.B.y + j * Math.cos(t * 1.1 + 1), j * Math.cos(t * 0.8));
    C.position.set(base.C.x + j * Math.sin(t * 1.2 + 4), base.C.y + j * Math.cos(t * 1.4 + 3), j * Math.sin(t * 1.0 + 1)).lerp(B.position, 1 - eio(st.third));   /* it leaves from inside the antiquark */
    lk.set(A.position, B.position); lkAC.set(A.position, C.position); lkBC.set(B.position, C.position);
    const kc = seg(st.third, 0, .25); B.material.color.lerpColors(B_ANTI, B_QUARK, kc); B.material.emissive.copy(B.material.color); C.material.color.lerpColors(B_ANTI, C_QUARK, kc); C.material.emissive.copy(C.material.color);
    const o = st.opacity; A.material.opacity = o; B.material.opacity = o * st.bOp; lk.material.opacity = 0.9 * o * st.bOp * st.lkOp; B.visible = st.bOp > 0.003; lk.visible = st.bOp * st.lkOp > 0.003;
    C.material.opacity = seg(st.third, 0, .05) * o; lkAC.material.opacity = lkBC.material.opacity = 0.9 * seg(st.third, .3, 1) * o;
    C.visible = lkAC.visible = lkBC.visible = st.third > 0.002;
    tri.material.opacity = 0.16 * st.tri * o; tri.visible = st.tri > 0.002;
    if (tri.visible) tri.geometry.setFromPoints([A.position, B.position, C.position]);
    g.scale.setScalar(st.scale); g.visible = o > 0.002;
  }
  /* world position of a point on the pair axis at fraction f (0 = A, 1 = B) */
  const _p = new THREE.Vector3();
  const axisPoint = (f, dy = 0) => _p.set((-0.5 + f) * st.sep * st.scale, dy * st.scale, 0).add(g.position).clone();
  /* where the quarks stand, without their own motion (world). Everything placed from the pair (its labels, the S06 ruler, hill and
     gauge, the S07 cloud) is placed from here: placed from the moving quarks, those texts shook by up to 10 px for as long as the
     beat was about the pair (round 26, tools/check-still.mjs). The quarks and their links alone jitter */
  const rest = () => { g.scale.setScalar(st.scale); g.updateWorldMatrix(true, false);
    const w = (x, y) => new THREE.Vector3(x, y, 0).applyMatrix4(g.matrixWorld), k = 1 - eio(st.third);   /* k: the third quark leaves from inside the antiquark */
    return { A: w(-st.sep / 2, 0), B: w(st.sep / 2, 0), C: w(st.sep / 2 * k, st.sep * 0.82 * (1 - k)) }; };
  const posA = () => rest().A;
  return { g, A, B, C, st, tick, axisPoint, posA, rest, lk };
})();

/* ======================= EFT tower (S03) ======================= */
const TOWER = (() => {
  const names = [['QCD', `scale ${H.mQ}`], ['NRQCD', `scale ${H.mQ}${I('v')}`], ['pNRQCD', `scale ${H.mQ}${I('v')}²`]];
  const SCALE = ['teal', 'amber', 'rose'];                       /* m_Q, m_Q v, m_Q v²: the S02 bar colours */
  const floors = names.map((nm, i) => {
    const m = W.plane(52, 40, TC(SCALE[i]), 0.0); m.rotation.x = -Math.PI / 2; m.position.set(...TP(2, LAY.towerX, LAY.floorsY[i], -8)); W.scene.add(m);
    const edge = W.cline([W.V3(-26, 0, -20), W.V3(26, 0, -20), W.V3(26, 0, 20), W.V3(-26, 0, 20), W.V3(-26, 0, -20)], TC(SCALE[i]), 0); edge.position.copy(m.position); W.crisp.add(edge);
    const lab = BOARDS.board({ html: `<div style="font-size:34px;color:var(--${SCALE[i]})">${nm[0]}</div><div class="small ital">${nm[1]}</div>`, px: 300, w: 12, at: TP(2, LAY.towerX - 8.5, LAY.floorsY[i] + 3.6, 10), a: 0 });   /* + 3.6: in the pNRQCD stop the NRQCD floor's far edge ran through 'scale m_Q v' */
    return { m, edge, lab };
  });
  /* rest (S04 onward): the camera has left the tower, but S06 and S07 play on its pNRQCD floor. The floors stay as quiet ground:
     outlines at half strength (below the S06 red zone's, clear of the S07 cloud), and each name stands on the back edge of its
     floor instead of the front one, where those frames hold it whole. Same x and y, so the overview and --layout see no change */
  const st = { a: [0, 0, 0], rest: 0 };
  function apply(i) { const f = floors[i], a = st.a[i], k = st.rest;
    f.m.material.opacity = lerp(0.05, 0.03, k) * a; f.m.visible = a > 0.003; f.edge.material.opacity = lerp(0.7, 0.35, k) * a; f.lab.anchor.z = lerp(10, -28, k); }
  function show(a, i) { st.a[i] = a; apply(i); floors[i].lab.show(a); }
  function rest(k) { st.rest = k; floors.forEach((f, i) => apply(i)); }
  return { floors, show, rest };
})();

/* ======================= Borel plane inset (S04 b4 → S05 b3) =======================
   A small tilted patch. The narration names it in S04 b4 (the branch point) and in S05 (the contour up to it, then the cut past it). place() puts it, show() fades it. */
const BOREL = (() => {
  const gL = new THREE.Group(), gC = new THREE.Group(); W.scene.add(gL); W.crisp.add(gC);
  const X = u => 30 * u - 15;                                   /* local x: the branch point u = ½ sits at the centre */
  const plane = W.plane(64, 28, TC('floor'), 0); plane.rotation.x = -Math.PI / 2; plane.position.set(4.5, 0, 0); plane.renderOrder = -2; gL.add(plane);
  const grid = new THREE.Group(); gC.add(grid);
  for (let x = -27; x <= 36; x += 6) grid.add(W.cline([W.V3(x, .02, -14), W.V3(x, .02, 14)], TC('grid'), 0));
  for (let z = -14; z <= 14; z += 7) grid.add(W.cline([W.V3(-27.5, .02, z), W.V3(36.5, .02, z)], TC('grid'), 0));
  const axis = W.cline([W.V3(X(0), .05, 0), W.V3(X(1.55), .05, 0)], TC('wire'), 0); gC.add(axis);
  const ticks = [0, 0.5, 1, 1.5].map(u => { const l = W.cline([W.V3(X(u), .05, -0.8), W.V3(X(u), .05, 0.8)], TC('wire'), 0); gC.add(l); return l; });
  /* the branch point: a crisp open circle (no bloom) with a wavy cut running right (S04-I3, #17) */
  const circ = (() => { const pts = []; for (let i = 0; i <= 48; i++) { const a = i / 48 * 6.2832; pts.push(W.V3(X(.5) + 1.3 * Math.cos(a), .06, 1.3 * Math.sin(a))); } return W.cline(pts, TC('rose'), 0); })(); gC.add(circ);
  const cut = (() => { const pts = []; for (let i = 0; i <= 90; i++) { const x = X(.5) + 1.3 + i / 90 * 27; pts.push(W.V3(x, .06, 0.55 * Math.sin(i / 90 * 27 * 1.6))); } return W.cline(pts, TC('rose'), 0); })(); gC.add(cut);
  /* the contour from 0 to the branch point (S05 b2) */
  const contour = new THREE.Mesh(new THREE.TubeGeometry(new THREE.LineCurve3(W.V3(X(0), .3, -2.2), W.V3(X(.5) - 1.6, .3, -0.3)), 4, 0.3, 8, false),
    new THREE.MeshBasicMaterial({ color: TC('amber'), transparent: true, opacity: 0, depthWrite: false })); contour.visible = false; contour.renderOrder = 3; gL.add(contour);
  const contourOp = a => { contour.material.opacity = a; contour.visible = a > 0.003; };
  const tl = [[0, '0'], [.5, '½'], [1, '1']].map(([u, t]) => ({ u, b: BOARDS.board({ html: `<span class="tiny">${t}</span>`, px: 40, w: 1.6, a: 0 }) }));
  tl.forEach(t => t.b.el.style.textAlign = 'center');
  const label = BOARDS.board({ html: `<span class="lbl">Borel plane (textbook): leading IR singularity at <i>u</i> = ½, a branch point</span>`, px: 760, w: 33, a: 0, cls: 'panel' });
  const cLabel = BOARDS.board({ html: `<span class="lbl">${I('V')}${sub(I('l'))} series (all ${I('l')} ≥ 0) Borel-summed up to the branch point → 𝒥 (Eq. 4, A.11–A.13)</span>`, px: 720, w: 31, a: 0, cls: 'panel' });
  const _v = new THREE.Vector3();
  /* centre (x, y, z), scale s; lab / cl: offsets of the two labels from the centre */
  function place(x, y, z, s, lab, cl) {
    for (const g of [gL, gC]) { g.position.set(x, y, z); g.scale.setScalar(s); g.rotation.x = 0.9; g.updateMatrixWorld(true); }
    tl.forEach(t => { gC.localToWorld(_v.set(X(t.u), -0.4, 3.4)); t.b.setV(_v); t.b.w = 4.2 * s; });
    label.set(x + lab[0], y + lab[1], z + 1); cLabel.set(x + cl[0], y + cl[1], z + 1);
  }
  function show(a) { plane.material.opacity = 0.92 * a; plane.visible = a > 0.003; grid.children.forEach(l => l.material.opacity = 0.9 * a); axis.material.opacity = 0.9 * a; ticks.forEach(t => t.material.opacity = 0.9 * a);
    circ.material.opacity = a; cut.material.opacity = 0.9 * a; tl.forEach(t => t.b.show(a)); }
  function dimPoint(k) { circ.material.opacity = k; cut.material.opacity = 0.9 * k; }
  const at = u => gC.localToWorld(W.V3(X(u), .06, 0));          /* a point of the u axis, in the World */
  return { place, show, dimPoint, label, cLabel, contour, contourOp, circ, cut, at };
})();

/* ======================= |term| dots (S04 b2 → S05 b2), a persistent DOM board ======================= */
/* World position of a point given in an svg's own coordinates (vbW: the width of its viewBox); the svg sits in board b */
function svgPoint(b, svg, vbW, sx, sy) { const rb = b.el.getBoundingClientRect(), rs = svg.getBoundingClientRect(), k = (rb.width / b.px) || 1, s = b.w / b.px, f = (rs.width / k / vbW) || 1;   /* f: the plot is fitted to the panel */
  return W.V3(b.anchor.x + ((rs.left - rb.left) / k + sx * f - b.px / 2) * s, b.anchor.y - ((rs.top - rb.top) / k + sy * f - b.h / 2) * s, b.anchor.z + 0.5); }
const TERMS = (() => {
  /* heights are schematic (S04-I4); the minimum sits within the first 3-4 dots (#15, S04-U2) */
  const HT = [1.0, 0.66, 0.46, 0.40, 0.46, 0.62, 0.9, 1.35, 2.0, 2.35];   /* the last one lands just above the panel: out of the frame, not adrift on the stage */
  const VF = [0.30, 0.52, 0.70, 0.82, 0.90, 0.95, 0.98, 0.99, 1, 1];  /* V_l / v_l fraction, schematic: the renormalon part dominates at large l */
  const Wd = 760, Hd = 250, X = l => 84 + l * 70, Y = h => 205 - h * 120;   /* the first dot stands clear of the axis */
  const b = BOARDS.board({ px: Wd, w: 30, cls: 'panel', a: 0, html: `<h2 class="t">|term| vs order</h2>` });
  const svg = S('svg', { viewBox: `0 0 ${Wd} ${Hd}`, width: Wd, height: Hd }, b.el);
  S('path', { d: `M60 205 H740 M60 205 V30`, class: 'ax' }, svg);
  const yl = S('text', { x: 66, y: 22, class: 'mt' }, svg); yl.innerHTML = `|<tspan class="it">v</tspan><tspan baseline-shift="sub" font-size="13" style="font-style:italic;font-family:var(--serif)">l</tspan> <tspan class="it">α</tspan><tspan baseline-shift="sub" font-size="12">s</tspan><tspan baseline-shift="super" font-size="12"><tspan font-size="13" style="font-style:italic;font-family:var(--serif)">l</tspan>+1</tspan>| (schematic)`;   /* the order is the italic l of the equation everywhere (round 19) */
  const xl = S('text', { x: 738, y: 232, 'text-anchor': 'end', class: 'mt' }, svg); xl.innerHTML = `<tspan class="it">l</tspan> (order)`;
  const out = l => l === HT.length - 1 ? { 'data-out': 1 } : {};   /* the last term leaves the panel on purpose (tools/check-text.mjs skips what carries data-out) */
  const dots = HT.map((h, l) => S('circle', { cx: X(l), cy: Y(h), r: 0, fill: 'var(--ink)', ...out(l) }, svg));
  const vdots = HT.map((h, l) => S('circle', { cx: X(l), cy: Y(h), r: 0, fill: 'var(--amber)', ...out(l) }, svg));     /* V_l, the renormalon part: amber, as its box caption */
  const rdots = HT.map((h, l) => S('circle', { cx: X(l), cy: Y(h), r: 0, fill: 'var(--teal)' }, svg));      /* v_l − V_l, the remainder: teal, l ≤ 2 only (S05-F4) */
  const minL = S('line', { x1: 60, y1: Y(0.40), x2: 740, y2: Y(0.40), stroke: 'var(--muted)', 'stroke-width': 1.2, 'stroke-dasharray': '7 5', opacity: 0 }, svg);
  const minT = S('text', { x: 420, y: Y(0.40) + 22, 'text-anchor': 'middle', class: 'mt' }, svg); minT.innerHTML = `minimal term ⇒ <tspan class="it">δV</tspan> ~ Λ<tspan baseline-shift="sub" font-size="12">QCD</tspan>`; minT.setAttribute('opacity', 0);
  /* the size of the smallest term, as a length: S04 carries it to the V(r) plot, where it is the width of the ambiguity band */
  const MX = X(3) - 16, MY0 = Y(0.40), MY1 = 205, MARK = `M${MX} ${MY0} V${MY1} M${MX - 4} ${MY0 + 6} L${MX} ${MY0} L${MX + 4} ${MY0 + 6} M${MX - 4} ${MY1 - 6} L${MX} ${MY1} L${MX + 4} ${MY1 - 6}`;
  const mk = S('path', { id: 'tmin', d: MARK, fill: 'none', stroke: 'var(--rose)', 'stroke-width': 2, opacity: 0 }, svg);
  let mkA = 0, mkS = 0; const mkSet = () => mk.setAttribute('opacity', mkA * (1 - mkS)), mark = a => { mkA = a; mkSet(); };
  const brk = S('path', { d: `M${X(0) - 14} 216 v6 H${X(2) + 14} v-6`, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.2, opacity: 0 }, svg);
  const brkT = S('text', { x: X(1), y: 244, 'text-anchor': 'middle', class: 'mt', opacity: 0 }, svg); brkT.innerHTML = `used: <tspan class="it">l</tspan> = 0, 1, 2`;   /* round 19: no L beside the l of the sum */
  const foot = D('div', '', b.el); foot.style.display = 'grid';   /* the caption (S04) and the legend (S05) take turns in one place */
  const cap = D('div', 'lbl t', foot, 'schematic: terms shrink, reach a minimum, grow');
  const legend = D('div', 'small t', foot, `<span class="ac">●</span> ${I('V')}${sub(I('l'))} (renormalon part, grows) &nbsp; <span class="tc">●</span> ${I('v')}${sub(I('l'))} − ${I('V')}${sub(I('l'))} (remainder, stays small; ${I('l')} = 0, 1, 2 only)`);
  cap.style.gridArea = legend.style.gridArea = '1 / 1'; legend.style.fontSize = '19px'; legend.style.whiteSpace = 'nowrap'; b.measure();   /* 19 px: the legend holds on one line (at 20 its last word went to a second) */
  /* S04 b2: dots appear one by one, the last one bursts out of the frame */
  function reveal(u) {
    rv(b.q('h2'), seg(u, 0, .15));
    dots.forEach((c, l) => { const k = back(seg(u, .05 + l * .07, .2 + l * .07)); c.setAttribute('r', Math.max(0, 7 * k)); c.setAttribute('cy', Y(HT[l])); });
    minL.setAttribute('opacity', seg(u, .55, .7)); minT.setAttribute('opacity', seg(u, .6, .75));
    brk.setAttribute('opacity', seg(u, .3, .45)); brkT.setAttribute('opacity', seg(u, .3, .45));
    rv(cap, seg(u, .85, 1)); rv(legend, 0);
  }
  /* S05 b1: each dot splits into V_l (amber) and, for l ≤ 2, the remainder (teal) */
  function split(u) {
    dots.forEach((c, l) => {
      const k = eo(seg(u, .05 + l * .05, .35 + l * .05));
      const hV = HT[l] * VF[l], hR = HT[l] * (1 - VF[l]);
      c.setAttribute('r', 7 * (1 - k));
      vdots[l].setAttribute('r', 6.5 * k); vdots[l].setAttribute('cy', lerp(Y(HT[l]), Y(hV) - 0, k)); vdots[l].setAttribute('cx', X(l) + 6 * k);
      const showR = l <= 2 ? 1 : 0;
      rdots[l].setAttribute('r', 5.5 * k * showR); rdots[l].setAttribute('cy', lerp(Y(HT[l]), Y(hR), k)); rdots[l].setAttribute('cx', X(l) - 6 * k);
    });
    minL.setAttribute('opacity', 1 - u); minT.setAttribute('opacity', 1 - u); mkS = u; mkSet(); rv(cap, 1 - seg(u, 0, .4)); rv(legend, seg(u, .5, .9));   /* the caption is gone before the legend takes its place */
  }
  /* S05 b2: the V_l column gathers at the board's right edge and leaves (a flying cluster carries it into the second box, the sum of the V_l) */
  function fold(u) {
    const k = eio(u);
    vdots.forEach((c, l) => { const hV = HT[l] * VF[l]; c.setAttribute('cx', lerp(X(l) + 6, 700, k)); c.setAttribute('cy', lerp(Y(hV), 110, k)); c.setAttribute('r', 6.5 * (1 - 0.5 * k)); c.setAttribute('opacity', 1 - seg(k, .7, 1)); });
  }
  /* S05 b2: the three remainder dots gather and leave toward the first box */
  function foldR(u) {
    const k = eio(u);
    rdots.forEach((c, l) => { if (l > 2) return; const hR = HT[l] * (1 - VF[l]); c.setAttribute('cx', lerp(X(l) - 6, X(1), k)); c.setAttribute('cy', lerp(Y(hR), 190, k)); c.setAttribute('opacity', 1 - seg(k, .7, 1)); });
  }
  const at = (sx, sy) => svgPoint(b, svg, Wd, sx, sy);   /* World position of a point of the plot */
  return { b, reveal, split, fold, foldR, at, X, mark, markAt: () => at(MX, (MY0 + MY1) / 2), markLen: MY1 - MY0, fit: () => svg.getBoundingClientRect().width / b.el.getBoundingClientRect().width * b.px / Wd };
})();

/* ======================= Band gauge (S02 b5, S04 b3, S10 b3, S13 b1) ======================= */
const GAUGE = (() => {
  const b = BOARDS.board({ px: 760, w: 28, cls: 'panel', a: 0, html: `<h2 id="gt"></h2><div id="grows"></div><div id="gnote" class="banner t"></div>` });
  const rowsEl = b.q('#grows');
  const rows = [];
  /* a row: its label, then the bar and its value. A bar keeps its length whatever the value's (its length is the point): a value
     too long for the line goes under its bar. The label column is as wide as the longest label, at most 250 px */
  function ensure(n) { while (rows.length < n) { const r = D('div', 'row', rowsEl); r.style.alignItems = 'flex-start'; const lab = D('div', 'small', r); lab.style.flex = 'none';
    const cell = D('div', '', r); cell.style.cssText = 'flex:1;min-width:0;display:flex;flex-wrap:wrap;align-items:center;column-gap:12px';
    const bar = D('div', 'bar', cell), val = D('div', 'small num', cell); val.style.whiteSpace = 'nowrap'; rows.push({ r, lab, bar, val, w: 0 }); } b.measure(); }
  /* cfg.rows: [{lab, w (0..1 of 380px), cls ('nnlo'|...), style ('hatch'|'solid'|'tick'), val, ghost}]
     ghost: the bar's full width stays drawn as a dashed outline and width() collapses the fill inside it */
  function set(cfg) {
    b.q('#gt').innerHTML = cfg.title || ''; b.q('#gt').className = cfg.tcls || ''; b.q('#gnote').innerHTML = cfg.note || '';
    ensure(cfg.rows.length);
    rows.forEach((r, i) => { const c = cfg.rows[i]; if (!c) { r.r.style.display = 'none'; return; } r.r.style.display = 'flex';
      r.lab.innerHTML = c.lab; r.val.innerHTML = c.val || '';
      const fill = c.style === 'solid' ? 'solid' : c.style === 'tick' ? '' : 'hatch';
      r.bar.className = 'bar ' + (c.cls || '') + ' ' + (c.ghost ? '' : fill);
      if (c.style === 'tick') { r.bar.style.cssText = 'flex:none;margin:4px 0;width:3px;height:22px;background:var(--ink);border-radius:1px'; }
      else r.bar.style.cssText = `flex:none;margin:7px 0;width:${Math.max(0, Math.round(380 * c.w))}px;height:16px` + (c.ghost ? ';border:1.5px dashed;box-sizing:border-box' : '');
      r.bar.innerHTML = c.ghost ? `<div class="bar ${fill}" style="height:100%;width:100%"></div>` : ''; r.full = c.ghost ? c.w : 0;
      r.r.style.opacity = c.a ?? 1; });
    const live = rows.filter((r, i) => cfg.rows[i]); live.forEach(r => r.lab.style.width = 'max-content');
    const lw = Math.min(250, Math.max(...live.map(r => r.lab.offsetWidth)) + 1); live.forEach(r => r.lab.style.width = lw + 'px');
    b.measure();
  }
  const W_ = (i, w) => { const r = rows[i]; if (!r) return; if (r.full) r.bar.firstChild.style.width = (100 * w / r.full) + '%'; else r.bar.style.width = Math.max(0, Math.round(380 * w)) + 'px'; };
  return { b, set, width: W_, rows };
})();

/* ======================= Number line (S09 b2, S09 b3, S10 b2, S13 b3) ======================= */
const NLINE = (() => {
  const Wn = 720, Hn = 150, AX = 92;                          /* axis y; up-side labels at 40/56, gap at 18/68, down-side at 128/145 */
  const b = BOARDS.board({ px: Wn, w: 28, cls: 'panel', a: 0, html: `<h2 id="nt" class="nt"></h2>` });
  const svg = S('svg', { viewBox: `0 0 ${Wn} ${Hn}`, width: Wn, height: Hn }, b.el);
  const note = D('div', 'small t', b.el, '');
  let key = null, parts = [];
  const xOf = (lo, hi) => v => 40 + (v - lo) / (hi - lo) * (Wn - 80);
  function axis(cfg, lo, hi) { svg.innerHTML = ''; parts = []; svg.dataset.lo = lo; svg.dataset.hi = hi;
    S('line', { x1: 30, y1: AX, x2: Wn - 30, y2: AX, class: 'ax' }, svg);
    const tk = S('text', { x: Wn - 28, y: AX + 22, 'text-anchor': 'end', class: 'mt' }, svg); tk.textContent = cfg.unit || ''; }
  /* one item at window X: its band, its marker, its words (at opacity lab) */
  function item(it, X, g, lab = 1) {
    const col = it.cls === 'lat' ? 'var(--lat)' : it.cls === 'nnlo' ? 'var(--nnlo)' : it.cls === 'nlo' ? 'var(--nlo)' : 'var(--ink)';
    if (it.dev) { const x0 = X(it.v - it.dev), x1 = X(it.v + it.dev);
      if (it.style === 'hatch') { S('rect', { x: x0, y: AX - 14, width: x1 - x0, height: 28, fill: 'none', stroke: col, 'stroke-width': 1.5 }, g); }
      else { S('rect', { x: x0, y: AX - 14, width: Math.max(2, x1 - x0), height: 28, fill: col, 'fill-opacity': 0.45 }, g); }
      const dv = S('text', { x: (x0 + x1) / 2, y: it.side === 'up' ? 56 : 145, 'text-anchor': 'middle', fill: col, 'font-size': 14, opacity: lab }, g); dv.textContent = it.devLab || 'max deviation'; }
    if (it.style === 'square') S('rect', { x: X(it.v) - 6, y: AX - 6, width: 12, height: 12, fill: col }, g);
    else if (it.style === 'tickmark') S('line', { x1: X(it.v), y1: AX - 16, x2: X(it.v), y2: AX + 16, stroke: col, 'stroke-width': 2, 'stroke-dasharray': '4 3' }, g);
    else if (it.style === 'hatchpt') { S('circle', { cx: X(it.v), cy: AX, r: 7, fill: 'none', stroke: col, 'stroke-width': 1.6 }, g); S('line', { x1: X(it.v) - 5, y1: AX + 5, x2: X(it.v) + 5, y2: AX - 5, stroke: col, 'stroke-width': 1.4 }, g); }
    else S('circle', { cx: X(it.v), cy: AX, r: 7, fill: col }, g);
    const t = S('text', { x: X(it.v), y: it.side === 'up' ? 40 : 128, 'text-anchor': it.anchor || 'middle', fill: col, opacity: lab }, g); t.innerHTML = it.lab;
  }
  function gapOf(cfg, X, g) { const xa = X(cfg.gap.a), xb = X(cfg.gap.b);
    S('line', { x1: xa, y1: 68, x2: xb, y2: 68, stroke: 'var(--ink2)', 'stroke-width': 1.3 }, g);
    S('path', { d: `M${xa} 68 l7 -4 v8 z M${xb} 68 l-7 -4 v8 z`, fill: 'var(--ink2)' }, g);
    const t = S('text', { x: (xa + xb) / 2, y: 18, 'text-anchor': 'middle', fill: 'var(--ink2)' }, g); t.innerHTML = cfg.gap.lab; }
  /* cfg: {title, lo, hi, unit, items:[{v, dev (max deviation, same unit), lab, cls, style:'solid'|'hatch'|'point'|'square', side:'up'|'down', sub}], gap:{a,b,lab}, note} */
  function config(cfg) {
    if (key === cfg.key) return;
    key = cfg.key; b.q('#nt').innerHTML = cfg.title || ''; note.innerHTML = cfg.note || '';
    axis(cfg, cfg.lo, cfg.hi); const X = xOf(cfg.lo, cfg.hi);
    for (const it of cfg.items) { const g = S('g', { opacity: 0 }, svg); parts.push(g); item(it, X, g); }
    if (cfg.gap) { const g = S('g', { opacity: 0 }, svg); parts.push(g); gapOf(cfg, X, g); }
    b.measure();
  }
  function reveal(u, per = 0.22) { rv(b.q('#nt'), seg(u, 0, .2)); parts.forEach((g, i) => g.setAttribute('opacity', seg(u, .15 + i * per, .35 + i * per))); rv(note, seg(u, .8, 1)); }
  /* one ruler: from A's window to B's by a smooth zoom (the camera's own path, in one dimension). The window opens until it holds
     both, travels, and closes on B. Ticks show the motion and carry no number; A's items ride the axis out; B's arrive with reveal().
     t <= 0 is A as revealed, t >= 1 is B not yet revealed. */
  function zoom(A, B, t) {
    if (t <= 0) { config(A); reveal(1); return; } if (t >= 1) { config(B); return; }
    key = null; const cA = (A.lo + A.hi) / 2, cB = (B.lo + B.hi) / 2, [f, w] = W.vanWijk(A.hi - A.lo, B.hi - B.lo, Math.abs(cB - cA), t), c = cA + f * (cB - cA), lo = Math.max(0, c - w / 2), hi = lo + w;   /* a mass axis does not go below zero */
    const fade = 1 - seg(t, 0, .25); b.q('#nt').innerHTML = A.title || ''; rv(b.q('#nt'), fade); note.innerHTML = A.note || ''; rv(note, fade);
    axis(A, lo, hi); const X = xOf(lo, hi), e = Math.pow(10, Math.floor(Math.log10(w / 7))), m = w / 7 / e, step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * e;
    for (let v = Math.max(0, Math.ceil(lo / step) * step); v <= hi; v += step) S('line', { x1: X(v), y1: AX - 5, x2: X(v), y2: AX + 5, class: 'ax tk', opacity: 0.8 * Math.sin(Math.PI * t) }, svg);
    for (const it of A.items) { const x = X(it.v); if (x < 30 || x > Wn - 30) continue; item(it, X, S('g', { opacity: 1 - seg(t, .15, .6) }, svg), fade); }
    if (A.gap) gapOf(A, X, S('g', { opacity: fade }, svg));
  }
  return { b, config, reveal, zoom };
})();

/* ======================= Walker cloud (S06 out → S07) ======================= */
const WALK = (() => {
  const N = 30000;                                           /* W8: ≤ 500k; fewer, larger points so the walkers read as a sample, not a haze */
  const pos = new Float32Array(N * 3), gs = new Float32Array(N * 3), seed = new Float32Array(N);
  const r = rng(11); const a0 = 4.6;                          /* |ψ|² ∝ e^{-2r/a₀}: r ~ Gamma(3, a₀/2) */
  for (let i = 0; i < N; i++) {
    const rr = -(a0 / 2) * Math.log(r() * r() * r());
    const z = 2 * r() - 1, ph = 6.2832 * r(), s = Math.sqrt(1 - z * z);
    pos[3 * i] = rr * s * Math.cos(ph); pos[3 * i + 1] = rr * s * Math.sin(ph); pos[3 * i + 2] = rr * z;
    gs[3 * i] = gauss(r); gs[3 * i + 1] = gauss(r); gs[3 * i + 2] = gauss(r); seed[i] = r();
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('aG', new THREE.BufferAttribute(gs, 3)); g.setAttribute('aS', new THREE.BufferAttribute(seed, 1));
  const m = new THREE.ShaderMaterial({
    uniforms: { uT: { value: 0 }, uSpread: { value: 1 }, uTau: { value: 0 }, uOp: { value: 0 }, uWeight: { value: 0 }, uJit: { value: 1 }, uSize: { value: 2.4 }, uA0: { value: a0 }, uRelax: { value: 0 }, uCol: { value: new THREE.Color(W.QCOL.ar) } },   /* the samples are positions of the antiquark: they carry its anticolour, and with the quark at the centre the picture stays colour-neutral */
    transparent: true, depthWrite: false, blending: TBLEND(),
    vertexShader: `attribute vec3 aG; attribute float aS; uniform float uT,uSpread,uTau,uWeight,uJit,uSize,uA0,uRelax; varying float vW;
      float h(float x){ return fract(sin(x*127.1+aS*311.7)*43758.5453); }
      void main(){
        vec3 p = position*uSpread;
        /* GFMC: Gaussian steps in imaginary time (kinetic term); the step index advances with τ */
        float st = floor(uTau*24.0);
        vec3 stepv = aG*0.55*sqrt(min(uTau,1.0)) + vec3(h(st)-0.5, h(st+7.0)-0.5, h(st+13.0)-0.5)*0.35*step(0.5,uTau*1000.0);
        p += stepv;
        /* ambient jitter between clicks: never reveals content */
        p += uJit*0.06*vec3(sin(uT*2.1+aS*40.0), cos(uT*1.7+aS*23.0), sin(uT*1.3+aS*61.0));
        float rr = length(p);
        vW = mix(1.0, clamp(1.6*uA0/(rr+0.6), 0.25, 2.2), uWeight);   /* potential term as a multiplicative weight: brighter near the origin */
        vW *= mix(1.0, 1.0 - smoothstep(1.0*uA0, 2.2*uA0, length(position)), uRelax);   /* projection: walkers far from the ground state die out */
        vec4 mv = modelViewMatrix*vec4(p,1.0); gl_Position = projectionMatrix*mv;
        gl_PointSize = uSize*clamp(80.0/-mv.z, 0.5, 3.5);
      }`,
    fragmentShader: `uniform float uOp; uniform vec3 uCol; varying float vW; void main(){ vec2 c=gl_PointCoord-0.5; float d=dot(c,c); if(d>0.25) discard;
      float a = (1.0-4.0*d)*uOp*0.13*vW; gl_FragColor = vec4(uCol,a); }`,
  });
  const pts = new THREE.Points(g, m); pts.frustumCulled = false; pts.visible = false; W.scene.add(pts);
  const u = m.uniforms;
  function set(o) { for (const k in o) u[k].value = o[k]; pts.visible = u.uOp.value > 0.003; }
  function tick(t) { u.uT.value = t; }
  return { pts, set, tick, u, a0 };
})();

/* ======================= S06: what the cutoff does, a gauge for α_s (round 24) =======================
   The gauge reads the coupling α_s at the scale μ used for the pair's separation r (0 at the left, ∞ at the right end stop).
   b1: μ = 1/r, quark B walks out from 0.2 fm; past 0.56 fm (1/r < Λ_QCD) α_s blows up and the needle slams past the end stop.
   st.clip (in step with the plot's morph): μ = μ′, B walks out again and the needle stays near 0.48. Bond and lens versions
   were compared on the phone and dropped by the author. α_s: two loops, n_f = 3, Λ = 0.35 GeV, so the pole is at 1/r = Λ
   (the ruler's red edge) and α_s(1 GeV) = 0.48. */
const PSI = (() => {
  const L = 12, seg_ = 60;   /* out to 1.1 fm, where the hill is under 6 % of its peak: a longer base drew a flat grey sheet */ const geo = new THREE.PlaneGeometry(L, 8, seg_, 10);
  const p = geo.attributes.position; const base = new Float32Array(p.array);
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: TC('soft'), transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
  mesh.rotation.x = -Math.PI / 2; W.scene.add(mesh);
  /* r²e^{−2r/a₀} in fm (1 fm = 12 units; the hill starts 1.5 units out), peak height 2.44. a₀ = 0.3 fm, between the Bohr
     radii of cc̄ (0.49 fm) and bb̄ (0.21 fm): most of the weight is short of the red (round 20; it peaked at 0.74 fm) */
  const A0 = 0.3;
  function shape(a) { for (let i = 0; i < p.count; i++) { const u = (base[3 * i] + L / 2 + 1.5) / 12 / A0; const h = u * u * Math.exp(2 - 2 * u) * 2.44; p.array[3 * i + 2] = h * a * (1 - Math.abs(base[3 * i + 1]) / 4.5); } p.needsUpdate = true; }
  shape(1);
  const LQ = 0.35, HC = 0.1973, RC = 12 * HC / LQ, RL = 20, Y0 = -1.6;   /* 1/r = Λ_QCD at 0.56 fm; the ruler ends at 1.7 fm (B walks to 1.25 fm; longer, it ran under the plot) */
  const B0 = 9 / (4 * Math.PI), B1 = 96 / (24 * Math.PI * Math.PI);
  const alpha = mu => { const l = Math.log(mu * mu / (LQ * LQ)); return l <= 0.02 ? Infinity : 1 / (B0 * l) * (1 - B1 / (B0 * B0) * Math.log(l) / l); };
  const g = new THREE.Group(); W.crisp.add(g);
  const mk = (pts, col) => { const l = W.cline(pts, col, 0); g.add(l); return l; };
  /* the ruler in the gauge's own zones (review 1.3): teal while α_s(1/r) < 0.82, amber to 1.5, red beyond; a taller red tick at the
     pole (0.56 fm). With μ′ the amber and red parts turn grey: α_s is finite there, but V still has no confinement */
  const rAt = a => { let lo = 0.05, hi = HC / LQ; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (alpha(HC / m) < a) lo = m; else hi = m; } return 12 * lo; };
  const XT = rAt(0.82), XA = rAt(1.5);
  const zones = { teal: mk([W.V3(0, Y0, 0), W.V3(XT, Y0, 0)], TC('teal')), amber: mk([W.V3(XT, Y0, 0), W.V3(XA, Y0, 0)], TC('amber')), red: mk([W.V3(XA, Y0, 0), W.V3(RL, Y0, 0)], TC('lat')) };
  const near = [zones.teal], farR = [zones.amber, zones.red], farT = [mk([W.V3(XT, Y0, 0), W.V3(RL, Y0, 0)], TC('dim'))];
  for (let x = 0; x <= RL; x += 2) { const pts = [W.V3(x, Y0, 0), W.V3(x, Y0 - (x % 4 === 0 ? 1.0 : 0.55), 0)];
    if (x <= XT) near.push(mk(pts, TC('teal'))); else { farR.push(mk(pts, TC(x <= XA ? 'amber' : 'lat'))); farT.push(mk(pts, TC('dim'))); } }
  farR.push(mk([W.V3(RC, Y0 + 0.6, 0), W.V3(RC, Y0 - 1.6, 0)], TC('lat')));   /* the pole: 1/r = Λ */
  /* the gauge: a DOM dial; 0 at the left, ∞ at the right end stop; angle = 180° · α/(1+α) (α = 1 straight up) */
  const GA = a => 180 * a / (1 + a), gp = th => { const t = th * Math.PI / 180; return [130 - 100 * Math.cos(t), 130 - 100 * Math.sin(t)]; };
  const arc = (a0, a1, col) => { const [x0, y0] = gp(a0), [x1, y1] = gp(a1); return `<path d="M${x0.toFixed(1)} ${y0.toFixed(1)} A100 100 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}" fill="none" stroke="var(--${col})" stroke-width="12"/>`; };
  const tick_ = (a, t) => { const th = GA(a), [x, y] = gp(th), [x2, y2] = [130 + (x - 130) * 1.22, 130 + (y - 130) * 1.22]; return `<text x="${x2.toFixed(0)}" y="${(y2 + 5).toFixed(0)}" text-anchor="middle" font-size="16" fill="var(--muted)">${t}</text>`; };
  const gauge = BOARDS.board({ html: `<div class="lbl" style="text-align:center">the coupling ${H.as} <span class="tiny">(schematic: two loops, Λ ≈ 0.35 GeV)</span></div><svg viewBox="0 0 260 150" width="260" height="150">${arc(0, 81, 'teal')}${arc(81, 108, 'amber')}${arc(108, 180, 'lat')}${tick_(0, '0')}${tick_(0.5, '0.5')}${tick_(1, '1')}<text x="${gp(180)[0] + 14}" y="135" font-size="18" fill="var(--muted)">∞</text><line id="ndl" x1="130" y1="130" x2="30" y2="130" stroke="var(--ink)" stroke-width="4" stroke-linecap="round"/><circle cx="130" cy="130" r="7" fill="var(--ink)"/></svg><div class="num" id="gread" style="text-align:center;font-size:20px"></div>`, px: 290, w: 10.5, a: 0, cls: 'panel' });
  const ndl = gauge.el.querySelector('#ndl'), gread = gauge.el.querySelector('#gread');
  const redLab = BOARDS.board({ html: `<span class="lbl">${I('μ')} = 1/${I('r')}: as the quarks move apart the coupling ${H.as} grows: about 1 by <span class="num">0.3 fm</span>, and at <span class="num">0.56 fm</span> it blows up</span>`, px: 420, w: 15, a: 0, cls: 'panel' });
  const confLab = BOARDS.board({ html: `<span class="lbl">${H.mup}: the scale stops at ${H.mucut} = <span class="num">1 GeV</span>, so ${H.as} stays at or below <span class="num">0.48</span> at any distance. ${I('V')} still has no linear confining term (string tension · ${I('r')}): a systematic of the method</span>`, px: 420, w: 15, a: 0, cls: 'panel' });
  const psiLab = BOARDS.board({ html: `<span class="lbl">schematic ${I('r')}²|${I('ψ')}|², bottomonium-like (${I('a')}₀ ≈ 0.3 fm): the separations a bound state samples</span>`, px: 380, w: 14, a: 0, cls: 'panel' });
  const st = { psi: 0, red: 0, rule: 0, clip: 0 };
  let V = { r: 0, mu: 0, alpha: 0, broken: false }, ang = 0, gHtml = '';
  const view = () => { const r = Math.max(PAIR.st.sep / 12, 0.02), mu = lerp(HC / r, Math.sqrt(1 + (HC / r) ** 2), st.clip), a = alpha(mu); return { r: +r.toFixed(3), mu: +mu.toFixed(3), alpha: isFinite(a) ? +a.toFixed(3) : 'inf', broken: !isFinite(a) }; };
  function tick(t) {
    const A = PAIR.posA(), s = PAIR.st.scale;
    mesh.position.set(A.x + L / 2 * s + 1.5, A.y + Y0, A.z); mesh.scale.setScalar(s); mesh.material.opacity = 0.34 * st.psi; mesh.visible = st.psi > 0.003;
    g.position.copy(A); g.scale.setScalar(s); g.visible = st.rule > 0.003 || st.red > 0.003;
    near.forEach(l => l.material.opacity = st.rule); farR.forEach(l => l.material.opacity = st.rule * (1 - st.clip)); farT.forEach(l => l.material.opacity = st.rule * st.clip);
    V = view(); const a = V.broken ? Infinity : V.alpha, on = st.red;
    /* no live number (a counter is text written and gone, check-transient): ∞ when it blows up, one bound once μ′ holds it */
    const aTxt = V.broken ? `${H.as} → ∞` : st.clip > 0.5 ? `${H.as} ≤ 0.48 at any ${I('r')}` : '';
    /* gauge */
    ang = V.broken ? 183 : GA(a);   /* past the end stop; still, so a beat ends where its click starts (check-jumps) */
    const [nx, ny] = gp(ang); ndl.setAttribute('x2', (130 + (nx - 130) * 0.92).toFixed(1)); ndl.setAttribute('y2', (130 + (ny - 130) * 0.92).toFixed(1));
    if (aTxt !== gHtml) { gread.innerHTML = aTxt; gHtml = aTxt; } gread.style.color = V.broken ? 'var(--lat)' : 'var(--ink)'; ndl.setAttribute('stroke', V.broken ? 'var(--lat)' : 'var(--ink)');
    gauge.set(A.x + 24 * s, A.y + 4 * s, A.z);   /* a unit higher since its caption says schematic: clear of the plot */ gauge.show(on);
    const lx = A.x + 17 * s, ly = A.y + 11.8 * s;   /* over the pair, clear of the question board */
    redLab.set(lx, ly, A.z); redLab.show(on * (1 - st.clip)); confLab.set(lx, ly, A.z); confLab.show(on * st.clip);
    psiLab.set(A.x + 8 * s, A.y - 8.6, A.z + 3); psiLab.show(st.psi);   /* clear of the S06 question board; under the pair's own caption, not on it */
  }
  return { st, tick, mesh, RC, zones, alpha, view, labels: [redLab, confLab, psiLab, gauge], parts: { gauge, angle: () => ang } };
})();

/* ======================= Lattice wall (S10 b2) ======================= */
const LATT = (() => {
  const g = new THREE.Group(); g.position.set(...TP(9, 0, 0, 0)); W.crisp.add(g);   /* behind the S10 baryons */
  const x0 = -12, x1 = 50, y0 = 60, y1 = 92, z = -14;
  for (let x = x0; x <= x1; x += 4) g.add(W.cline([W.V3(x, y0, z), W.V3(x, y1, z)], TC('lat'), 0));
  for (let y = y0; y <= y1; y += 4) g.add(W.cline([W.V3(x0, y, z), W.V3(x1, y, z)], TC('lat'), 0));
  for (let x = x0; x <= x1; x += 8) for (let y = y0; y <= y1; y += 8) g.add(W.cline([W.V3(x, y, z), W.V3(x, y, z - 6)], TC('lat'), 0));
  const lab = BOARDS.board({ html: `<span class="lbl lat">lattice QCD (Refs. [26–28, 31]): not this work's method</span>`, px: 560, w: 34, at: TP(9, 10, y1 - 1.7, z), a: 0, cls: 'panel' });   /* on the wall's top edge, large enough to read from the S10 shelf frame */
  function show(a) { g.children.forEach(l => l.material.opacity = 0.32 * a); lab.show(a); }
  return { show, g };
})();

/* per-frame ambient for every persistent object */
function tickObjects(t) { SHELF.tick(t); PAIR.tick(t); PSI.tick(t); WALK.tick(t); }
