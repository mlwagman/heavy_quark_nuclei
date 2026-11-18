'use strict';
/* space.js: the three.js World (one scene laid out in depth), the bloom pipeline,
   the camera rig and the factories for lit space objects. */
const W = (() => {
  const canvas = document.getElementById('gl');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(1);                                   /* W8: pixel ratio 1 */
  renderer.setSize(1920, 1080, false);
  renderer.autoClear = false;
  renderer.toneMapping = THREE.NoToneMapping;

  const scene = new THREE.Scene();                            /* lit + bloom */
  scene.background = new THREE.Color(TC('bg'));
  const crisp = new THREE.Scene();                            /* drawn after bloom, no bloom: the Borel branch point */
  const camera = new THREE.PerspectiveCamera(38, 1920 / 1080, 0.5, 2400);

  /* bloom on the lit scene only. No MSAA on the bloom target (W8). */
  const composer = new THREE.EffectComposer(renderer);
  composer.setPixelRatio(1); composer.setSize(1920, 1080);
  composer.addPass(new THREE.RenderPass(scene, camera));
  const bloom = new THREE.UnrealBloomPass(new THREE.Vector2(1920, 1080), TN('bloom'), 0.3, 0.72);
  composer.addPass(bloom);
  composer.addPass(new THREE.OutputPass());

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xfff4e6, 1.3); key.position.set(30, 60, 80); scene.add(key);
  const fill = new THREE.DirectionalLight(0xcfd8ff, 0.5); fill.position.set(-60, -20, 40); scene.add(fill);

  /* ---------- materials ---------- */
  /* colour charge: r, g, b for quarks; ar, ag, ab (anti-red = cyan, anti-green = magenta, anti-blue = yellow) for antiquarks.
     Every hadron is colour-neutral: a meson is a colour with its anticolour, a baryon is r g b, a tetraquark is two such pairs. */
  const QCOL = { r: TC('qr'), g: TC('qg'), b: TC('qb'), ar: TC('qar'), ag: TC('qag'), ab: TC('qab') };
  const CHARGE = { 2: ['r', 'ar'], 3: ['r', 'g', 'b'], 4: ['ar', 'ag', 'r', 'g'] };   /* n = 4: the two antiquarks first (the compact pair of the tetraquark glyph); when it splits, 0 pairs with 2 and 1 with 3 */
  const GLOW = TN('q-glow'), BGC = new THREE.Color(TC('bg'));   /* quark colours by index, as the deck's glyph dots (decorative: not flavour, not colour charge) */
  const matQuark = (o = {}) => { const c = QCOL[o.c ?? 'r']; return new THREE.MeshStandardMaterial({ color: c, emissive: c, emissiveIntensity: GLOW * (o.glow ?? 1), roughness: 0.5, metalness: 0, transparent: true, opacity: o.opacity ?? 1 }); };
  const matLink = (o = {}) => new THREE.MeshBasicMaterial({ color: o.color ?? TC('link'), transparent: true, opacity: o.opacity ?? 0.85 });
  const geoQuark = new THREE.SphereGeometry(1, 28, 18);
  const geoLink = new THREE.CylinderGeometry(1, 1, 1, 8, 1, true);

  /* a quark: a sphere in its colour charge (o.c: r, g, b, ar, ag, ab) */
  function quark(r = 1, o = {}) {
    const m = new THREE.Mesh(geoQuark, matQuark(o)); m.scale.setScalar(r); m.userData.r = r; m.userData.base = new THREE.Color(QCOL[o.c ?? 'r']); m.userData.c = o.c ?? 'r'; return m;
  }
  /* a thin link between two points; .set(a,b) re-aims it */
  function link(o = {}) {
    const m = new THREE.Mesh(geoLink, matLink(o)); m.userData.th = o.th ?? 0.12;
    const up = new THREE.Vector3(0, 1, 0), dir = new THREE.Vector3();
    m.set = (a, b) => { dir.subVectors(b, a); const L = dir.length(); m.position.copy(a).addScaledVector(dir, 0.5);
      m.scale.set(m.userData.th, Math.max(1e-3, L), m.userData.th); m.quaternion.setFromUnitVectors(up, dir.normalize()); return m; };
    return m;
  }
  /* hadron glyph: n quarks on a slow tilted orbit, links between them (S02-I2 cartoon) */
  let nGlyph = 0; const _home = [0, 1, 2, 3].map(() => new THREE.Vector3());
  function glyph(n, o = {}) {
    const g = new THREE.Group(); const R = o.R ?? 2.2, r = o.r ?? 0.85;
    g.userData = { n, R, r, spin: o.spin ?? 0.35, phase: o.phase ?? (nGlyph++ * 2.4) % 6.2832,   /* the same pose on every load (it was random: a still glyph must look the same in every page) */ live: 0, tilt: o.tilt ?? 0.5, quarks: [], links: [], glowMul: 1, dim: 1 };
    for (let i = 0; i < n; i++) { const q = quark(r, { ...o, c: CHARGE[n][i] }); g.add(q); g.userData.quarks.push(q); }
    const pairs = n === 2 ? [[0, 1]] : n === 3 ? [[0, 1], [1, 2], [2, 0]] : [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]];
    for (const [a, b] of pairs) { const l = link({ opacity: 0.6, th: 0.09 * r }); l.userData.ab = [a, b]; g.add(l); g.userData.links.push(l); }
    if (o.custom) { g.userData.custom = o.custom; }
    /* a glyph moves only while a beat is about it (u.live, 0..1, set by its owner from the beat's focus): its place is its pose
       at t = 0 mixed with its orbit, so a still glyph has one pose and nothing jumps as live comes and goes */
    g.tick = (t) => {
      const u = g.userData, k = u.live;
      if (k < 1) { pose(0); if (k <= 0) return; u.quarks.forEach((q, i) => _home[i].copy(q.position)); }
      pose(t);
      if (k < 1) { u.quarks.forEach((q, i) => q.position.lerpVectors(_home[i], q.position, k)); for (const l of u.links) l.set(u.quarks[l.userData.ab[0]].position, u.quarks[l.userData.ab[1]].position); }
    };
    const pose = (t) => {
      const u = g.userData; if (u.custom) { u.custom(g, t); return; }
      for (let i = 0; i < n; i++) { const a = u.phase + t * u.spin + i * 6.2832 / n; const q = u.quarks[i];
        q.position.set(u.R * Math.cos(a), u.R * Math.sin(a) * Math.sin(u.tilt), u.R * Math.sin(a) * Math.cos(u.tilt)); }
      for (const l of u.links) l.set(u.quarks[l.userData.ab[0]].position, u.quarks[l.userData.ab[1]].position);
    };
    g.setOpacity = (a) => { g.userData.op = a; for (const q of g.userData.quarks) q.material.opacity = a; for (const l of g.userData.links) l.material.opacity = 0.6 * a; g.visible = a > 0.002; };
    /* k < 1: the glyph recedes toward the background; k > 1: a flash, read as a size pulse (emissive only helps on the dark theme) */
    g.setGlow = (k) => { const u = g.userData; u.pulse = Math.max(0, k - 1); for (const q of u.quarks) { q.material.emissiveIntensity = GLOW * k; q.material.color.lerpColors(BGC, q.userData.base, 0.4 + 0.6 * Math.min(1, k)); if (!u.custom) q.scale.setScalar(q.userData.r * (1 + 0.35 * Math.min(1, u.pulse))); } };
    g.userData.live = 1; g.tick(0); g.userData.live = 0;
    return g;
  }
  /* translucent floor / wall plane */
  function plane(w, h, color = TC('floor'), opacity = 0.12) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }));
    return m;
  }
  /* crisp poly-line (no bloom): goes into `crisp` */
  function cline(points, color = TC('wire'), opacity = 1) {
    const g = new THREE.BufferGeometry().setFromPoints(points);
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  }

  /* ---------- ambient dust: the World has depth between stations ---------- */
  let W_dust = null;
  (function dust() {
    const N = 150000, pos = new Float32Array(N * 3), r = rng(7);   /* over the whole World (round 18: x -330..330, y -290..420) */
    for (let i = 0; i < N; i++) { pos[3 * i] = (r() - .5) * 660; pos[3 * i + 1] = -290 + r() * 710; pos[3 * i + 2] = (r() - .5) * 160; }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const m = new THREE.ShaderMaterial({
      uniforms: { uT: { value: 0 }, uCol: { value: new THREE.Color(TC('dust')) }, uA: { value: TN('dust-a') } }, transparent: true, depthWrite: false, blending: TBLEND(),
      vertexShader: `uniform float uT; void main(){ vec3 p=position; p.y+=sin(uT*0.11+p.x*0.05)*0.6; p.x+=cos(uT*0.07+p.z*0.04)*0.6;
        vec4 mv=modelViewMatrix*vec4(p,1.0); gl_Position=projectionMatrix*mv; gl_PointSize=clamp(90.0/-mv.z,0.6,2.2); }`,
      fragmentShader: `uniform vec3 uCol; uniform float uA; void main(){ vec2 c=gl_PointCoord-0.5; float d=dot(c,c); if(d>0.25) discard; gl_FragColor=vec4(uCol,uA*(1.0-4.0*d)); }`,
    });
    const pts = new THREE.Points(g, m); pts.frustumCulled = false; scene.add(pts); W_dust = pts;
  })();

  /* ---------- camera rig ---------- */
  const cam = { pos: new THREE.Vector3(0, 72, 90), look: new THREE.Vector3(0, 72, 0) };
  function applyCamera() {
    /* the camera stands exactly where the beat put it. Every board is projected from it, so the slow drift it used to have
       (0.35 units, a 30 s period) moved every text on the stage between clicks, by up to 14 px on the close panels (round 26,
       tools/check-still.mjs). What still moves at rest: the dust, and the particles the beat is about */
    camera.position.copy(cam.pos);
    camera.lookAt(cam.look);
    camera.updateMatrixWorld();
  }
  /* one leg of a camera move as a smooth zoom-pan (van Wijk and Nuij, 2003): the view pulls back while it travels and closes in
     on arrival, so that what crosses the stage never moves faster than the eye follows. A short move stays a straight line,
     a long one shows both ends together half-way. RHO sets how readily it pulls back (their value, √2). */
  const RHO = Math.SQRT2, VW = 2 * Math.tan(38 * Math.PI / 360) * 16 / 9;   /* view width per unit of camera distance */
  /* from a window of width w0 to one of width w1, a distance d away, at t: [fraction of the way, width now] */
  function vanWijk(w0, w1, d, t) {
    if (d < 1e-3) return [t, w0 * Math.pow(w1 / w0, t)];       /* no travel: a plain zoom */
    const r2 = RHO * RHO, b0 = (w1 * w1 - w0 * w0 + r2 * r2 * d * d) / (2 * w0 * r2 * d), b1 = (w1 * w1 - w0 * w0 - r2 * r2 * d * d) / (2 * w1 * r2 * d);
    const r0 = Math.log(Math.sqrt(b0 * b0 + 1) - b0), r1 = Math.log(Math.sqrt(b1 * b1 + 1) - b1), s = r0 + t * (r1 - r0);
    return [w0 / (r2 * d) * (Math.cosh(r0) * Math.tanh(s) - Math.sinh(r0)), w0 * Math.cosh(r0) / Math.cosh(s)];
  }
  function zoomPan(A, B, t) {
    const dA = A.pos.distanceTo(A.look), dB = B.pos.distanceTo(B.look);
    const dx = B.look.x - A.look.x, dy = B.look.y - A.look.y, [f, w] = vanWijk(dA * VW, dB * VW, Math.hypot(dx, dy), t);
    const look = new THREE.Vector3(A.look.x + f * dx, A.look.y + f * dy, A.look.z + t * (B.look.z - A.look.z));
    const dir = A.pos.clone().sub(A.look).divideScalar(dA).lerp(B.pos.clone().sub(B.look).divideScalar(dB), t).normalize();   /* the tilt turns evenly */
    return { pos: look.clone().addScaledVector(dir, w / VW), look };
  }
  /* a keyframe path: [{pos,look}, ...]; u in [0,1] samples it. Two keyframes, or a path the camera halts on at every keyframe
     (stops): smooth zoom-pan legs. More keyframes flown through without a halt: Catmull-Rom (an arc shaped by hand). */
  function samplePath(kfs, u, stops) {
    if (kfs.length === 1) return kfs[0];
    if (kfs.length === 2 || stops) { const n = kfs.length - 1, x = Math.min(Math.max(u, 0), 1) * n, i = Math.min(n - 1, Math.floor(x)); return zoomPan(kfs[i], kfs[i + 1], x - i); }
    const cp = new THREE.CatmullRomCurve3(kfs.map(k => k.pos)), cl = new THREE.CatmullRomCurve3(kfs.map(k => k.look));
    return { pos: cp.getPoint(u), look: cl.getPoint(u) };
  }

  function render(t) {
    applyCamera();
    if (W_dust) W_dust.material.uniforms.uT.value = t;
    renderer.clear();
    composer.render();
    renderer.clearDepth();
    renderer.render(crisp, camera);
  }
  /* project a world point to stage px; returns {x,y,depth} */
  const _v = new THREE.Vector3();
  function project(p) {
    _v.copy(p).project(camera);
    const depth = camera.position.distanceTo(p);
    return { x: (_v.x + 1) * 960, y: (1 - _v.y) * 540, depth, behind: _v.z > 1 || _v.z < -1 };
  }
  const tanHalf = Math.tan(camera.fov * Math.PI / 360);
  const ppu = depth => 540 / (depth * tanHalf);              /* stage px per world unit at that depth */
  const V3 = (x, y, z) => new THREE.Vector3(x, y, z);
  const KF = (pos, look) => ({ pos: V3(...pos), look: V3(...look) });

  return { renderer, scene, crisp, camera, composer, bloom, cam, render, project, ppu, quark, link, glyph, plane, cline, matQuark, matLink, samplePath, vanWijk, V3, KF, QCOL, CHARGE };
})();
