/* world-space.js — the World's 3D layer (ticket 04): one lazy three.js renderer on #space, behind the hosted stage.
   Ticket 03 budget: pixel ratio 1; UnrealBloom on a HalfFloat target WITHOUT MSAA (a 4× MSAA HalfFloat bloom target
   read back black). No WebGL context exists until a rebuilt scene (ticket 06) calls start(); the rAF loop runs only
   while started and not held (held = a backup clip covers the scene).
     WorldSpace.start({scene, camera, tick?(t)}) · render() · stop() · hold(on) · renderer · bloom · samples · frames */
window.WorldSpace = (function () {
  'use strict';
  const cv = document.getElementById('space');
  let R = null, composer = null, view = null, raf = 0, held = false, frames = 0;
  function ensure() {
    if (R) return;
    R = new THREE.WebGLRenderer({ canvas: cv, antialias: false, powerPreference: 'high-performance' });
    R.setPixelRatio(1); R.setSize(innerWidth, innerHeight, false);
    composer = new THREE.EffectComposer(R, new THREE.WebGLRenderTarget(innerWidth, innerHeight, { type: THREE.HalfFloatType }));
    composer.setPixelRatio(1);
    addEventListener('resize', () => { R.setSize(innerWidth, innerHeight, false); composer.setSize(innerWidth, innerHeight); });
  }
  function render(t) { if (view.tick) view.tick(t); composer.render(); frames++; }
  function loop(ts) { raf = 0; if (!view || held) return; render(ts / 1000); raf = requestAnimationFrame(loop); }
  function start(v) {
    ensure();
    while (composer.passes.length) composer.removePass(composer.passes[0]);
    composer.addPass(new THREE.RenderPass(v.scene, v.camera));
    composer.addPass(new THREE.UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.8, 0.4, 0.7));
    composer.addPass(new THREE.OutputPass());
    view = v; cv.classList.add('on');
    if (!raf && !held) raf = requestAnimationFrame(loop);
  }
  function stop() { cancelAnimationFrame(raf); raf = 0; view = null; cv.classList.remove('on'); }
  function hold(on) {
    if (held === !!on) return;
    held = !!on;
    if (held) { cancelAnimationFrame(raf); raf = 0; } else if (view && !raf) raf = requestAnimationFrame(loop);
  }
  return {
    start, stop, hold, render: () => render(performance.now() / 1000),
    get renderer() { return R; }, get frames() { return frames; }, get running() { return !!raf; },
    get bloom() { return !!composer && composer.passes.some(p => p instanceof THREE.UnrealBloomPass); },
    get samples() { return composer ? composer.renderTarget1.samples : null; },
  };
})();
