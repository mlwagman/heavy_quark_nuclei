/* series-level.js — the 3D content of the Zoom loop's bottom level (ticket 06: S04, S05), drawn by WorldSpace.
   The Borel floor is a faint grid on the plane y = 0 (below the bloom threshold: it never glows); its u-axis, the
   contour, the branch point and the cut are crisp SVG in rebuild-space.js (storyboard W1, #17). The static pair is
   the one glowing object: two warm-white quarks and a thin link (catalogue S03-I1, labelled in the S04 Board).
     SeriesLevel.frame({cam, pairAlpha, floorAlpha, t}) · setBackground(css colour) · scene · camera */
window.SeriesLevel = (function () {
  'use strict';
  const RC = RebuildCore;
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.1, 200);
  scene.fog = new THREE.Fog(0x000000, 11, 20); // far grid lines melt into the background
  const g = [];
  // half-cell offset: no grid line runs along the u-axis or through the branch point (x = 4)
  for (let x = -7.5; x <= 22; x++) g.push(x, 0, -6, x, 0, 4);
  for (let z = -5.5; z <= 4; z++) g.push(-8, 0, z, 22, 0, z);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(g, 3));
  const floorMat = new THREE.LineBasicMaterial({ color: 0x3a4660, transparent: true });
  const floor = new THREE.LineSegments(geo, floorMat);
  scene.add(floor);
  const quarkMat = new THREE.MeshBasicMaterial({ color: 0xf0e2c8, transparent: true, fog: false }); // a little under white: a smaller bloom halo (animation review #3)
  const linkMat = new THREE.MeshBasicMaterial({ color: 0x8a93a8, transparent: true, fog: false });
  const q1 = new THREE.Mesh(new THREE.SphereGeometry(0.15, 32, 16), quarkMat), q2 = new THREE.Mesh(q1.geometry, quarkMat);
  const link = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.8, 8), linkMat);
  q1.position.x = -0.4; q2.position.x = 0.4; link.rotation.z = Math.PI / 2;
  const pair = new THREE.Group();
  pair.add(q1, q2, link); pair.position.set(...RC.PAIR);
  scene.add(pair);
  function frame({ cam, pairAlpha, floorAlpha, t }) {
    camera.position.set(...cam.pos); camera.fov = cam.fov; camera.aspect = innerWidth / innerHeight;
    camera.lookAt(...cam.target); camera.updateProjectionMatrix();
    quarkMat.opacity = pairAlpha; linkMat.opacity = 0.8 * pairAlpha; pair.visible = pairAlpha > 0;
    floorMat.opacity = 0.6 * floorAlpha; floor.visible = floorAlpha > 0;
    pair.rotation.y = 0.35 * Math.sin(0.4 * t); // ambient only (W2): a slow sway, never reveals anything
  }
  function setBackground(css) { scene.background = new THREE.Color(css); scene.fog.color.copy(scene.background); }
  return { scene, camera, frame, setBackground };
})();
