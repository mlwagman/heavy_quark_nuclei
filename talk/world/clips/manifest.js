/* Backup clips (ticket 04 contract; ticket 06 records the first). One entry per scene that has a clip:
     window.WORLD_CLIPS = { "<scene number 1..15>": { src: "clips/<file>.webm", marks: [t0, t1, …, tn] } }
   n = that scene's beat count; tk = clip time (s) at which beat k is settled; t0 = the scene opening.
   A click plays the clip on to the next mark and holds; back seeks and holds. An entry whose marks do not fit the
   scene (length ≠ n+1, not rising) is ignored, and forcing that scene does nothing. Clip files are gitignored
   (regenerable); force one with world.html?backup=4,5 or the V key. */
window.WORLD_CLIPS = { "4": { "src": "clips/s04-renormalon.webm", "marks": [ 0, 5.2, 11.4, 15.9333, 22.8 ] }, "5": { "src": "clips/s05-mrs.webm", "marks": [ 0, 4.2, 14.4, 20.6, 25.8, 34 ] } };
