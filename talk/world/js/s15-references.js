'use strict';
/* ================= SCENE 15 — references ================= */
(function(){
const {s,b}=scene('Scene 15 / 15','Key references');
const w=D('div','rv',b,'');
w.style.cssText='display:flex;gap:48px;font:14px/1.75 var(--sans);color:var(--muted)';
D('div','',w,'<b style="color:var(--ink)">This work & predecessors</b><br>Assi, Kronfeld, Vaiva, Wagman — arXiv:2607.08817 · FERMILAB-PUB-26-0061-T (2026)<br>Assi, Wagman — PRD 108, 096004 (2023) · PRD 110, 094001 (2024)<br>Assi, Grebe, Wagman — PRD 113, 016013 (2026)<br><br><b style="color:var(--ink)">MRS</b><br>Komijani — JHEP 08 (2017) 062<br>Brambilla, Komijani, Kronfeld, Vairo (TUMQCD) — PRD 97, 034503 (2018)<br>Kronfeld — JHEP 12 (2023) 108<br>Bazavov et al. (FNAL/MILC/TUMQCD) — PRD 98, 054517 (2018)<br><br><b style="color:var(--ink)">pNRQCD / renormalons</b><br>Brambilla, Pineda, Soto, Vairo — RMP 77, 1423 (2005)<br>Beneke — Phys. Rept. 317, 1 (1999)');
D('div','',w,'<b style="color:var(--ink)">Lattice QCD comparisons</b><br>Meinel — PRD 82, 114514 (2010)<br>Brown, Detmold, Meinel, Orginos — PRD 90, 094507 (2014)<br>Mathur, Padmanath, Mondal — PRL 121, 202002 (2018)<br>Mathur, Padmanath, Chakraborty — PRL 130, 111901 (2023)<br>Gregory et al. — PRL 104, 022001 (2010)<br><br><b style="color:var(--ink)">Bounds, tetraquarks, toponium</b><br>Detmold — PRL 114, 222001 (2015)<br>Czarnecki, Leng, Voloshin — PLB 778, 233 (2018)<br>CMS — Rept. Prog. Phys. 88, 087801 (2025)<br>ATLAS — arXiv:2601.11780<br>Maltoni, Severi, Tentori, Vryonidou — arXiv:2401.08751<br>PDG — PRD 110, 030001 (2024)');
scenes.push({el:s,beats:[{d:14,f(p){rv(w,p);}}],notes:[
 '<b>b1.</b> These are the key references: the earlier M R S work, the lattice calculations we compare against, and the experimental results at the top anti-top threshold. The paper is arXiv 2607.08817.',
]});
})();

