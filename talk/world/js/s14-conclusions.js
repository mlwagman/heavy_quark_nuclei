'use strict';
/* ================= SCENE 14 — conclusions ================= */
(function(){
const {s,b}=scene('Scene 14 / 15 — Conclusions','Conclusions and outlook');
b.style.flexDirection='column';
const c1=D('div','card rv',b,'<div class="big"><b>First application of MRS inside a bound-state pNRQCD calculation</b> — NNLO static potentials + VMC / GFMC.</div>');
const c2=D('div','card rv',b,'<div class="big">Scale sensitivity of charm and bottom hadrons down by <span class="tc num">up to two orders of magnitude</span>; results stable order by order.</div>');
const c3=D('div','card rv',b,'<h2>Predictions — 2 tuned masses + <span style="text-transform:none">m<sub>t</sub></span> input</h2><div id="fam"></div>');
const fam=c3.querySelector('#fam');
const chips=['B<sub>c</sub>','η<sub>t</sub> (toponium)','Ω<sub>ccc</sub> … Ω<sub>bbb</sub>','top baryons → Ω<sub>ttt</sub>','bound b b <span class="ov">t</span> <span class="ov">t</span> · b c <span class="ov">t</span> <span class="ov">t</span> · c c <span class="ov">t</span> <span class="ov">t</span>'].map(t=>{
  const c=D('span','chip m rv',fam,t);c.style.borderColor='var(--teal)';return c;});
const c4=D('div','card rv',b,'<h2>Next</h2><div class="big">𝒪(1/<i class="m">m<sub>Q</sub></i>) + spin potentials — where the gap points · MRS derivation for non-singlet and 3-body channels · NNLO tetraquarks · excited states · the t <span class="ov">t</span> threshold vs new LHC data · dark-sector bound states.</div>');
const ty=D('div','rv',b,'<div style="display:flex;align-items:center;justify-content:center;gap:40px;margin-top:14px"><div style="font:italic 34px/1.3 var(--serif)">Thank you <span class="small" style="font-style:normal">— happy to take questions</span></div><div style="text-align:center"><img src="assets/qr.png" alt="arXiv" style="width:150px;display:block;border-radius:6px"><div class="small num" style="margin-top:5px">arXiv 2607.08817</div></div></div>');
scenes.push({el:s,beats:[
 {d:14,f(p){rv(c1,p);}},
 {d:12,f(p){rv(c2,p);}},
 {d:18,f(p){rv(c3,seg(p,0,.2));chips.forEach((c,i)=>rv(c,seg(p,.15+i*.15,.35+i*.15)));}},
 {d:20,f(p){rv(c4,seg(p,0,.4));rv(ty,seg(p,.6,.95));}},
],notes:[
 '<b>b1.</b> To conclude. This is the first use of minimal renormalon subtraction inside a bound-state potential N R Q C D calculation, with next-to-next-to-leading order static potentials and quantum Monte Carlo.',
 '<b>b2.</b> The scale sensitivity comes down by up to two orders of magnitude for charm and bottom states, and the results are stable order by order in every channel we looked at.',
 '<b>b3.</b> The result is a family of predictions from two tuned masses and the top mass. The B c meson and toponium. The triply heavy baryons from triply charm to triply bottom, and baryons with top quarks up to the triply top Omega. And the bound all-heavy tetraquarks.',
 '<b>b4.</b> What comes next. The one over m Q and spin-dependent potentials, where the gap points. An M R S derivation for the non-singlet and three-body channels. Tetraquarks at the next order, and excited states. The top anti-top threshold against the new LHC data. Thank you, and I am happy to take questions.',
]});
})();

