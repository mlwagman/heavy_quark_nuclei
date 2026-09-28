'use strict';
/* ================= SCENE 13 — what did MRS buy ================= */
(function(){
const {s,b}=scene('Scene 13 / 15 — Synthesis','What did renormalon subtraction buy us?');
b.style.flexDirection='column';
const tb=D('div','card',b,'<table class="tbl" style="width:100%"><tr><th>diagnostic</th><th>fixed order</th><th></th><th>MRS</th></tr></table><div class="small" style="margin-top:6px">λ-bands are a scheme-stability diagnostic, not a truncation-error estimate — order-by-order shifts gauge the truncation.</div>');
const T=tb.querySelector('table');
const rows=[['scale band, Ω<sub>bbb</sub> at NNLO','~700 MeV','~10 MeV'],
            ['order-by-order shifts (baryons, |NNLO−NLO|)','33–55 MeV','4–18 MeV'],
            ['tuned m<sub>c</sub>, m<sub>b</sub> drift','4–11%','&lt; 1%'],
            ['LO / NLO / NNLO bands','LO apart','overlap (partial)']].map(r=>{
  /* λ-band disclaimer added after the table below */
  const tr=D('tr','',T);
  D('td','st',tr,r[0]).style.fontStyle='normal';
  const a=D('td','rv rc num',tr,r[1]);
  const ar=D('td','rv',tr,'→');ar.style.color='var(--muted)';
  const c=D('td','rv tc num',tr,r[2]);
  return {a,ar,c};});
const row2=D('div','',b);row2.style.cssText='display:flex;gap:16px;margin-top:16px';
const h1=D('div','card rv',row2,'<h2>Hallmarks of improved convergence</h2><div class="big">For charm and bottom hadrons, every diagnostic moves the same way. Taken together they say the leading renormalon <i>was</i> the obstruction — not one problem among several.</div>');
h1.style.flex='1';
const h2=D('div','card rv',row2,'<h2 class="tc">The payoff</h2><div class="big">Order-by-order shifts of <span class="num">4–18 MeV</span> make the remaining <span class="num">125–175 MeV</span> gap <i>interpretable</i>: consistent with missing 1/<i class="m">m<sub>Q</sub></i> physics, <b>not perturbative noise</b>.</div><svg viewBox="0 0 560 90" width="560" height="90" id="pv"></svg>');
h2.style.flex='1';
const pv=h2.querySelector('#pv');
E('line',{x1:20,y1:48,x2:540,y2:48,stroke:'var(--muted)','stroke-width':1.5},pv);
const gh=E('rect',{x:130,y:14,width:0,height:68,rx:4,fill:'var(--red)','fill-opacity':.18},pv);
const gpt=E('circle',{cx:280,cy:48,r:0,fill:'var(--teal)'},pv);
const gband=E('rect',{x:274,y:38,width:12,height:20,rx:2,fill:'var(--teal)','fill-opacity':.4,opacity:0},pv);
const lpt=E('rect',{x:414,y:42,width:12,height:12,fill:'var(--red)',opacity:0},pv);
E('text',{x:280,y:30,'text-anchor':'middle','font-size':'15',class:'axl'},pv).textContent='ours: shifts 4–18 MeV';
E('text',{x:420,y:30,'text-anchor':'middle','font-size':'15',class:'axl'},pv).textContent='lattice';
const glab=E('text',{x:350,y:82,'text-anchor':'middle','font-size':'15',fill:'var(--rose)',opacity:0},pv);
glab.textContent='← 125–175 MeV, now meaningful →';
scenes.push({el:s,beats:[
 {d:26,f(p){rows.forEach((r,i)=>{rv(r.a,seg(p,i*.16,.15+i*.16));rv(r.ar,seg(p,.08+i*.16,.22+i*.16));rv(r.c,seg(p,.14+i*.16,.3+i*.16));});}},
 {d:20,f(p){rv(h1,p);}},
 {d:22,f(p){rv(h2,seg(p,0,.25));const u=eo(seg(p,.2,.5));gh.setAttribute('width',300*(1-u)+0*u);gh.setAttribute('x',130+150*u);
   gpt.setAttribute('r',7*back(seg(p,.45,.65))>0?7*back(seg(p,.45,.65)):0);op(gband,seg(p,.5,.7));op(lpt,seg(p,.6,.75));op(glab,seg(p,.75,.95));}},
],notes:[
 '<b>b1.</b> So what did renormalon subtraction buy us? Four diagnostics that all move the same way. The scale band for the triply bottom Omega: from about 700 MeV to about 10. The order-by-order shifts of the baryon masses: from 33 to 55 MeV down to 4 to 18. The drift of the tuned masses: from 4 to 11 percent to below 1. And bands where leading order sat apart now overlap.',
 '<b>b2.</b> Together these are hallmarks of improved convergence. The series is still asymptotic, only the leading renormalon is gone, but every diagnostic says it was the obstruction, not one problem among several. And the lambda bands diagnose scheme stability; they are not a truncation-error estimate.',
 '<b>b3.</b> The payoff: with order-by-order shifts of only 4 to 18 MeV, the remaining gap of 125 to 175 MeV to the lattice becomes interpretable. It is consistent with missing one over m Q physics, not perturbative noise.',
]});
})();

