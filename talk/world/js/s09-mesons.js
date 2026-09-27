'use strict';
/* ================= SCENE 9 — mesons ================= */
(function(){
const {s,b}=scene('Scene 9 / 15 — Results II','Toponium and mixed-flavour mesons');
const L=D('div','col',b);L.style.width='760px';
const tb=D('div','card',L,'<table class="tbl" style="width:100%"><tr><th>state</th><th>NNLO MRS [GeV]</th><th>compare</th></tr>'+
 '<tr class="rv" id="r1"><td class="st">η<sub>t</sub></td><td>342.47(4)(77)</td><td>≈ 343 (NRQCD threshold)</td></tr>'+
 '<tr class="rv" id="r2"><td class="st"><span class="ov">M</span><sub>b<span class="ov">c</span></sub></td><td>6.29121(4)(731)</td><td>6.316(7) (exp. + lattice)</td></tr>'+
 '<tr class="rv" id="r3"><td class="st"><span class="ov">M</span><sub>t<span class="ov">c</span></sub></td><td>173.98462(4)(883)</td><td>—</td></tr>'+
 '<tr class="rv" id="r4"><td class="st"><span class="ov">M</span><sub>t<span class="ov">b</span></sub></td><td>177.00684(39)(2372)</td><td>—</td></tr></table>');
const rows=['r1','r2','r3','r4'].map(id=>tb.querySelector('#'+id));
const leg=D('div','card rv',L,'<h2>Reading the numbers</h2><div class="big">first parenthesis: <span class="tc">GFMC statistics</span> · second: <span class="rc">scale variation λ ∈ {½, 1, 2}</span> &nbsp;·&nbsp; <i class="m">m<sub>t</sub></i> = 172.52 GeV input</div>');
const R=D('div','col',b);R.style.flex='1';
const c1=D('div','card rv',R,'<h2 class="rc">Toponium</h2><div class="big">LHC (CMS 2025 + ATLAS 2026) sees a &gt;5σ excess at the t <span class="ov">t</span> threshold with <b>pseudoscalar-like</b> quantum numbers. MRS potentials would sharpen exactly the NRQCD Green’s functions those analyses rely on.</div>');
const c2=D('div','card rv',R,'<h2 class="tc">B<sub>c</sub> — parameter-free</h2><div class="big">Once <i class="m">m<sub>c</sub></i>, <i class="m">m<sub>b</sub></i> are fixed there is nothing left to adjust.</div><svg viewBox="0 0 560 96" width="560" height="96" id="bc"></svg>');
const bc=c2.querySelector('#bc');
E('line',{x1:30,y1:52,x2:530,y2:52,stroke:'var(--muted)','stroke-width':1.5},bc);
const bX=v=>30+(v-6.24)*500/(6.36-6.24);
const bband=E('rect',{x:bX(6.29121-0.00731),y:40,width:0,height:24,rx:4,fill:'var(--teal)','fill-opacity':.3},bc);
const bpt=E('circle',{cx:bX(6.29121),cy:52,r:0,fill:'var(--teal)'},bc);
const ept=E('circle',{cx:bX(6.316),cy:52,r:0,fill:'var(--rose)'},bc);
E('text',{x:bX(6.29121),y:26,'text-anchor':'middle','font-size':'16',fill:'var(--teal)'},bc).textContent='ours 6.2912';
E('text',{x:bX(6.316),y:86,'text-anchor':'middle','font-size':'16',fill:'var(--rose)'},bc).textContent='exp+lat 6.316';
const c3=D('div','card rv',R,'<h2>No comparison yet</h2><div class="big">t <span class="ov">c</span> and t <span class="ov">b</span> — benchmarks, not observables.</div>');
scenes.push({el:s,beats:[
 {d:20,f(p){rv(rows[0],seg(p,0,.3));rv(c1,seg(p,.35,.8));}},
 {d:18,f(p){rv(rows[1],seg(p,0,.25));rv(c2,seg(p,.2,.5));
   bband.setAttribute('width',(bX(6.29121+0.00731)-bX(6.29121-0.00731))*eo(seg(p,.4,.7)));
   bpt.setAttribute('r',6*back(seg(p,.5,.75))>0?6*back(seg(p,.5,.75)):0);
   ept.setAttribute('r',6*back(seg(p,.65,.9))>0?6*back(seg(p,.65,.9)):0);}},
 {d:13,f(p){rv(rows[2],seg(p,0,.35));rv(rows[3],seg(p,.2,.55));rv(c3,seg(p,.5,.9));}},
 {d:14,f(p){rv(leg,p);}},
],notes:[
 '<b>b1.</b> Toponium first. The pseudoscalar comes out at 342.47 GeV, with an uncertainty of 77 in the last digits, against roughly 343 from nonrelativistic threshold treatments. This is the zero-width limit, ignoring the decay of the top. That is directly relevant to the LHC excess.',
 '<b>b2.</b> The B c meson is parameter-free: once the charm and bottom masses are fixed there is nothing left to adjust. We find 6.29121 GeV, with an uncertainty of 731 in the last digits, about 25 MeV below the combined experimental and lattice value of 6.316, comparable to, though outside, our scale-variation range.',
 '<b>b3.</b> The mixed top mesons, top-charm and top-bottom, are benchmarks rather than observables: no comparison values exist, and the top decays before it can hadronize.',
 '<b>b4.</b> To read the numbers: first parenthesis, Monte Carlo statistics; second, the scale variation over lambda; and the top mass is an input at 172.52 GeV.',
]});
})();

