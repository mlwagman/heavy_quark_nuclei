'use strict';
/* ================= SCENE 12 — tetraquarks ================= */
(function(){
const {s,b}=scene('Scene 12 / 15 — Results V','All-heavy tetraquarks QQ<span class="ov">Q</span>′<span class="ov">Q</span>′ : who binds?');
b.style.flexDirection='column';
const fc=D('div','card rv',b);fc.style.cssText+=';align-self:center;width:1050px';
D('h2','',fc,'Binding ratio R vs m<sub>Q</sub>/m<sub>Q</sub>′ — R &gt; 0 ⇔ bound below the two-quarkonium threshold · MRS at LO/NLO');
const fw=D('div','',fc);fw.style.cssText='background:#fff;border-radius:4px;padding:6px';
const fimg=D('img','',fw);fimg.src='assets/s12-figure.png';fimg.style.cssText='width:100%;display:block';
const fcap=D('div','small rv',fc,'critical <span class="rc num">m<sub>Q</sub>/m<sub>Q</sub>′ = 0.085(5)</span> (§V A) at NLO — fixed-order NLO said ≈0.125. MRS requires a <b>smaller</b> mass ratio: the binding window tightens by ~30%, with the same verdicts in both schemes.');
fcap.style.marginTop='8px';
const row2=D('div','',b);row2.style.cssText='display:flex;gap:14px;margin-top:14px';
const q=D('div','card rv',row2,'<h2>The question</h2><div class="m" style="font-size:18px">R = (Δ<i>E</i><sub>4Q</sub> − 2Δ<i>E</i><sub>Q<span class="ov">Q</span>′</sub>) / 2Δ<i>E</i><sub>Q<span class="ov">Q</span>′</sub></div><div class="big" style="margin-top:6px">binding requires a mass <b>hierarchy</b></div>');
const bin=D('div','card rv',row2,'<h2>Verdicts</h2><div style="display:flex;gap:14px"><div style="flex:1"><div class="small tc" style="font-weight:600">BOUND · ΔE ≈ 509–742 MeV</div><div id="bb"></div></div><div style="flex:1"><div class="small rc" style="font-weight:600">UNBOUND · ratio ≫ 0.085</div><div id="ub"></div></div></div>');
const bb=bin.querySelector('#bb'), ub=bin.querySelector('#ub');
const bchips=['b b <span class="ov">t</span> <span class="ov">t</span>','b c <span class="ov">t</span> <span class="ov">t</span>','c c <span class="ov">t</span> <span class="ov">t</span>'].map(t=>{const c=D('span','chip m rv',bb,t);c.style.cssText+=';border-color:var(--teal)';return c;});
const uchips=['b b <span class="ov">b</span> <span class="ov">b</span>','c c <span class="ov">b</span> <span class="ov">b</span>'].map(t=>{const c=D('span','chip m rv',ub,t);c.style.cssText+=';border-color:var(--red)';return c;});
const col=D('div','card rv',row2,'<h2>Colour structure</h2><div id="cb"></div><div class="big" style="margin-top:6px"><span class="m">3⊗<span class="ov">3</span></span> beats <span class="m">6⊗<span class="ov">6</span></span> by 50–150 MeV <span class="small">(§V B)</span> ⇒ consistent with a <b>diquark–antidiquark</b> structure</div>');
[q,bin,col].forEach(k=>k.style.flex='1');
const cb=col.querySelector('#cb');
const cr1=D('div','',cb);cr1.style.cssText='display:flex;align-items:center;gap:10px;margin:5px 0';
D('div','small m',cr1,'3⊗<span class="ov">3</span>').style.width='52px';
const cb1=D('div','',cr1);cb1.style.cssText='height:13px;border-radius:3px;background:var(--teal);width:0';
const cr2=D('div','',cb);cr2.style.cssText='display:flex;align-items:center;gap:10px;margin:5px 0';
D('div','small m',cr2,'6⊗<span class="ov">6</span>').style.width='52px';
const cb2=D('div','',cr2);cb2.style.cssText='height:13px;border-radius:3px;background:var(--muted);width:0';
scenes.push({el:s,beats:[
 {d:18,f(p){rv(q,p);}},
 {d:22,f(p){rv(fc,seg(p,0,.4));rv(fcap,seg(p,.6,.95));}},
 {d:26,f(p){rv(bin,seg(p,0,.2));bchips.forEach((c,i)=>rv(c,seg(p,.2+i*.15,.4+i*.15)));
   uchips.forEach((c,i)=>rv(c,seg(p,.65+i*.15,.85+i*.15)));}},
 {d:18,f(p){rv(col,seg(p,0,.25));cb1.style.width=(220*eo(seg(p,.25,.55)))+'px';cb2.style.width=(150*eo(seg(p,.45,.75)))+'px';}},
],notes:[
 '<b>b1.</b> Finally, the all-heavy tetraquarks. Which combinations bind? The ratio R compares the tetraquark binding energy with twice that of the mixed quarkonium: positive means bound below threshold.',
 '<b>b2.</b> Binding needs a mass hierarchy. The critical ratio of the lighter to the heavier quark mass is 0.085, with an uncertainty of 5 in the last digit, at next-to-leading order, against about 0.125 in fixed order. M R S requires a smaller ratio: the window tightens by about 30 percent, with the same verdicts. These results stop at next-to-leading order; the four-quark potentials are not known beyond that.',
 '<b>b3.</b> The verdicts. Bound: b b anti-t anti-t, b c anti-t anti-t, and c c anti-t anti-t, from 509 to 742 MeV. Here the M R S scale range, several hundred MeV, is wider than in fixed order; what stays small is the shift from one order to the next. No binding found, within our variational space, for b b anti-b anti-b and c c anti-b anti-b. Their mass ratios, 1 and 0.324, sit far above the threshold.',
 '<b>b4.</b> In colour, triplet-antitriplet beats sextet-antisextet by 50 to 150 MeV, consistent with a diquark-antidiquark structure, although the colour-angle convergence deserves further study.',
]});
})();

