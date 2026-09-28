'use strict';
/* ================= SCENE 11 — ratio / Detmold ================= */
(function(){
const {s,b}=scene('Scene 11 / 15 — Results IV','The ratio M<sub>QQQ</sub>/M<sub>Q<span class="ov">Q</span></sub> and the Detmold bound');
b.style.flexDirection='column';
const row=D('div','',b);row.style.cssText='display:flex;gap:22px;justify-content:center';
function fpanel(title,tok){
  const c=D('div','card rv',row);c.style.width='640px';
  D('h2','',c,title);
  const w=D('div','',c);w.style.cssText='background:#fff;border-radius:4px;padding:6px';
  const img=D('img','',w);img.src=tok;img.style.cssText='width:100%;display:block';
  return c;}
const PF=fpanel('Fixed order','assets/s11-fixed-order.png');
const PM=fpanel('MRS','assets/s11-mrs.png');
const row2=D('div','',b);row2.style.cssText='display:flex;gap:14px;margin-top:14px';
const k1=D('div','card rv',row2,'<h2>Why a ratio?</h2><div class="big">The leading renormalon partially cancels in M<sub>QQQ</sub>/M<sub>Q<span class="ov">Q</span></sub>. Dashed line: Detmold bound, M<sub>QQQ</sub> ≥ (3/2) M<sub>Q<span class="ov">Q</span></sub>.</div>');
const k2=D('div','card rv',row2,'<h2 class="tc">MRS</h2><div class="big">All three orders collapse into <span class="num tc">1.50–1.51</span> — just above the bound, essentially saturating it.</div>');
const k3=D('div','card rv',row2,'<h2 class="rc">Lattice</h2><div class="big">Red points 1.52–1.56 (Tab. I + III, lattice [27]), drifting toward us as <i class="m">m<sub>Q</sub></i> grows — the same 1/<i class="m">m<sub>Q</sub></i> gap, seen in the ratio.</div>');
const k4=D('div','card rv',row2,'<h2>Fixed-order “agreement” is bought by enormous bands</h2><div class="banner" style="font-size:21px">Better agreement with a bigger error bar is not better physics.</div>');
[k1,k2,k3,k4].forEach(k=>k.style.flex='1');
scenes.push({el:s,beats:[
 {d:14,f(p){rv(k1,p);}},
 {d:20,f(p){rv(PM,seg(p,0,.4));rv(k2,seg(p,.5,.9));}},
 {d:20,f(p){rv(k3,p);}},
 {d:19,f(p){rv(PF,seg(p,0,.4));rv(k4,seg(p,.5,.9));}},
],notes:[
 '<b>b1.</b> Now a ratio: the baryon mass over the meson mass. The leading renormalon partially cancels here. The dashed line is the Detmold bound: a triply heavy baryon cannot be lighter than three halves of the quarkonium.',
 '<b>b2.</b> With M R S, all three orders collapse into a narrow band between 1.50 and 1.51, just above the bound, essentially saturating it. The three orders sit on top of each other.',
 '<b>b3.</b> The lattice points, in red, sit at 1.52 to 1.56 and drift down toward our band as the quark mass grows. That is the same one over the heavy quark mass gap, seen again in the ratio.',
 '<b>b4.</b> A caution about fixed order. It looks consistent with the lattice, but only through its enormous scale bands. Better agreement with a bigger error bar is not better physics.',
]});
})();

