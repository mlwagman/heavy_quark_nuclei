'use strict';
/* ================= SCENE 8 — quarkonium tuning ================= */
(function(){
const {s,b}=scene('Scene 8 / 15 — Results I','Quarkonium: tuning m<sub>c</sub>, m<sub>b</sub>');
b.style.flexDirection='column';
const row=D('div','',b);row.style.cssText='display:flex;gap:22px;justify-content:center';
function fpanel(title,tok,cap,col){
  const c=D('div','card rv',row);c.style.width='660px';
  D('h2','',c,title);
  const w=D('div','',c);w.style.cssText='background:#fff;border-radius:4px;padding:6px';
  const img=D('img','',w);img.src=tok;img.style.cssText='width:100%;display:block';
  const cp=D('div','small rv',c,cap);cp.style.cssText='margin-top:8px;color:'+col;
  return {c,cp};}
const P1=fpanel('Fixed order','assets/s08-fixed-order.png','broad bands — LO does not even overlap the higher orders','var(--red)');
const P2=fpanel('MRS','assets/s08-mrs.png','2–3× narrower (Fig. 1 caption) — orders sit on top of each other','var(--teal)');
const row2=D('div','',b);row2.style.cssText='display:flex;gap:16px;margin-top:14px';
const dft=D('div','card rv',row2,'<h2>Tuned to spin-averaged 1S — mass drift by order</h2><div id="dr"></div>');
const dr=dft.querySelector('#dr');
const d1=D('div','',dr);d1.style.cssText='display:flex;align-items:center;gap:12px;margin:6px 0';
D('div','small',d1,'fixed order').style.width='90px';
const db1=D('div','',d1);db1.style.cssText='height:15px;border-radius:3px;background:var(--red);width:0';
const dv1=D('div','small num rc',d1,'');
const d2=D('div','',dr);d2.style.cssText='display:flex;align-items:center;gap:12px;margin:6px 0';
D('div','small',d2,'MRS').style.width='90px';
const db2=D('div','',d2);db2.style.cssText='height:15px;border-radius:3px;background:var(--teal);width:0';
const dv2=D('div','small num tc',d2,'');
const res=D('div','card rv',row2,'<h2>Tuned masses</h2><div class="m num" style="font-size:23px"><i>m<sub>c</sub></i> = <span id="mc">—</span> GeV &nbsp;·&nbsp; <i>m<sub>b</sub></i> = <span id="mb">—</span> GeV</div><div class="banner rv" id="pred" style="font-size:23px;margin-top:8px">Everything after this slide is a prediction.<br><span class="small" style="font-style:normal">two tuned numbers (m<sub>c</sub>, m<sub>b</sub>) + one input (m<sub>t</sub>)</span></div>');
const mc=res.querySelector('#mc'), mb=res.querySelector('#mb'), pred=res.querySelector('#pred');
scenes.push({el:s,beats:[
 {d:22,f(p){rv(P1.c,seg(p,0,.35));rv(P1.cp,seg(p,.75,1));}},
 {d:18,f(p){rv(P2.c,seg(p,0,.35));rv(P2.cp,seg(p,.75,1));}},
 {d:20,f(p){rv(dft,seg(p,0,.2));const u1=eo(seg(p,.2,.6));db1.style.width=(430*u1)+'px';dv1.textContent=u1>0.45?'4–'+Math.max(5,Math.round(11*u1))+'%':'';
   const u2=eo(seg(p,.55,.85));db2.style.width=(40*u2)+'px';dv2.textContent=u2>0.5?'< 1%':'';}},
 {d:16,f(p){rv(res,seg(p,0,.25));mc.textContent=p>0.2?(1.555*eo(seg(p,.2,.6))).toFixed(3):'—';mb.textContent=p>0.25?(4.802*eo(seg(p,.25,.65))).toFixed(3):'—';
   if(p>0.62){mc.textContent='1.555';mb.textContent='4.802';}rv(pred,seg(p,.7,1));}},
],notes:[
 '<b>b1.</b> Now the results: the binding energy against the quark mass in fixed-order perturbation theory, at leading, next-to-leading, and next-to-next-to-leading order. The shaded bands are the scale variation, and two things are wrong with them. They are broad, and leading order does not even overlap the higher orders. That is the fingerprint of the renormalon.',
 '<b>b2.</b> Now the same plot with minimal renormalon subtraction. The bands are two to three times narrower at next-to-leading and next-to-next-to-leading order, and the orders sit on top of one another. That is what improved convergence looks like.',
 '<b>b3.</b> We tune the charm and bottom masses to the spin-averaged one S quarkonium states. In fixed order the extracted mass drifts by 4 to 11 percent as we go up in order. With M R S the drift is below 1 percent.',
 '<b>b4.</b> The tuned values are a charm mass of 1.555 GeV and a bottom mass of 4.802 GeV. These are effective masses. Everything that follows is a prediction from these two numbers, plus the top mass as an input.',
]});
})();

