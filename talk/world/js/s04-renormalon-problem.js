'use strict';
/* ================= SCENE 4 — renormalon problem ================= */
(function(){
const {s,b}=scene('Scene 4 / 15 — Background II','The renormalon problem');
const L=D('div','col',b);L.style.width='620px';
const e1=D('div','card rv',L,'<h2>The series</h2><div class="m" style="display:flex;align-items:center;flex-wrap:wrap;column-gap:6px"><span><i>V<sub>R</sub></i>(<i>r</i>) = −</span><span class="frac"><span><i>C<sub>R</sub></i></span><span class="den"><i>r</i></span></span><span class="sum"><span class="sg">Σ</span><span class="lim"><i>l</i></span></span><span><i>v<sub>l</sub></i> <i>α<sub>s</sub></i><sup>l+1</sup>(<i>μ</i>)</span></div>');
const e2=D('div','card rv',L,'<h2 class="rc">Factorial growth</h2><div class="m" style="display:flex;align-items:center;flex-wrap:wrap;column-gap:6px"><span><i>v<sub>l</sub></i> ~ <i>V</i><sub>0</sub> (2<i>β</i><sub>0</sub>)<sup>l</sup></span><span class="frac"><span>Γ(<i>l</i>+1+<i>b</i>)</span><span class="den">Γ(1+<i>b</i>)</span></span></div><div class="small" style="margin-top:6px">the series is <b>asymptotic</b>, not convergent</div>');
const e3=D('div','card rv',L,'<h2>Origin — infrared</h2><div class="big">Gluons with <i class="m">k ~ Λ</i><sub>QCD</sub> contaminate the potential ⇒ irreducible ambiguity <i class="m">δV ~ Λ</i><sub>QCD</sub></div><div class="small" style="margin-top:6px">the same renormalon as the pole mass</div>');
const e4=D('div','card rv',L,'<h2 class="rc">Symptom</h2><div class="big"><i class="m">μ</i> → 2<i class="m">μ</i> moves a triply-heavy baryon mass by <span class="rc num">~700 MeV</span> at NNLO; LO / NLO / NNLO bands do not overlap</div>');
const R=D('div','col',b);R.style.flex='1';
const psv=E('svg',{viewBox:'0 0 760 330',width:760,height:330},D('div','card',R,'<h2>|term| vs order</h2>'));
axes(psv,60,290,730,20);
mlab(psv,726,314,'<i>ℓ</i> (order)',{anchor:'end'});
mlab(psv,74,26,'|<i>v<sub>ℓ</sub></i> <i>α<sub>s</sub></i><sup>ℓ+1</sup>|',{size:18});
const terms=[250,180,130,96,78,72,80,110,170,255].map((h,i)=>
  E('circle',{cx:100+64*i,cy:290-h*0.93,r:0,fill:'var(--c1)'},psv));
const floor=E('line',{x1:60,y1:290-72*0.93,x2:730,y2:290-72*0.93,stroke:'var(--red)','stroke-width':1.5,'stroke-dasharray':'7 5'},psv);dl(floor);
const flab=mlab(psv,470,290-72*0.93+26,'minimal term ⇒ <i>δV</i> ~ Λ<sub>QCD</sub>',{size:18,color:'var(--red)'});op(flab,0);
const pcap=E('text',{x:395,y:322,'text-anchor':'middle','font-size':'17','font-style':'italic','font-family':'var(--serif)',opacity:0},psv);
pcap.textContent='terms shrink, reach a minimum, blow up';
const bsv=E('svg',{viewBox:'0 0 760 210',width:760,height:210},D('div','card',R,'<h2>Bands at successive orders — fixed order</h2>'));
axes(bsv,60,180,730,16);
mlab(bsv,74,26,'Δ<i>E</i>',{size:17});
const bnds=[['LO','var(--c1)',130,26,90,220],['NLO','var(--amber)',86,30,340,230],['NNLO','var(--c3)',38,26,600,130]].map((bb,i)=>{
  const r=E('rect',{x:bb[4],y:bb[2],width:0,height:bb[3],rx:4,fill:bb[1],'fill-opacity':.5},bsv);
  const t=E('text',{x:bb[4]-8,y:bb[2]+bb[3]/2+5,'text-anchor':'end','font-size':'17',fill:bb[1],opacity:0},bsv);t.textContent=bb[0];
  return {r,t,w:bb[5]};});
const gap=E('text',{x:395,y:204,'text-anchor':'middle','font-size':'17',fill:'var(--red)',opacity:0},bsv);
gap.textContent='disjoint — no meaningful uncertainty';
scenes.push({el:s,beats:[
 {d:15,f(p){rv(e1,seg(p,0,.4));rv(e2,seg(p,.45,.9));}},
 {d:18,f(p){terms.forEach((c,i)=>c.setAttribute('r',6*back(seg(p,i*0.06,i*0.06+0.18))>0?6*back(seg(p,i*0.06,i*0.06+0.18)):0));
   ds(floor,seg(p,.6,.8));op(flab,seg(p,.75,.9));op(pcap,seg(p,.85,1));}},
 {d:13,f(p){rv(e3,p);}},
 {d:20,f(p){rv(e4,seg(p,0,.25));bnds.forEach((bb,i)=>{bb.r.setAttribute('width',bb.w*eo(seg(p,.15+i*.2,.45+i*.2)));op(bb.t,seg(p,.3+i*.2,.5+i*.2));});op(gap,seg(p,.85,1));}},
],notes:[
 '<b>b1.</b> Now the problem. Write the potential as a series in the strong coupling. The coefficients do not stay bounded: they grow factorially with the order. The series is asymptotic, not convergent. This is the renormalon problem.',
 '<b>b2.</b> Here is the picture. The terms shrink for a while, reach a minimum, and then blow up. No matter where you truncate, you cannot beat an error of order lambda QCD.',
 '<b>b3.</b> The origin is infrared. Gluons with momenta near lambda QCD contaminate the perturbative potential and leave an irreducible ambiguity in it. It is the same renormalon that afflicts the pole mass.',
 '<b>b4.</b> And here is the symptom. Doubling the scale mu moves a triply heavy baryon mass by about 700 MeV at next-to-next-to-leading order, and the bands at successive orders do not overlap at all. You cannot quote a meaningful uncertainty.',
]});
})();

