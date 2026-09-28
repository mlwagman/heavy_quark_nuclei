'use strict';
/* ================= SCENE 6 — mu'(r) ================= */
(function(){
const {s,b}=scene('Scene 6 / 15 — Method I','Making MRS work inside a bound-state solver');
const L=D('div','card',b);L.style.width='880px';
D('h2','',L,'The interpolating scale');
const psv=E('svg',{viewBox:'0 0 860 470',width:860,height:470},L);
const X=r=>70+(r/3)*770, Y=m=>420-(m/4)*390;
axes(psv,70,420,840,20);
mlab(psv,836,444,'<i>r</i> [fm]',{anchor:'end',size:18});
mlab(psv,94,28,'<i>μ</i> [GeV]',{size:18});
E('rect',{x:70,y:Y(0.45),width:770,height:420-Y(0.45),fill:'var(--red)','fill-opacity':.10},psv);
mlab(psv,834,Y(0.13),'nonperturbative — <i>α<sub>s</sub></i> blows up',{anchor:'end',size:17,color:'var(--red)'});
const mu=(r,lam,cut)=>Math.sqrt(cut*cut+lam*lam/(r*r));
function curveD(f){let d='';for(let r=0.14;r<=3.001;r+=0.03){const y=Math.min(4.4,f(r));d+=(d?'L':'M')+X(r).toFixed(1)+' '+Y(Math.min(y,4.2)).toFixed(1)+' ';}return d;}
const c1=E('path',{d:curveD(r=>0.55/r),fill:'none',stroke:'var(--teal)','stroke-width':2.5},psv);dl(c1);
const cutL=E('line',{x1:70,y1:Y(1),x2:840,y2:Y(1),stroke:'var(--amber)','stroke-width':1.5,'stroke-dasharray':'7 5',opacity:0},psv);
const cutT=mlab(psv,834,Y(1)-16,'<i>μ</i><sub>cut</sub> = 1 GeV — saturates',{anchor:'end',size:17,color:'var(--amber)'});op(cutT,0);
const frm=mlab(psv,470,52,'<i>μ′</i>(<i>r</i>; <i>λ</i>) = √<span class="sqrt"><i>μ</i><sup>2</sup><sub>cut</sub> + <i>λ</i><sup>2</sup>/<i>r</i><sup>2</sup></span>',{anchor:'middle',size:28,color:'var(--ink)'});op(frm,0);
const wf=E('path',{d:'',fill:'var(--teal)','fill-opacity':.22,stroke:'none',opacity:0},psv);
function wfD(a){let d='M'+X(0.15)+' '+Y(0);for(let r=0.15;r<=2.6;r+=0.04){const h=a*1.1*Math.exp(-Math.pow((r-0.75)/0.42,2));d+='L'+X(r).toFixed(1)+' '+Y(h).toFixed(1)+' ';}return d+'L'+X(2.6)+' '+Y(0)+'Z';}
const wfT=mlab(psv,X(0.78),Y(1.32),'|<i>ψ</i>|<sup>2</sup> — bound states live here',{anchor:'middle',size:17,color:'var(--teal)'});op(wfT,0);
const g2=E('path',{d:curveD(r=>mu(r,0.55,2)),fill:'none',stroke:'var(--muted)','stroke-width':1.8,'stroke-dasharray':'5 5',opacity:0},psv);
const g2T=mlab(psv,X(1.9),Y(2)-16,'<i>μ</i><sub>cut</sub> 2 GeV → fixed order (Fig. 7: frozen <i>α<sub>s</sub></i>)',{size:16});op(g2T,0);
const bandC=E('path',{d:'',fill:'var(--rose)','fill-opacity':.16,opacity:0},psv);
(function(){const xs=[],lo=[],hi=[];for(let r=0.14;r<=3.001;r+=0.05){xs.push(r);lo.push(Math.min(4.2,mu(r,0.275,1)));hi.push(Math.min(4.2,mu(r,1.1,1)));}
 bandC.setAttribute('d',bandPath(X,Y,xs,lo,hi));})();
const lamT=mlab(psv,X(0.62),Y(3.5),'<i>λ</i> ∈ {½, 1, 2} — the scale probe',{size:17,color:'var(--rose)'});op(lamT,0);
const R=D('div','col',b);R.style.flex='1';
const k1=D('div','card rv',R,'<h2>Why not μ = 1/r?</h2><div class="big">A wavefunction samples <i>all</i> separations — at large <i class="m">r</i> you walk straight out of the perturbative regime.</div>');
const k2=D('div','card rv',R,'<h2>Smooth interpolation</h2><div class="big"><i class="m">μ′</i> ≈ <i class="m">λ</i>/<i class="m">r</i> at short distance → saturates at <span class="ac num">1 GeV</span>. No kinks in the force — stable Monte Carlo.</div>');
const k3=D('div','card rv',R,'<h2>The cutoff matters</h2><div class="big"><span class="ac num">1 GeV</span> keeps MRS <i>active</i> where bound states live. <span class="num">2 GeV</span> collapses the scheme back to fixed order — nothing gained.</div>');
const k4=D('div','card rv',R,'<h2>What we vary</h2><div class="big"><span class="rc">λ ∈ {½, 1, 2}</span> — the analogue of the usual factor-of-two variation.<br><i class="m">n<sub>f</sub></i> = 3 / 4 / 5, set by the lightest valence quark (<i>c</i>/<i>b</i>/<i>t</i>).</div>');
scenes.push({el:s,beats:[
 {d:18,f(p){ds(c1,seg(p,.1,.6));rv(k1,seg(p,.5,.9));},
  z(){c1.setAttribute('d',curveD(r=>0.55/r));dl(c1);rv(k1,0);}},
 {d:18,f(p){const m=eo(seg(p,.1,.6));c1.setAttribute('d',curveD(r=>(1-m)*(0.55/r)+m*mu(r,0.55,1)));
    if(p>0.05)c1.style.strokeDasharray='none';
    op(cutL,seg(p,.5,.7));op(cutT,seg(p,.55,.75));op(frm,seg(p,.6,.85));rv(k2,seg(p,.7,1));}},
 {d:20,f(p){op(wf,seg(p,0,.3));wf.setAttribute('d',wfD(eo(seg(p,.05,.45))));op(wfT,seg(p,.4,.6));
    op(g2,seg(p,.55,.75));op(g2T,seg(p,.65,.85));rv(k3,seg(p,.7,1));}},
 {d:19,f(p){op(bandC,seg(p,.1,.4));op(lamT,seg(p,.3,.55));rv(k4,seg(p,.5,.9));}},
],notes:[
 '<b>b1.</b> Inside a bound-state solver there is a difficulty. A wavefunction samples all separations. Set the scale to one over r and at large distance you walk out of the perturbative regime.',
 '<b>b2.</b> So we use a smooth interpolating scale: lambda over r at short distance, saturating at 1 GeV at long distance. Smooth means no kinks in the force, and no kinks means the Monte Carlo stays stable.',
 '<b>b3.</b> The cutoff matters. At 1 GeV the subtraction stays active exactly where the bound states live. Push it to 2 GeV and the scheme collapses back to fixed order.',
 '<b>b4.</b> What we vary: lambda over one half, one, and two, the analogue of the usual factor-of-two scale variation. And the number of light flavours, 3, 4 or 5, is set by the lightest valence quark: charm, bottom or top.',
]});
})();

