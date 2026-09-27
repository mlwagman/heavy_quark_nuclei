'use strict';
/* ================= SCENE 10 — baryons vs lattice ================= */
(function(){
const {s,b}=scene('Scene 10 / 15 — Results III','Triply-heavy baryons vs. lattice QCD');
b.style.flexDirection='column';
const tb=D('div','card',b,'<table class="tbl" style="width:100%"><tr><th></th><th>Ω<sub>ccc</sub></th><th>Ω<sub>ccb</sub></th><th>Ω<sub>cbb</sub></th><th>Ω<sub>bbb</sub></th></tr>'+
 '<tr id="ro"><th>NNLO MRS [GeV]</th><td class="rv">4.6186(1)(34)</td><td class="rv">7.8511(1)(52)</td><td class="rv">11.0701(1)(83)</td><td class="rv">14.2311(1)(112)</td></tr>'+
 '<tr id="rl"><th>lattice QCD [GeV]</th><td class="rv">4.796(8)(18)</td><td class="rv">8.005(6)(11)</td><td class="rv">11.194(5)(12)</td><td class="rv">14.366(7)(9)</td></tr>'+
 '<tr id="rd"><th>difference [MeV]</th><td class="rv rc">~175</td><td class="rv rc">~155</td><td class="rv rc">~125</td><td class="rv rc">~135</td></tr></table>');
const ro=[...tb.querySelectorAll('#ro .rv')], rl=[...tb.querySelectorAll('#rl .rv')], rd=[...tb.querySelectorAll('#rd .rv')];
const row2=D('div','',b);row2.style.cssText='display:flex;gap:16px;margin-top:14px;flex:1';
const bandC=D('div','card rv',row2,'<h2>The scale band, Ω<sub>bbb</sub> at NNLO</h2><svg viewBox="0 0 520 150" width="520" height="150" id="bs"></svg><div class="banner" style="font-size:20px"><span id="bandN" class="num">700</span> MeV → <span class="tc">nearly two orders of magnitude</span></div>');
bandC.style.flex='1';
const bs=bandC.querySelector('#bs'), bandN=bandC.querySelector('#bandN');
E('line',{x1:30,y1:75,x2:490,y2:75,stroke:'var(--muted)','stroke-width':1},bs);
const brect=E('rect',{x:30,y:45,width:460,height:60,rx:5,fill:'var(--red)','fill-opacity':.45},bs);
E('text',{x:30,y:32,'font-size':'16',class:'axl'},bs).textContent='fixed order ~700 MeV';
const blab=E('text',{x:30,y:132,'font-size':'16',fill:'var(--teal)',opacity:0},bs);
blab.textContent='MRS ~10 MeV';
const gapC=D('div','card rv',row2,'<h2>The undershoot is a pattern</h2><svg viewBox="0 0 520 160" width="520" height="160" id="gs"></svg><div class="big">fractional gap ∼ 1/<i class="m">m<sub>Q</sub></i> ⇒ missing <span class="rc">𝒪(1/m<sub>Q</sub>) + spin</span> — not noise</div>');
gapC.style.flex='1';
const gs=gapC.querySelector('#gs');
const gaps=[3.8,1.9,1.1,1.0];
const gbars=gaps.map((g,i)=>{
  E('text',{x:70+120*i,y:150,'text-anchor':'middle','font-size':'16',class:'axl'},gs).textContent=['Ωccc','Ωccb','Ωcbb','Ωbbb'][i];
  const r=E('rect',{x:44+120*i,y:130,width:52,height:0,rx:4,fill:'var(--rose)','fill-opacity':.6},gs);
  const t=E('text',{x:70+120*i,y:120,'text-anchor':'middle','font-size':'17',fill:'var(--rose)',opacity:0},gs);
  t.textContent=g.toFixed(1)+'%';return {r,t,g};});
const top=D('div','card rv',b,'<div class="big" style="margin:0">Top-quark benchmarks up to <span class="m num">Ω<sub>ttt</sub> = 514.46(4)(94) GeV</span> &nbsp;·&nbsp; <span class="small">NB fixed order sits closer to lattice (80–110 MeV below) — the MRS advantage is the band, not the central value</span></div>');
top.style.marginTop='14px';
scenes.push({el:s,beats:[
 {d:16,f(p){ro.forEach((c,i)=>rv(c,seg(p,i*.2,.3+i*.2)));}},
 {d:18,f(p){rl.forEach((c,i)=>rv(c,seg(p,i*.18,.3+i*.18)));}},
 {d:18,f(p){rv(bandC,seg(p,0,.2));const u=eo(seg(p,.25,.7));
   const wpx=460*(1-u)+5.75*u;
   brect.setAttribute('width',wpx);brect.setAttribute('x',30);brect.setAttribute('fill',u>0.85?'var(--teal)':'var(--red)');
   bandN.textContent=Math.max(10,Math.round(700*Math.pow(10,-1.845*u))); if(u>=1)bandN.textContent='10';
   op(blab,seg(p,.7,.85));rd.forEach((c,i)=>rv(c,seg(p,.78+i*.05,.9+i*.05)));}},
 {d:16,f(p){rv(gapC,seg(p,0,.2));gbars.forEach((gb,i)=>{const u=eo(seg(p,.15+i*.14,.4+i*.14));
   gb.r.setAttribute('height',u*gb.g*26);gb.r.setAttribute('y',130-u*gb.g*26);op(gb.t,seg(p,.3+i*.14,.5+i*.14));});}},
 {d:12,f(p){rv(top,p);}},
],notes:[
 '<b>b1.</b> Triply heavy baryons, where we meet lattice QCD. The first row is ours: next-to-next-to-leading order with minimal renormalon subtraction, from triply charm to triply bottom, each with its statistical and its scale uncertainty.',
 '<b>b2.</b> The second row is lattice QCD. We undershoot the lattice by 125 to 175 MeV. Note that fixed order actually sits closer, 80 to 110 MeV below the lattice. The advantage of M R S is not in the central value. It is in what comes next.',
 '<b>b3.</b> Here is the scale band for the triply bottom Omega at next-to-next-to-leading order. Fixed order: about 700 MeV. M R S: about 10 MeV. Nearly two orders of magnitude, and that is what makes the comparison with the lattice meaningful.',
 '<b>b4.</b> And the undershoot is a pattern, not noise. As a fraction of the mass, the gap falls from 3.8 percent for triply charm to 1.0 percent for triply bottom. A one over the heavy quark mass trend: exactly what you expect if what is missing are the one over m Q corrections and the spin-dependent terms.',
 '<b>b5.</b> The same machinery gives benchmarks with top quarks, up to the triply top Omega at 514.46 GeV, with an uncertainty of 94 in the last digits.',
]});
})();

