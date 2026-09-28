'use strict';
/* ================= SCENE 2 — why heavy hadrons ================= */
(function(){
const {s,b}=scene('Scene 2 / 15 — Motivation','Why all-heavy hadrons?');
const L=D('div','col',b);L.style.width='560px';
const lsvg=E('svg',{viewBox:'0 0 560 520',width:560,height:520},L);
const states=[['c <tspan text-decoration="overline">c</tspan>',2,'charmonium'],['b <tspan text-decoration="overline">b</tspan>',2,'bottomonium'],['Ω<sub>ccc</sub>',3,'triply charmed'],['QQ<tspan text-decoration="overline">Q′</tspan><tspan text-decoration="overline">Q′</tspan>',4,'all-heavy tetraquark']];
const icons=states.map((st,i)=>{
  const cx=140+280*(i%2), cy=130+260*(i>>1);
  const g=E('g',{transform:`translate(${cx},${cy}) scale(0)`},lsvg);
  E('circle',{cx:0,cy:0,r:62,fill:'var(--panel)',stroke:'var(--line)'},g);
  const cols=['var(--teal)','var(--amber)','var(--rose)','var(--c1)'];
  for(let q=0;q<st[1];q++)orbDot(g,1,0,0,st[1]===2?26:34,0.7+0.15*q,q*6.28/st[1],cols[q],8);
  const tx=E('text',{x:0,y:92,'text-anchor':'middle','font-size':'22','font-family':'var(--serif)','font-style':'italic'},g);
  tx.innerHTML=st[0].replace(/<sub>(.*?)<\/sub>/,'$1');
  E('text',{x:0,y:114,'text-anchor':'middle','font-size':'16',class:'axl'},g).textContent=st[2];
  return {g,cx,cy};
});
const R=D('div','col',b);R.style.flex='1';
const h=D('div','card rv',R,'<h2>Clean scale hierarchy</h2><div id="hb"></div><div class="small" style="margin-top:6px">⇒ controlled expansions in <i>α<sub>s</sub></i> and <i>v</i></div>');
const hb=h.querySelector('#hb');
const hier=[['m<sub>Q</sub>',420,'var(--teal)'],['m<sub>Q</sub>v',150,'var(--amber)'],['m<sub>Q</sub>v²',52,'var(--rose)']].map(x=>{
  const row=D('div','',hb);row.style.cssText='display:flex;align-items:center;gap:12px;margin:4px 0';
  D('div','m',row,x[0]).style.width='64px';
  const bar=D('div','',row);bar.style.cssText=`height:16px;border-radius:3px;background:${x[2]};width:0px`;
  return {bar,w:x[1]};
});
const tri=D('div','card rv',R,'<h2>Precision laboratory</h2><svg viewBox="0 0 420 150" width="420" height="108" id="tsv"></svg>');
const tsv=tri.querySelector('#tsv');
const tn=[[70,120,'EFT'],[350,120,'experiment'],[210,26,'lattice QCD']];
const te=[E('line',{x1:70,y1:120,x2:350,y2:120,stroke:'var(--muted)','stroke-width':1.5},tsv),
          E('line',{x1:70,y1:120,x2:210,y2:26,stroke:'var(--muted)','stroke-width':1.5},tsv),
          E('line',{x1:350,y1:120,x2:210,y2:26,stroke:'var(--muted)','stroke-width':1.5},tsv)];
te.forEach(dl);
tn.forEach(n=>{E('circle',{cx:n[0],cy:n[1],r:5,fill:'var(--teal)'},tsv);
  E('text',{x:n[0],y:n[1]+(n[1]>60?24:-14),'text-anchor':'middle','font-size':'17'},tsv).textContent=n[2];});
const tlab=E('text',{x:210,y:95,'text-anchor':'middle','font-size':'19','font-style':'italic','font-family':'var(--serif)',fill:'var(--amber)',opacity:0},tsv);
tlab.textContent='compare at ~10 MeV';
const tim=D('div','card rv',R,'<h2>Timely</h2><div style="display:flex;flex-wrap:wrap;gap:4px 0"><span class="chip">lattice Ω<sub>QQQ</sub> at ±10–20 MeV</span><span class="chip rc">CMS 2025: &gt;5σ at t <span class="ov">t</span> threshold (§I)</span><span class="chip">ATLAS 2026 — same excess</span></div><svg viewBox="0 0 420 84" width="420" height="72" id="bsv"></svg>');
const bsv=tim.querySelector('#bsv');
E('line',{x1:16,y1:66,x2:404,y2:66,stroke:'var(--muted)','stroke-width':1.5},bsv);
const bump=E('path',{d:'',fill:'var(--rose)','fill-opacity':.3,stroke:'var(--rose)','stroke-width':2},bsv);
E('text',{x:212,y:80,'text-anchor':'middle','font-size':'15',class:'axl'},bsv).textContent='√s → 2m_t — toponium-like excess';
function bumpD(a){let d='M16 66 ';for(let x=16;x<=404;x+=4){const y=66-a*46*Math.exp(-Math.pow((x-210)/34,2));d+=`L${x} ${y.toFixed(1)} `;}return d;}
const ob=D('div','card rv',R,'<h2 class="rc">The obstacle</h2><div id="obr"></div><div class="banner rv" id="obb" style="font-size:20px;margin-top:8px">This talk: remove the leading renormalon.</div>');
const obr=ob.querySelector('#obr'), obb=ob.querySelector('#obb');
const or1=D('div','',obr);or1.style.cssText='display:flex;align-items:center;gap:12px;margin:4px 0';
D('div','small',or1,'target precision').style.width='170px';
const sb=D('div','',or1);sb.style.cssText='height:14px;border-radius:3px;background:var(--teal);width:38px';
D('div','small num',or1,'~10 MeV');
const or2=D('div','',obr);or2.style.cssText='display:flex;align-items:center;gap:12px;margin:4px 0';
D('div','small',or2,'fixed-order scale dependence').style.width='170px';
const rb=D('div','',or2);rb.style.cssText='height:14px;border-radius:3px;background:var(--red);width:0px';
const rbl=D('div','small num',or2,'');
scenes.push({el:s,beats:[
 {d:16,f(p){icons.forEach((ic,i)=>{const u=back(seg(p,i*0.18,i*0.18+0.35));ic.g.setAttribute('transform',`translate(${ic.cx},${ic.cy}) scale(${Math.max(0,u)})`);});}},
 {d:13,f(p){rv(h,seg(p,0,.2));hier.forEach((r,i)=>{r.bar.style.width=(r.w*eo(seg(p,.15+i*.22,.5+i*.22)))+'px';});}},
 {d:13,f(p){rv(tri,seg(p,0,.2));te.forEach((e,i)=>ds(e,seg(p,.15+i*.18,.5+i*.18)));op(tlab,seg(p,.75,1));}},
 {d:18,f(p){rv(tim,seg(p,0,.25));bump.setAttribute('d',bumpD(eo(seg(p,.45,.9))));}},
 {d:17,f(p){rv(ob,seg(p,0,.2));const w=300*eo(seg(p,.2,.65));rb.style.width=w+'px';
   const v=Math.round(100*eo(seg(p,.2,.65)));rbl.innerHTML='~'+v+' MeV'+(v>=100?' (quarkonium) (§III A)<br>~700–850 MeV (Ω masses, Tab. III)':'');
   rv(obb,seg(p,.75,1));}},
],notes:[
 '<b>b1.</b> Why should you care about hadrons made only of heavy quarks? Because they are the cleanest bound states QCD offers: charmonium, bottomonium, the triply charmed Omega, and the all-heavy tetraquarks.',
 '<b>b2.</b> In every one of them the quark mass is much larger than the momentum, which is much larger than the binding energy. That separation of scales gives controlled expansions in the coupling and in the velocity.',
 '<b>b3.</b> So they are a precision laboratory: a place where effective field theory, lattice QCD, and experiment can all be compared at the ten MeV level.',
 '<b>b4.</b> And the timing is right. Lattice calculations of triply heavy baryons now quote 10 to 20 MeV errors. And CMS in 2025, then ATLAS in 2026, report a five sigma excess at the top anti-top threshold, toponium-like.',
 '<b>b5.</b> Here is the obstacle. Fixed-order potential N R Q C D carries a scale dependence of order 100 MeV in binding energies, and 700 to 850 MeV in the triply heavy baryon masses. That is as large as the signal we are chasing. This talk is about removing that obstacle.',
]});
})();

