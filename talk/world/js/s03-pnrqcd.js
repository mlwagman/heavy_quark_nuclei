'use strict';
/* ================= SCENE 3 — pNRQCD ================= */
(function(){
const {s,b}=scene('Scene 3 / 15 — Background I','pNRQCD: QCD as a Schrödinger problem');
const L=D('div','col',b);L.style.width='560px';
const tw=E('svg',{viewBox:'0 0 560 520',width:560,height:520},L);
const boxes=[['QCD','scale m<tspan font-size="13" dy="3">Q</tspan>',60],['NRQCD','scale m<tspan font-size="13" dy="3">Q</tspan>v',220],['pNRQCD','scale m<tspan font-size="13" dy="3">Q</tspan>v²',380]].map((bx,i)=>{
  const g=E('g',{transform:`translate(235,${bx[2]+40}) scale(0)`,opacity:0},tw);
  E('rect',{x:-130,y:-40,width:260,height:80,rx:8,fill:'var(--panel)',stroke:i===2?'var(--teal)':'var(--line)','stroke-width':i===2?2:1},g);
  E('text',{x:0,y:-6,'text-anchor':'middle','font-size':'27','font-family':'var(--serif)'},g).textContent=bx[0];
  const t2=E('text',{x:0,y:22,'text-anchor':'middle','font-size':'17',class:'axl','font-style':'italic'},g);t2.innerHTML=bx[1];
  return g;});
const arrows=[0,1].map(i=>{
  const a=E('line',{x1:235,y1:140+160*i,x2:235,y2:200+160*i,stroke:'var(--amber)','stroke-width':2.5},tw);dl(a);
  const t=mlab(tw,252,162+160*i,'<span>integrate out <i>p</i> ~ <i>m<sub>Q</sub></i>'+(i?'<i>v</i>':'')+'</span>',{size:17,color:'var(--amber)',w:300});op(t,0);
  const t2=mlab(tw,252,188+160*i,'<span>'+(i?'soft modes → the potentials <i>V<sub>R</sub></i>(<i>r</i>)':'hard modes → matching coefficients')+'</span>',{size:14,color:'var(--muted)',w:300});op(t2,0);
  return {a,t,t2};});
const cap=E('text',{x:235,y:492,'text-anchor':'middle','font-size':'19','font-style':'italic','font-family':'var(--serif)',opacity:0},tw);
cap.textContent='what remains is quantum mechanics';
const R=D('div','col',b);R.style.flex='1';
const eq=D('div','card rv',R,'<h2>The Hamiltonian</h2><div class="m" style="display:flex;align-items:center;flex-wrap:wrap;column-gap:6px"><span><i>H</i> =</span><span class="sum"><span class="sg">Σ</span><span class="lim"><i>i</i></span></span><span class="frac"><span>−∇<sup>2</sup><sub><i>i</i></sub></span><span class="den">2<i>m<sub>i</sub></i></span></span><span>+</span><span class="sum"><span class="sg">Σ</span><span class="lim"><i>i</i>&lt;<i>j</i></span></span><span><i>V<sub>R</sub></i>(<i>r<sub>ij</sub></i>),</span><span style="width:14px"></span><span><i>V</i><sub>1</sub><sup>Q<span class="ov">Q</span></sup> = −<i>C<sub>F</sub></i></span><span class="frac"><span><i>α<sub>V</sub></i>(<i>r</i>,<i>μ</i>)</span><span class="den"><i>r</i></span></span></div><div class="small" style="margin-top:6px">an ordinary Schrödinger problem (+ 3-body terms at NNLO) — potentials <b>matched</b> to QCD, not modelled</div>');
const mt=D('div','card rv',R,'<h2>Matching</h2><div class="big">Potentials known to <b>NNLO</b> in <i class="m">α<sub>s</sub></i></div><div style="display:flex;gap:24px;margin-top:10px"><div class="chip m">Q<span class="ov">Q</span> singlet: <span class="ac">C<sub>F</sub> = 4/3</span></div><div class="chip m">QQ in baryon: <span class="ac">C<sub>F</sub>/2</span></div></div><div class="small" style="margin-top:8px">3-body forces enter at NNLO — included at fixed order (no resummation known)</div>');
const st=D('div','card rv',R,'<h2>Static limit</h2><div class="big">Spin effects are 𝒪(<i class="m">v</i>²) — ~5% for <i>b</i>, ~10% for <i>c</i> — dropped here</div>');
scenes.push({el:s,beats:[
 {d:18,f(p){boxes.forEach((g,i)=>{const u=back(seg(p,i*0.3,i*0.3+0.28));
    g.setAttribute('transform',`translate(235,${100+160*i}) scale(${Math.max(0,u)})`);op(g,seg(p,i*0.3,i*0.3+0.15));});
   arrows.forEach((ar,i)=>{ds(ar.a,seg(p,.22+.3*i,.34+.3*i));op(ar.t,seg(p,.26+.3*i,.4+.3*i));op(ar.t2,seg(p,.32+.3*i,.46+.3*i));});
   op(cap,seg(p,.85,1));}},
 {d:14,f(p){rv(eq,p);}},
 {d:17,f(p){rv(mt,p);}},
 {d:14,f(p){rv(st,p);}},
],notes:[
 '<b>b1.</b> First, the framework. Start from QCD at the heavy-quark mass. Integrate out the hard modes: nonrelativistic QCD. Integrate out the soft modes: potential N R Q C D, at the binding-energy scale. What remains is quantum mechanics.',
 '<b>b2.</b> The Hamiltonian is a kinetic term for each quark plus a potential between each pair. Those potentials are not modelled. They are matched to QCD, order by order in the coupling.',
 '<b>b3.</b> The matching is known to next-to-next-to-leading order. The colour factors differ by channel: four thirds for a quark-antiquark singlet, and half of that for a quark pair inside a baryon. Three-body forces enter at that order, and we include them.',
 '<b>b4.</b> We work in the static limit. Spin-dependent effects are suppressed by the velocity squared, about 5 percent for bottom and 10 percent for charm, and they are dropped here.',
]});
})();

