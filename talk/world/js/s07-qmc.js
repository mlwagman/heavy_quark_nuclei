'use strict';
/* ================= SCENE 7 — QMC ================= */
(function(){
const {s,b}=scene('Scene 7 / 15 — Method II','Solving the few-body problem: quantum Monte Carlo');
const L=D('div','col',b);L.style.width='700px';
const vc=D('div','card rv',L,'<h2>VMC — variational minimum</h2><div class="m" style="display:flex;align-items:center;flex-wrap:wrap;column-gap:6px"><span>Δ<i>E</i><sub>VMC</sub> = min<sub><i>ω</i></sub></span><span class="frac"><span>⟨Ψ<sub>T</sub>|<i>H</i>|Ψ<sub>T</sub>⟩</span><span class="den">⟨Ψ<sub>T</sub>|Ψ<sub>T</sub>⟩</span></span></div><svg viewBox="0 0 640 190" width="640" height="190" id="vsv"></svg>');
const vsv=vc.querySelector('#vsv');
axes(vsv,50,160,610,16);
mlab(vsv,606,182,'<i>ω</i>',{anchor:'end',size:17});
mlab(vsv,62,28,'⟨<i>H</i>⟩',{size:17});
const vX=u=>50+560*u, vE=u=>150-305*Math.pow(u-0.62,2);
const vcv=E('path',{d:(()=>{let d='';for(let u=0.02;u<=1;u+=0.02)d+=(d?'L':'M')+vX(u).toFixed(1)+' '+vE(u).toFixed(1)+' ';return d;})(),fill:'none',stroke:'var(--c1)','stroke-width':2.5},vsv);dl(vcv);
const ball=E('circle',{cx:vX(0.06),cy:vE(0.06),r:8,fill:'var(--amber)',opacity:0},vsv);
const vmin=E('text',{x:vX(0.62),y:vE(0.62)+26,'text-anchor':'middle','font-size':'16',fill:'var(--amber)',opacity:0},vsv);
vmin.textContent='upper bound + starting point';
const gc=D('div','card rv',L,'<h2>GFMC — imaginary-time projection</h2><div class="m">e<sup>−Hτ</sup> |Ψ<sub>T</sub>⟩ → |Ψ<sub>0</sub>⟩ &nbsp; as τ → ∞</div><svg viewBox="0 0 640 200" width="640" height="200" id="gsv"></svg>');
const gsv=gc.querySelector('#gsv');
axes(gsv,50,170,610,16);
mlab(gsv,606,192,'<i>τ</i>',{anchor:'end',size:17});
mlab(gsv,62,28,'Δ<i>E</i>(<i>τ</i>)',{size:17});
const E0=132;
const gas=E('line',{x1:50,y1:E0,x2:610,y2:E0,stroke:'var(--rose)','stroke-width':1.5,'stroke-dasharray':'7 5'},gsv);dl(gas);
const gX=u=>50+560*u, gY=u=>E0-(E0-42)*Math.exp(-4.2*u);
const gcv=E('path',{d:(()=>{let d='';for(let u=0;u<=1;u+=0.02)d+=(d?'L':'M')+gX(u).toFixed(1)+' '+gY(u).toFixed(1)+' ';return d;})(),fill:'none',stroke:'var(--teal)','stroke-width':2.5},gsv);dl(gcv);
const gT=E('text',{x:330,y:E0+26,'text-anchor':'middle','font-size':'17','font-style':'italic','font-family':'var(--serif)',fill:'var(--rose)',opacity:0},gsv);
gT.textContent='descends monotonically from above onto ΔE₀';
const R=D('div','col',b);R.style.flex='1';
const w1=D('div','card rv',R,'<h2>Trial wavefunctions — per channel</h2><div class="big" style="display:flex;flex-direction:column;gap:8px"><span>mesons: <span class="m tc">e<sup>−r/a₀</sup></span> — Coulombic</span><span>baryons: <span class="tc">product of pair Coulomb factors</span></span><span>tetraquarks: <span class="tc">Hylleraas forms</span> in colour <span class="m">3⊗<span class="ov">3</span></span> / <span class="m">6⊗<span class="ov">6</span></span></span></div>');
const w2=D('div','card rv',R,'<h2>Errors</h2><div class="big">Bootstrap errors: <span class="num">1–10 MeV</span>. Systematics tracked: τ extent, step δτ.</div>');
scenes.push({el:s,beats:[
 {d:16,f(p){rv(vc,seg(p,0,.2));ds(vcv,seg(p,.15,.45));
   const q=seg(p,.42,.98);const u=0.62-0.56*Math.exp(-3.2*q)*Math.cos(7.85*q);op(ball,seg(p,.4,.5));
   ball.setAttribute('cx',vX(u));ball.setAttribute('cy',vE(u)-9);op(vmin,seg(p,.9,1));}},
 {d:20,f(p){rv(gc,seg(p,0,.2));ds(gas,seg(p,.15,.35));ds(gcv,seg(p,.3,.85));op(gT,seg(p,.8,1));}},
 {d:18,f(p){rv(w1,p);}},
 {d:12,f(p){rv(w2,p);}},
],notes:[
 '<b>b1.</b> We solve the few-body problem with quantum Monte Carlo. Variational Monte Carlo first: minimize the energy over the parameters of a trial wavefunction. That gives an upper bound and a good starting point.',
 '<b>b2.</b> Then Green\'s function Monte Carlo: evolve in imaginary time to project onto the ground state. The energy estimate descends monotonically from above onto the true ground-state energy.',
 '<b>b3.</b> The trial wavefunctions are built per channel. Mesons: a Coulombic exponential. Baryons: a product of pair Coulomb factors. Tetraquarks: Hylleraas forms, explicit functions of all the pair distances, in both the triplet-antitriplet and the sextet-antisextet colour configurations.',
 '<b>b4.</b> Statistical errors come from bootstrap, typically 1 to 10 MeV, and the imaginary-time extent and step size are tracked as systematics, so that what we quote is a converged ground-state energy.',
]});
})();

