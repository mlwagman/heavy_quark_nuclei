'use strict';
/* ================= SCENE 5 — MRS ================= */
(function(){
const {s,b}=scene('Scene 5 / 15 — Background III','Minimal renormalon subtraction (MRS)');
b.style.flexDirection='column';
const th=D('div','rv',b,'<span style="font:italic 23px/1.3 var(--serif)">Isolate the growth, <span class="ac">resum it</span>, make the ambiguity <span class="rc">one explicit constant</span>.</span>');
const eqrow=D('div','',b);eqrow.style.cssText='display:flex;align-items:stretch;gap:0;margin:10px 0 4px';
D('div','m',eqrow,'<i>V<sub>R</sub></i>(<i>r</i>) &nbsp;=&nbsp;').style.cssText='display:flex;align-items:center;font-size:26px';
const T1=D('div','card',eqrow,'<div class="m" style="font-size:22px;display:flex;align-items:center;flex-wrap:wrap;column-gap:5px"><span>−</span><span class="frac"><span><i>C<sub>R</sub></i></span><span class="den"><i>r</i></span></span><span class="sum"><span class="ulim"><i>l</i><sub>max</sub></span><span class="sg">Σ</span><span class="lim"><i>l</i>=0</span></span><span>[<i>v<sub>l</sub></i> − <i>V<sub>l</sub></i>] <i>α<sub>s</sub></i><sup>l+1</sup>(<i>μ′</i>(<i>r</i>))</span></div><div class="small tc" style="margin-top:8px;max-width:400px">leading renormalon subtracted — no factorial growth at the leading rate. <i>V<sub>l</sub></i> = the renormalon part of <i>v<sub>l</sub></i>, known to <i>l</i><sub>max</sub> = 2 (NNLO)</div>');
const O1=D('div','m',eqrow,'&nbsp;−&nbsp;');O1.style.cssText='display:flex;align-items:center;font-size:26px';
const T2=D('div','card',eqrow,'<div class="m" style="font-size:20px;display:flex;align-items:center;flex-wrap:wrap;column-gap:5px"><span class="frac"><span><i>C<sub>R</sub></i></span><span class="den"><i>r</i></span></span><span class="sum"><span class="ulim">∞</span><span class="sg">Σ</span><span class="lim"><i>l</i>=0</span></span><span><i>V<sub>l</sub></i> <i>α<sub>s</sub></i><sup>l+1</sup> =</span><span class="frac"><span><i>C<sub>R</sub></i></span><span class="den"><i>r</i></span></span><span class="frac"><span><i>V</i><sub>0</sub></span><span class="den">2<i>β</i><sub>0</sub></span></span><span>𝒥(<i>b</i>,</span><span class="frac"><span>1</span><span class="den">2<i>β</i><sub>0</sub><i>α<sub>s</sub></i></span></span><span>)</span></div><div class="small ac" style="margin-top:8px;max-width:340px">the full renormalon series, resummed to ∞ in closed form</div>');
const O2=D('div','m',eqrow,'&nbsp;+&nbsp;');O2.style.cssText='display:flex;align-items:center;font-size:26px';
const T3=D('div','card',eqrow,'<div class="m" style="font-size:26px">Λ<sub>R</sub></div><div class="small rc" style="margin-top:8px;max-width:240px"><b>one power correction</b> — free of the leading-renormalon ambiguity; absorbed into <i>m<sub>Q</sub></i></div>');
T3.style.cssText+=';box-shadow:0 0 0 1px var(--rose)';
T1.style.borderColor='var(--teal)';T2.style.borderColor='var(--amber)';T3.style.borderColor='var(--rose)';
const jn=D('div','rv small',b,'𝒥(<i>c</i>,<i>y</i>) = e<sup>−y</sup> Γ(−<i>c</i>) γ*(−<i>c</i>,−<i>y</i>) — an incomplete-gamma closed form');jn.style.cssText='margin-top:2px';
const lr=D('div','card rv',b,'<h2 class="rc">Where does the renormalon go?</h2><div class="big" style="max-width:1300px"><span class="mx">Λ<sub>R</sub></span> comes out <b>free of the leading-renormalon ambiguity</b>. For the <b>singlet</b> it is absorbed into the quark mass: <span class="mx">m<sub>Q</sub> + Λ<sub>R</sub></span> is <span class="rc"><b>fixed to the spin-averaged 1S states</b></span>. For <b>non-singlet</b> channels (baryons, tetraquarks) it is neglected — extending MRS there is a scheme choice, left to future work.</div>');
lr.style.cssText+=';margin-top:8px;border-color:var(--rose)';
const row2=D('div','',b);row2.style.cssText='display:flex;gap:16px;margin-top:8px';
const v0=D('div','card rv',row2,'<h2>Normalization has converged</h2><div class="m num" style="font-size:24px"><i>V</i><sub>0</sub> = <span id="v0v">0.771</span> <span class="small">(NNLO)</span> → 0.772 <span class="small">(N³LO)</span> <span class="tc">✓</span> <span class="small">(§II B)</span></div>');
const mn=D('div','card rv',row2,'<h2>“Minimal” matters</h2><div class="big">Same <span class="ov">MS</span> inputs — the series is only reorganized: the renormalon tail is isolated and resummed (Komijani ’17; Kronfeld ’23).</div>');
const nw=D('div','card rv',row2,'<h2 class="rc">What is new</h2><div class="big">Proven for quark masses — all flavours except top (TUMQCD). <span class="rc">First use inside a bound-state problem.</span></div>');
scenes.push({el:s,beats:[
 {d:12,f(p){rv(th,p);}},
 {d:30,f(p){[T1,O1,T2,O2,T3].forEach((e,i)=>{rv(e,seg(p,i*0.16,i*0.16+0.22));});rv(jn,seg(p,.85,1));}},
 {d:18,f(p){rv(lr,p);}},
 {d:15,f(p){rv(v0,p);}},
 {d:24,f(p){rv(mn,seg(p,0,.45));rv(nw,seg(p,.5,.95));}},
],notes:[
 '<b>b1.</b> Here is the idea. Instead of ignoring the factorial growth, isolate it, resum it in closed form, and push the entire remaining ambiguity into one explicit constant.',
 '<b>b2.</b> The potential becomes three pieces. First, the truncated series with the renormalon part subtracted term by term, so the leading factorial growth is gone. Second, the renormalon series itself, summed from the first term to infinity in closed form. Third, lambda R, a single power correction.',
 '<b>b3.</b> Lambda R is the key. It is free of the leading renormalon ambiguity. For the colour singlet it is absorbed into the quark mass, which is then fixed to the spin-averaged one S states. For the non-singlet channels, the baryons and the tetraquarks, it is neglected: extending M R S there is a scheme choice, left to future work.',
 '<b>b4.</b> The normalization of the renormalon is well determined: 0.771 at next-to-next-to-leading order, and 0.772 at the next order. It has converged.',
 '<b>b5.</b> And minimal matters. With the same M S bar inputs, the series is only reorganized: the renormalon tail is isolated and resummed, following Komijani in 2017 and Kronfeld in 2023. It is proven for quark masses, all flavours except top. This is its first use inside a bound-state problem.',
]});
})();

