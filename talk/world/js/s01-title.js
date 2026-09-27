'use strict';
/* ================= SCENE 1 — title ================= */
(function(){
const {s,b}=scene('Minimal renormalon subtraction in pNRQCD · July 2026 · FERMILAB-PUB-26-0061-T');
b.style.cssText='flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:22px';
const orb=D('div','',b);const osvg=E('svg',{width:220,height:120,viewBox:'0 0 220 120'},orb?orbFix(orb):orb);
function orbFix(x){return x;}
orbDot(osvg,0,110,60,44,0.9,0,'var(--teal)',7);
orbDot(osvg,0,110,60,44,0.9,2.09,'var(--amber)',7);
orbDot(osvg,0,110,60,44,0.9,4.19,'var(--rose)',7);
E('circle',{cx:110,cy:60,r:44,fill:'none',stroke:'var(--line)','stroke-width':1},osvg);
const T=D('div','rv',b,'<div style="font:400 52px/1.2 var(--serif);text-wrap:balance">Renormalon-subtracted nonrelativistic QCD<br>for heavy-hadron systems</div>');
const A=D('div','rv',b,'<div class="big">Benoît Assi · Andreas S. Kronfeld · <b>Simon Vaiva</b> · Michael L. Wagman</div>');
const F=D('div','rv small',b,'Fermilab Theory Division &nbsp;·&nbsp; Aix-Marseille Univ, CPT &nbsp;·&nbsp; Univ. of Cincinnati');
const X=D('div','rv',b,'<span class="chip num" style="border-color:var(--teal);color:var(--teal)">arXiv:2607.08817</span>');
const TH=D('div','',b);
const t1=D('span','rv',TH,'A decades-old infrared obstruction — the leading renormalon&nbsp; ');
const t2=D('span','rv',TH,'→&nbsp; removed.&nbsp; ');
const t3=D('span','rv rc',TH,'→&nbsp; a whole family of states becomes predictable.');
TH.style.cssText='font:italic 25px/1.4 var(--serif);margin-top:10px;max-width:1000px'; /* wraps clear of the QR block (animation review 2026-09-11) */
const QR=D('div','rv',s);
QR.style.cssText='position:absolute;right:56px;bottom:18px;display:flex;flex-direction:column;align-items:center;gap:7px';
QR.innerHTML='<img src="assets/qr.png" alt="arXiv" style="width:200px;display:block;border-radius:6px"><span style="font:500 15px/1 var(--mono);letter-spacing:.1em;color:var(--muted)">arxiv.org/abs/2607.08817</span>';
scenes.push({el:s,orb:0,beats:[
 {d:20,f(p){rv(T,seg(p,0,.3));rv(A,seg(p,.25,.5));rv(F,seg(p,.45,.65));rv(X,seg(p,.6,.85));rv(QR,seg(p,.7,.95));}},
 {d:24,f(p){rv(t1,seg(p,0,.3));rv(t2,seg(p,.35,.6));rv(t3,seg(p,.65,.95));}},
],notes:[
 '<b>b1.</b> Thank you. I am Simon Vaiva, and this talk is about renormalon-subtracted nonrelativistic QCD for heavy-hadron systems. It is work with Benoit Assi, Andreas Kronfeld, and Michael Wagman, between Fermilab, the Centre de Physique Theorique in Marseille, and the University of Cincinnati. The paper appeared in July on the arXiv as 2607.08817.',
 '<b>b2.</b> The short version is this. A well-known infrared obstruction has limited perturbative predictions for heavy hadrons for decades. We remove it. And once it is removed, a whole family of states that was out of reach becomes predictable. That is the thesis of this talk.',
]});
})();

