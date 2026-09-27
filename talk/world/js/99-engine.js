'use strict';
/* ================= engine ================= */
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
if(RM)scenes.forEach(sc=>sc.beats.forEach(b=>b.d=0.15));
let si=0, done=0, run=null, paused=false, tAmb=0, last=null;
const $=id=>document.getElementById(id);
const dotsEl=$('dots'), prog=$('prog'), scix=$('scix'), notesEl=$('notes');
let notesOn=false;
function ui(){
  const B=scenes[si].beats;
  scix.textContent=(si+1)+' / '+scenes.length;
  while(dotsEl.children.length<B.length)dotsEl.appendChild(document.createElement('span'));
  while(dotsEl.children.length>B.length)dotsEl.removeChild(dotsEl.lastChild);
  [...dotsEl.children].forEach((d,i)=>{d.classList.toggle('done',i<done);
    d.classList.toggle('run',!!run&&run.i===i);});
  if(!run)prog.style.width=0;
  if(notesOn){const N=scenes[si].notes||[];const i=run?run.i:Math.max(0,done-1);
    notesEl.innerHTML=(N[i]||'')+(done===0&&!run&&N[0]?'':'');
    if(done===0&&!run)notesEl.innerHTML=N[0]||'';}
  fitNotes();
}
/* notes rail (N): shrink the deck so the rail never covers content (REVIEW.md 2026-08-07, TIER 2) */
function fitNotes(){const dk=$('deck');dk.style.transform='';if(!notesOn)return;
  const nr=notesEl.getBoundingClientRect(),dr=dk.getBoundingClientRect();
  dk.style.transformOrigin='top center';
  dk.style.transform=`scale(${Math.max(0.5,Math.min(1,(nr.top-dr.top-8)/dr.height)).toFixed(3)})`;}
function beatF(b){(b.F||(()=>b.f(1)))();}
function beatZ(b){(b.z||(()=>b.f(0)))();}
function applyUpTo(n){
  run=null;const B=scenes[si].beats;n=clamp(n,0,B.length);
  for(let k=0;k<B.length;k++)k<n?beatF(B[k]):beatZ(B[k]);
  done=n;ui();
}
let glTimer=null;
function show(i,end){
  i=clamp(i,0,scenes.length-1);if(i===si)return;
  if(glTimer){clearTimeout(glTimer);glTimer=null;
    scenes.forEach(sc=>{sc.el.style.transition='';sc.el.style.transform='';sc.el.style.visibility='';
      if(sc!==scenes[si])sc.el.classList.remove('cur');});}
  const dir=i>si?1:-1;
  const oldS=scenes[si].el, newS=scenes[i].el;
  si=i;applyUpTo(end?scenes[si].beats.length:0);
  newS.classList.add('cur');
  if(RM){oldS.classList.remove('cur');return;}
  /* traveling-canvas glide: old exits one way, new enters from the other */
  oldS.style.visibility='visible';
  newS.style.transition='none';oldS.style.transition='none';
  newS.style.transform=`translateX(${dir*100}%)`;oldS.style.transform='translateX(0)';
  void newS.offsetWidth;
  const tr='transform .7s cubic-bezier(.4,0,.2,1)';
  newS.style.transition=tr;oldS.style.transition=tr;
  newS.style.transform='translateX(0)';oldS.style.transform=`translateX(${-dir*100}%)`;
  glTimer=setTimeout(()=>{glTimer=null;oldS.classList.remove('cur');
    oldS.style.transition='';oldS.style.transform='';oldS.style.visibility='';
    newS.style.transition='';newS.style.transform='';},720);
}
function next(){
  if(paused)setPaused(false);
  const B=scenes[si].beats;
  if(run){const b=B[run.i];b.f(1);beatF(b);done=run.i+1;run=null;ui();}
  else if(done<B.length){beatZ(B[done]);run={i:done,t:0};ui();}
  else if(si<scenes.length-1)show(si+1);
}
function prev(){
  if(run){const i=run.i;run=null;applyUpTo(i);}
  else if(done>0)applyUpTo(done-1);
  else if(si>0)show(si-1,true);
}
function setPaused(p){paused=p;$('paused').classList.toggle('show',p);
  if(auto){p?AUD.pause():AUD.play();}}
/* ---- narrated rehearsal mode: audio drives the beat schedule ---- */
const NARR={"sched": [{"t": 0.8, "s": 0, "b": 1, "dur": 24.05}, {"t": 26.25, "s": 0, "b": 2, "dur": 15.07}, {"t": 43.92, "s": 1, "b": 1, "dur": 13.47}, {"t": 58.8, "s": 1, "b": 2, "dur": 13.97}, {"t": 74.17, "s": 1, "b": 3, "dur": 10.38}, {"t": 85.95, "s": 1, "b": 4, "dur": 19.95}, {"t": 107.3, "s": 1, "b": 5, "dur": 13.2}, {"t": 123.1, "s": 2, "b": 1, "dur": 19.73}, {"t": 144.22, "s": 2, "b": 2, "dur": 11.75}, {"t": 157.38, "s": 2, "b": 3, "dur": 15.8}, {"t": 174.57, "s": 2, "b": 4, "dur": 10.1}, {"t": 187.27, "s": 3, "b": 1, "dur": 10.2}, {"t": 198.87, "s": 3, "b": 2, "dur": 12.15}, {"t": 212.42, "s": 3, "b": 3, "dur": 11.65}, {"t": 225.47, "s": 3, "b": 4, "dur": 15.05}, {"t": 243.12, "s": 4, "b": 1, "dur": 11.38}, {"t": 255.9, "s": 4, "b": 2, "dur": 21.62}, {"t": 278.93, "s": 4, "b": 3, "dur": 25.15}, {"t": 305.48, "s": 4, "b": 4, "dur": 11.47}, {"t": 318.35, "s": 4, "b": 5, "dur": 18.15}, {"t": 339.1, "s": 5, "b": 1, "dur": 14.5}, {"t": 355.0, "s": 5, "b": 2, "dur": 15.93}, {"t": 372.32, "s": 5, "b": 3, "dur": 13.72}, {"t": 387.45, "s": 5, "b": 4, "dur": 15.2}, {"t": 405.25, "s": 6, "b": 1, "dur": 16.38}, {"t": 423.02, "s": 6, "b": 2, "dur": 13.2}, {"t": 437.62, "s": 6, "b": 3, "dur": 14.28}, {"t": 453.3, "s": 6, "b": 4, "dur": 10.03}, {"t": 465.93, "s": 7, "b": 1, "dur": 18.18}, {"t": 485.5, "s": 7, "b": 2, "dur": 11.3}, {"t": 498.2, "s": 7, "b": 3, "dur": 14.47}, {"t": 514.08, "s": 7, "b": 4, "dur": 12.75}, {"t": 529.43, "s": 8, "b": 1, "dur": 25.8}, {"t": 556.63, "s": 8, "b": 2, "dur": 20.85}, {"t": 578.88, "s": 8, "b": 3, "dur": 9.38}, {"t": 589.65, "s": 8, "b": 4, "dur": 15.43}, {"t": 607.68, "s": 9, "b": 1, "dur": 12.53}, {"t": 621.6, "s": 9, "b": 2, "dur": 18.77}, {"t": 641.77, "s": 9, "b": 3, "dur": 17.82}, {"t": 661.0, "s": 9, "b": 4, "dur": 18.32}, {"t": 680.73, "s": 9, "b": 5, "dur": 8.07}, {"t": 691.4, "s": 10, "b": 1, "dur": 16.77}, {"t": 709.57, "s": 10, "b": 2, "dur": 9.68}, {"t": 720.65, "s": 10, "b": 3, "dur": 13.32}, {"t": 735.38, "s": 10, "b": 4, "dur": 13.43}, {"t": 751.4, "s": 11, "b": 1, "dur": 15.38}, {"t": 768.17, "s": 11, "b": 2, "dur": 23.38}, {"t": 792.95, "s": 11, "b": 3, "dur": 19.62}, {"t": 813.97, "s": 11, "b": 4, "dur": 18.5}, {"t": 835.07, "s": 12, "b": 1, "dur": 21.98}, {"t": 858.45, "s": 12, "b": 2, "dur": 10.82}, {"t": 870.67, "s": 12, "b": 3, "dur": 15.12}, {"t": 888.4, "s": 13, "b": 1, "dur": 14.1}, {"t": 903.9, "s": 13, "b": 2, "dur": 7.53}, {"t": 912.82, "s": 13, "b": 3, "dur": 11.4}, {"t": 925.62, "s": 13, "b": 4, "dur": 19.35}, {"t": 947.57, "s": 14, "b": 1, "dur": 11.18}], "total": 961.35};
let auto=false, evIdx=0;
const AUD=new Audio('assets/narration.mp3');
function setAuto(a){
  auto=a;$('auto').classList.toggle('show',a);
  if(a){paused=false;$('paused').classList.remove('show');
    evIdx=0;show(0);applyUpTo(0);AUD.currentTime=0;AUD.play();}
  else AUD.pause();
}
function loop(ts){
  if(window.__RENDER){requestAnimationFrame(loop);return;}
  if(last===null)last=ts;
  const dt=paused?0:Math.min(0.05,(ts-last)/1000);
  last=ts;tAmb+=dt;
  for(const o of ORB){if(o.si!==si)continue;
    o.el.setAttribute('cx',o.cx+o.r*Math.cos(o.sp*tAmb+o.ph));
    o.el.setAttribute('cy',o.cy+o.r*Math.sin(o.sp*tAmb+o.ph)*0.55);}
  if(auto&&!paused){
    const at=AUD.currentTime;
    while(evIdx<NARR.sched.length&&NARR.sched[evIdx].t<=at){
      const ev=NARR.sched[evIdx++];
      if(ev.s!==si)show(ev.s);
      let g=0;while(g++<8&&!(run&&run.i===ev.b-1)&&done<ev.b)next();
    }
    if(AUD.ended)setAuto(false);
  }
  if(run){
    const B=scenes[si].beats,b=B[run.i];
    run.t+=dt*(window.MOTION==='instant'?60:window.MOTION==='calm'?1.6:3);
    prog.style.width=(100*clamp(run.t/b.d))+'%';
    if(run.t>=b.d){b.f(1);beatF(b);done=run.i+1;run=null;ui();}
    else b.f(run.t/b.d);
  }
  requestAnimationFrame(loop);
}
scenes[0].el.classList.add('cur');
applyUpTo(0);
requestAnimationFrame(loop);
/* deterministic render: window.__frame(t) steps the world on a virtual clock.
   Call with monotonically increasing t (seconds); pairs with the narration schedule. */
window.__frame=function(vt){
  window.__RENDER=true;
  const dt=Math.max(0,vt-(window.__vt??0));window.__vt=vt;
  tAmb+=dt;
  for(const o of ORB){if(o.si!==si)continue;
    o.el.setAttribute('cx',o.cx+o.r*Math.cos(o.sp*tAmb+o.ph));
    o.el.setAttribute('cy',o.cy+o.r*Math.sin(o.sp*tAmb+o.ph)*0.55);}
  while(evIdx<NARR.sched.length&&NARR.sched[evIdx].t<=vt){
    const ev=NARR.sched[evIdx++];
    if(ev.s!==si)show(ev.s);
    let g=0;while(g++<8&&!(run&&run.i===ev.b-1)&&done<ev.b)next();
  }
  if(run){
    const B=scenes[si].beats,b=B[run.i];
    run.t+=dt*(window.MOTION==='instant'?60:window.MOTION==='calm'?1.6:3);
    prog.style.width=(100*clamp(run.t/b.d))+'%';
    if(run.t>=b.d){b.f(1);beatF(b);done=run.i+1;run=null;ui();}
    else b.f(run.t/b.d);
  }
};
addEventListener('keydown',ev=>{
  if(ev.target.closest&&ev.target.closest('#tweaks-root'))return;
  const k=ev.key;
  if(k==='a'||k==='A'){setAuto(!auto);return;}
  if(auto&&k!=='p'&&k!=='P'&&k!=='n'&&k!=='N'&&k!=='f'&&k!=='F')setAuto(false);
  if(k===' '||k==='ArrowRight'||k==='PageDown'){ev.preventDefault();next();}
  else if(k==='ArrowLeft'||k==='PageUp'){prev();}
  else if(k==='ArrowDown'){ev.preventDefault();show(si+1);}
  else if(k==='ArrowUp'){ev.preventDefault();show(si-1);}
  else if(k==='r'||k==='R'){setPaused(false);applyUpTo(0);}
  else if(k==='p'||k==='P'){setPaused(!paused);}
  else if(k==='n'||k==='N'){notesOn=!notesOn;notesEl.classList.toggle('show',notesOn);ui();}
  else if(k==='f'||k==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();}
  else if(k>='0'&&k<='9'){applyUpTo(+k);}
});
addEventListener('click',ev=>{if(ev.target.closest&&ev.target.closest('#tweaks-root'))return;if(auto){setAuto(false);return;}next();});
const stage=$('stage');
function fitStage(){const sc=Math.min(innerWidth/1600,innerHeight/900);
  stage.style.transform=`translate(-50%,-50%) scale(${sc})`;}
stage.style.top='50%';stage.style.left='50%';
addEventListener('resize',fitStage);fitStage();
/* ---------- tweaks hook ---------- */
window.__deckTweaks={apply(t){
  const de=document.documentElement;
  if(!t.theme||t.theme==='auto')de.removeAttribute('data-theme');else de.setAttribute('data-theme',t.theme);
  window.MOTION=t.motion||'dynamic';
  const a=t.accent||[];
  if(a[0]==='#6FD3C7'){['--teal','--amber','--rose'].forEach(v=>de.style.removeProperty(v));}
  else{de.style.setProperty('--teal',a[0]);de.style.setProperty('--amber',a[1]);de.style.setProperty('--rose',a[2]);}
}};
