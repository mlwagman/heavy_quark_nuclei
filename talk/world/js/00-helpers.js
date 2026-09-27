'use strict';
/* ---------- helpers ---------- */
const NS='http://www.w3.org/2000/svg';
function E(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);p.appendChild(e);return e;}
function D(t,c,p,h){const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;if(p)p.appendChild(e);return e;}
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const eo=t=>1-Math.pow(1-clamp(t),3);
const eio=t=>{t=clamp(t);return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;};
const back=t=>{t=clamp(t);const c=1.70158;return 1+(c+1)*Math.pow(t-1,3)+c*Math.pow(t-1,2);};
const seg=(p,a,b)=>clamp((p-a)/(b-a));
let RVN=0;const RVV=['right','left','up','pop','down','tilt'];
window.MOTION='dynamic';
function rv(el,p){
  if(window.MOTION==='calm'){p=clamp(p);el.style.opacity=p;el.style.transform=`translateY(${10*(1-p)}px)`;return;}
  if(!el._v)el._v=RVV[(RVN++)%RVV.length];
  const t=eo(clamp(p));el.style.opacity=Math.min(1,t*2.5);
  switch(el._v){
    case 'left':el.style.transform=`translateX(${-140*(1-t)}px)`;break;
    case 'up':el.style.transform=`translateY(${80*(1-t)}px)`;break;
    case 'down':el.style.transform=`translateY(${-60*(1-t)}px)`;break;
    case 'pop':el.style.transform=`scale(${Math.max(.001,.55+.45*back(p))})`;break;
    case 'tilt':el.style.transform=`translateX(${110*(1-t)}px) rotate(${2.5*(1-t)}deg)`;break;
    default:el.style.transform=`translateX(${140*(1-t)}px)`;
  }
}
function op(e,v){v=clamp(v);const t=eo(v),o=Math.min(1,v*2);
  if(e.namespaceURI===NS){e.setAttribute('opacity',o);
    const d=e.firstChild&&e.firstChild.classList&&e.firstChild.classList.contains('mlab')?e.firstChild:null;
    if(d)d.style.transform=`translateX(${48*(1-t)}px)`;
  }else{e.style.opacity=o;e.style.transform=`translateX(${70*(1-t)}px)`;}}
function dl(p){const L=p.getTotalLength();p.style.strokeDasharray=L;p.style.strokeDashoffset=L;return L;}
function ds(p,t){const L=parseFloat(p.style.strokeDasharray||1);p.style.strokeDashoffset=L*(1-clamp(t));}
function axes(g,x0,y0,x1,y1){ /* y0 = bottom */
  E('line',{x1:x0,y1:y0,x2:x1,y2:y0,stroke:'var(--muted)','stroke-width':1.5},g);
  E('line',{x1:x0,y1:y0,x2:x0,y2:y1,stroke:'var(--muted)','stroke-width':1.5},g);
}
function bandPath(X,Y,xs,lo,hi){
  let d='M'+X(xs[0]).toFixed(1)+' '+Y(hi[0]).toFixed(1);
  for(let i=1;i<xs.length;i++)d+='L'+X(xs[i]).toFixed(1)+' '+Y(hi[i]).toFixed(1);
  for(let i=xs.length-1;i>=0;i--)d+='L'+X(xs[i]).toFixed(1)+' '+Y(lo[i]).toFixed(1);
  return d+'Z';
}
function mlab(g,x,y,html,o){ /* math label in deck typography, foreignObject anchor */
  o=o||{};const size=o.size||17,anchor=o.anchor||'start',W=o.w||460,H=Math.round(size*3);
  const fx=anchor==='middle'?x-W/2:anchor==='end'?x-W:x;
  const fo=E('foreignObject',{x:fx,y:y-H/2,width:W,height:H,style:'overflow:visible'},g);
  const d=document.createElement('div');d.className='mlab';
  d.style.cssText=`width:${W}px;height:${H}px;display:flex;align-items:center;justify-content:${anchor==='middle'?'center':anchor==='end'?'flex-end':'flex-start'};font-size:${size}px;color:${o.color||'var(--muted)'}`;
  d.innerHTML=html;fo.appendChild(d);
  return fo;
}
const ORB=[]; /* ambient orbiters {si,el,cx,cy,r,sp,ph} */
function orbDot(g,si,cx,cy,r,sp,ph,col,rad){
  const e=E('circle',{cx:cx+r,cy:cy,r:rad,fill:col},g);
  ORB.push({si,el:e,cx,cy,r,sp,ph});
  return e;
}
const deck=document.getElementById('deck');
function scene(eyebrow,title){
  const s=D('section','scene',deck);
  D('div','eyebrow',s,eyebrow);
  if(title)D('h1','',s,title);
  const b=D('div','sbody',s);
  return {s,b};
}
const scenes=[];

