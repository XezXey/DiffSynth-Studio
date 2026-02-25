"""
viz3d_utils.py
==============
Self-contained HTML skeleton / image-overlay visualiser.

Architecture
------------
All panels live in one page (no iframes). Each ROW has its own shared playback
bar that drives ALL panels in that row in sync.

Panel types
-----------
  panel_3d(pred, gt=None, ...)
      Three.js scene with OrbitControls, checkerboard + mirror floor, VSM
      shadows, multiple overlaid skeletons — mirrors main.js style.

  panel_2d(pred, gt=None, ...)
      Canvas 2-D skeleton viewer (auto-normalised).

  panel_image_overlay(images, joints=None, joints_gt=None, ...)
      Canvas: images drawn as background with 2-D skeleton overlaid.
      Ideal for showing input video frames with predicted / GT keypoints.

Quick-start
-----------
    from viz3d_utils import generate_html, panel_3d, panel_2d, panel_image_overlay

    generate_html(
        rows=[
            {
                "label":  "sample 0",
                "panels": [
                    panel_3d(pred=pred_3d, gt=gt_3d),
                    panel_image_overlay(images=frames, joints=pred_2d),
                ],
            }
        ],
        output_path="viewer.html",
        joint_set="smpl22",   # "smpl22" | "coco17" | "custom"
    )

    # shapes:  pred_3d / gt_3d: (T, J, 3)
    #          pred_2d:         (T, J, 2)
    #          frames: list[str path | np.ndarray (H,W,3) | PIL.Image]

One-liner
---------
    from viz3d_utils import visualize_sample
    visualize_sample(pred=pred_3d, gt=gt_3d, images=frames, joints_2d=pred_2d)
"""

from __future__ import annotations
import base64, json, math
import numpy as np
from io import BytesIO
from pathlib import Path
from typing import Optional, Any

# ── Skeleton topology ─────────────────────────────────────────────────────────
SMPL22_EDGES = [
    [0,1],[1,4],[4,7],[7,10],
    [0,2],[2,5],[5,8],[8,11],
    [0,3],[3,6],[6,9],[9,12],[12,15],
    [12,13],[13,16],[16,18],[18,20],
    [12,14],[14,17],[17,19],[19,21],
]
COCO17_EDGES = [
    [0,1],[0,2],[1,3],[2,4],[5,6],
    [5,7],[7,9],[6,8],[8,10],
    [5,11],[6,12],[11,12],
    [11,13],[13,15],[12,14],[14,16],
]
_JOINT_SETS = {"smpl22": SMPL22_EDGES, "coco17": COCO17_EDGES}

PRED_COLOR = "#4ECDC4"
GT_COLOR   = "#FF6B6B"

# ── Image encoding ────────────────────────────────────────────────────────────
def _encode_image(img: Any, quality: int = 80) -> str:
    """Encode a file-path / numpy array / PIL Image → base64 JPEG data-URL."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        PILImage = None
    if isinstance(img, (str, Path)):
        if PILImage is None: raise ImportError("pip install Pillow")
        pil = PILImage.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        if PILImage is None: raise ImportError("pip install Pillow")
        pil = PILImage.fromarray(img.astype(np.uint8))
    elif PILImage and isinstance(img, PILImage.Image):
        pil = img.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def _encode_images(images: list, quality: int = 80) -> list:
    n = len(images)
    if n > 20:
        print(f"  Encoding {n} images (quality={quality}) ...", end="", flush=True)
    out = [_encode_image(img, quality) for img in images]
    if n > 20:
        print(f" {sum(len(s) for s in out)//1024} KB")
    return out

# ── Panel builders ────────────────────────────────────────────────────────────
def panel_3d(
    pred: np.ndarray,
    gt:   Optional[np.ndarray] = None,
    extra_skeletons: Optional[list] = None,
    pred_color: str = PRED_COLOR,
    gt_color:   str = GT_COLOR,
    label: str = "",
) -> dict:
    """
    3-D Three.js skeleton panel.

    Parameters
    ----------
    pred            : (T, J, 3)  predicted skeleton
    gt              : (T, J, 3)  ground-truth skeleton  (optional)
    extra_skeletons : list of {"joints":(T,J,3), "color":"#hex", "label":str}
    pred_color, gt_color : hex tints
    label           : cell badge text
    """
    skels = [{"joints": np.asarray(pred).tolist(), "color": pred_color, "label": "pred"}]
    if gt is not None:
        skels.append({"joints": np.asarray(gt).tolist(), "color": gt_color, "label": "gt"})
    if extra_skeletons:
        for s in extra_skeletons:
            skels.append({"joints": np.asarray(s["joints"]).tolist(),
                          "color": s.get("color","#aaaaaa"), "label": s.get("label","")})
    return {"type": "3d", "skeletons": skels, "label": label}

def panel_2d(
    pred: np.ndarray,
    gt:   Optional[np.ndarray] = None,
    extra_skeletons: Optional[list] = None,
    pred_color: str = PRED_COLOR,
    gt_color:   str = GT_COLOR,
    label: str = "",
) -> dict:
    """
    2-D canvas skeleton panel.

    Parameters
    ----------
    pred : (T, J, 2)  predicted 2-D keypoints
    gt   : (T, J, 2)  ground-truth              (optional)
    """
    skels = [{"joints": np.asarray(pred).tolist(), "color": pred_color, "label": "pred"}]
    if gt is not None:
        skels.append({"joints": np.asarray(gt).tolist(), "color": gt_color, "label": "gt"})
    if extra_skeletons:
        for s in extra_skeletons:
            skels.append({"joints": np.asarray(s["joints"]).tolist(),
                          "color": s.get("color","#aaaaaa"), "label": s.get("label","")})
    return {"type": "2d", "skeletons": skels, "label": label}

def panel_image_overlay(
    images:        list,
    joints:        Optional[np.ndarray] = None,
    joints_gt:     Optional[np.ndarray] = None,
    joint_color:   str = PRED_COLOR,
    gt_color:      str = GT_COLOR,
    label:         str = "",
    image_quality: int = 75,
) -> dict:
    """
    Image + 2-D skeleton overlay panel.

    Parameters
    ----------
    images       : list of T items — str path | np.ndarray (H,W,3) | PIL Image
    joints       : (T, J, 2)  skeleton drawn on top  (optional)
    joints_gt    : (T, J, 2)  gt overlay             (optional)
    joint_color  : hex colour for pred skeleton
    gt_color     : hex colour for gt skeleton
    label        : cell badge text
    image_quality: JPEG quality 1-95  (lower → smaller file)
    """
    encoded = _encode_images(images, quality=image_quality)
    skels   = []
    if joints is not None:
        skels.append({"joints": np.asarray(joints).tolist(), "color": joint_color, "label": "pred"})
    if joints_gt is not None:
        skels.append({"joints": np.asarray(joints_gt).tolist(), "color": gt_color, "label": "gt"})
    return {"type": "image_overlay", "images": encoded, "skeletons": skels, "label": label}


# ── Embedded CSS ─────────────────────────────────────────────────────────────
_CSS = """\
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
:root {
  --bg:#07070d; --surface:#0d0d1a; --border:rgba(255,255,255,0.07);
  --accent:#4ECDC4; --accent2:#FF6B6B; --text:#e0e0ed; --muted:#525272;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; min-height:100vh; }
header {
  padding:18px 28px 12px; border-bottom:1px solid var(--border);
  display:flex; align-items:baseline; gap:14px;
}
header h1 {
  font-family:'Space Mono',monospace; font-size:15px; font-weight:700;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
header span { font-size:11px; color:var(--muted); font-family:'Space Mono',monospace; }
.legend { display:flex; gap:18px; padding:7px 28px; background:var(--surface); border-bottom:1px solid var(--border); }
.legend-item { display:flex; align-items:center; gap:6px; font-size:11px; font-family:'Space Mono',monospace; }
.dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.sample-row { border-bottom:1px solid var(--border); }
.sample-row:last-child { border-bottom:none; }
.row-header {
  display:flex; align-items:center; gap:12px; padding:5px 14px;
  background:var(--surface); border-bottom:1px solid var(--border);
}
.row-label {
  font-family:'Space Mono',monospace; font-size:10px; font-weight:700;
  color:var(--muted); background:rgba(0,0,0,0.4); border:1px solid var(--border);
  padding:2px 8px; border-radius:10px; white-space:nowrap;
}
.row-playbar { display:flex; align-items:center; gap:8px; flex:1; }
.row-playbar button {
  background:none; border:none; color:#bbb; cursor:pointer; font-size:13px;
  padding:2px 5px; border-radius:4px; transition:color .2s,background .2s;
}
.row-playbar button:hover { color:#fff; background:rgba(255,255,255,.1); }
.row-playbar button.active { color:var(--accent); }
.row-playbar input[type=range] {
  -webkit-appearance:none; flex:1; max-width:260px; height:3px;
  background:rgba(255,255,255,.15); border-radius:2px; outline:none; cursor:pointer;
}
.row-playbar input[type=range]::-webkit-slider-thumb {
  -webkit-appearance:none; width:11px; height:11px;
  border-radius:50%; background:var(--accent); cursor:pointer;
}
.row-playbar .frame-lbl {
  font-family:'Space Mono',monospace; font-size:10px; color:var(--muted);
  min-width:58px; text-align:right;
}
.panels-grid { display:grid; }
.panel-cell { position:relative; border-right:1px solid var(--border); overflow:hidden; }
.panel-cell:last-child { border-right:none; }
.panel-cell > canvas, .panel-cell > div > canvas { display:block; width:100%!important; height:100%!important; }
.panel-badge {
  position:absolute; top:7px; left:9px; z-index:30;
  font-family:'Space Mono',monospace; font-size:9px; font-weight:700;
  background:rgba(0,0,0,.55); backdrop-filter:blur(4px);
  border:1px solid var(--border); padding:2px 7px; border-radius:9px;
  color:var(--muted); pointer-events:none; user-select:none;
}
.skel-legends {
  position:absolute; bottom:8px; left:8px; z-index:30;
  display:flex; gap:5px; flex-wrap:wrap; pointer-events:none;
}
.skel-badge {
  font-family:'Space Mono',monospace; font-size:9px; font-weight:700;
  padding:2px 7px; border-radius:9px; backdrop-filter:blur(4px);
}
"""

# ── Embedded JavaScript ───────────────────────────────────────────────────────
# Uses __EDGES__, __ALL_ROWS__, __CELL_HEIGHT__ placeholders replaced at render time.
_JS = """\
(function(){
'use strict';
const EDGES=__EDGES__;
const ALL_ROWS=__ALL_ROWS__;
const CELL_HEIGHT=__CELL_HEIGHT__;

// ── Normalise 3-D skeletons (shared scale for pred+gt) ───────────────────────
function normSkels3D(sl){
  var mn=[1e9,1e9,1e9],mx=[-1e9,-1e9,-1e9];
  for(var _i=0;_i<sl.length;_i++){var s=sl[_i];
    for(var fi=0;fi<s.joints.length;fi++){var fr=s.joints[fi];
      for(var ji=0;ji<fr.length;ji++){var jt=fr[ji];
        for(var k=0;k<3;k++){if(jt[k]<mn[k])mn[k]=jt[k];if(jt[k]>mx[k])mx[k]=jt[k];}
      }}}
  var range=Math.max(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2])||1;
  var ctr=[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2];
  var yL=(mx[1]-mn[1])/(2*range);
  return sl.map(function(s){return Object.assign({},s,{normed:s.joints.map(function(fr){
    return fr.map(function(jt){return [(jt[0]-ctr[0])/range,(jt[1]-ctr[1])/range+yL,(jt[2]-ctr[2])/range];});})});});
}

// ── Normalise 2-D skeletons to canvas space ──────────────────────────────────
function normSkels2D(sl,W,H){
  var pad=0.1,x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
  for(var _i=0;_i<sl.length;_i++){var s=sl[_i];
    for(var fi=0;fi<s.joints.length;fi++){var fr=s.joints[fi];
      for(var ji=0;ji<fr.length;ji++){var jt=fr[ji];
        if(jt[0]<x0)x0=jt[0];if(jt[0]>x1)x1=jt[0];
        if(jt[1]<y0)y0=jt[1];if(jt[1]>y1)y1=jt[1];
      }}}
  var rx=x1-x0||1,ry=y1-y0||1;
  var sc=Math.min(W*(1-2*pad)/rx,H*(1-2*pad)/ry);
  var cx=(x0+x1)/2,cy=(y0+y1)/2;
  return sl.map(function(s){return Object.assign({},s,{normed:s.joints.map(function(fr){
    return fr.map(function(jt){return [W/2+(jt[0]-cx)*sc,H/2+(jt[1]-cy)*sc];});})});});
}

// ── Three.js floor: checkerboard + Reflector ─────────────────────────────────
function buildFloor(scene){
  var P=0.5,N=Math.ceil(10/P);
  var c2=document.createElement('canvas').getContext('2d');
  c2.canvas.width=c2.canvas.height=2;
  c2.fillStyle='#2e2e4a';c2.fillRect(0,0,2,2);
  c2.fillStyle='#1e1e34';c2.fillRect(0,1,1,1);
  var tex=new THREE.CanvasTexture(c2.canvas); tex.magFilter=THREE.NearestFilter;
  var g=new THREE.PlaneGeometry(N*P,N*P,N,N).toNonIndexed(); g.rotateX(-Math.PI/2);
  var uv=g.attributes.uv,cnt=0,fl=0;
  for(var i=0;i<uv.count;i++){if(i>0&&i%6===0){cnt++;if(cnt%N===0)fl=1-fl;}uv.setXY(i,(cnt+fl)%2,(cnt+fl)%2);}
  var mat=new THREE.MeshPhongMaterial({color:0xffffff,map:tex,opacity:0.85,transparent:true});
  var mesh=new THREE.Mesh(g,mat); mesh.receiveShadow=true; scene.add(mesh);
  if(THREE.Reflector){
    var mg=new THREE.PlaneGeometry(N*P,N*P);
    var mir=new THREE.Reflector(mg,{clipBias:0.003,textureWidth:512,textureHeight:512,color:0x222244});
    mir.rotateX(-Math.PI/2); mir.position.y=-0.001; scene.add(mir);
  }
}

// ── Lights (matching main.js) ─────────────────────────────────────────────────
function buildLights(scene){
  scene.add(new THREE.AmbientLight(0xffffff,0.45));
  var d=new THREE.DirectionalLight(0xffffff,0.9); d.position.set(2,4,3); d.castShadow=true;
  d.shadow.radius=1.5; d.shadow.blurSamples=12; d.shadow.bias=-0.002;
  d.shadow.mapSize.width=d.shadow.mapSize.height=1024;
  var sc=d.shadow.camera; sc.left=sc.bottom=-6; sc.right=sc.top=6; sc.near=0.5; sc.far=30;
  scene.add(d);
  var pt=new THREE.PointLight(0xffffff,0.3); pt.position.set(4,8,4); scene.add(pt);
}

// ── 2-D drawing helper ────────────────────────────────────────────────────────
function draw2D(ctx,W,H,items,edges){
  ctx.clearRect(0,0,W,H);
  for(var _i=0;_i<items.length;_i++){
    var s=items[_i];
    ctx.strokeStyle=s.color+'88'; ctx.lineWidth=2;
    for(var e=0;e<edges.length;e++){
      var a=edges[e][0],b=edges[e][1];
      ctx.beginPath(); ctx.moveTo(s.pts[a][0],s.pts[a][1]); ctx.lineTo(s.pts[b][0],s.pts[b][1]); ctx.stroke();
    }
    ctx.shadowBlur=7;
    for(var j=0;j<s.pts.length;j++){
      ctx.beginPath(); ctx.arc(s.pts[j][0],s.pts[j][1],5,0,Math.PI*2);
      ctx.fillStyle=s.color; ctx.shadowColor=s.color; ctx.fill();
    }
    ctx.shadowBlur=0;
  }
}

// ── 3-D panel ─────────────────────────────────────────────────────────────────
function init3DPanel(cell,pd){
  var W=cell.clientWidth||400,H=cell.clientHeight||CELL_HEIGHT;
  var rdr=new THREE.WebGLRenderer({antialias:true});
  rdr.setPixelRatio(Math.min(devicePixelRatio,2));
  rdr.setSize(W,H); rdr.shadowMap.enabled=true; rdr.shadowMap.type=THREE.VSMShadowMap;
  rdr.setClearColor(0x1a1a2e); cell.appendChild(rdr.domElement);
  var scene=new THREE.Scene();
  var cam=new THREE.PerspectiveCamera(50,W/H,0.01,200); cam.position.set(0,1.6,3.5);
  var orb=new THREE.OrbitControls(cam,rdr.domElement);
  orb.target.set(0,0.5,0); orb.enableDamping=true; orb.dampingFactor=0.08;
  orb.minDistance=0.5; orb.maxDistance=20; orb.update();
  buildLights(scene); buildFloor(scene); scene.add(new THREE.AxesHelper(0.35));
  var edges=EDGES;
  var ns=normSkels3D(pd.skeletons); var J=ns[0].normed[0].length;
  var objs=ns.map(function(sk){
    var col=new THREE.Color(sk.color);
    var jm=new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:0.5});
    var jg=new THREE.SphereGeometry(0.022,10,10);
    var jts=[];
    for(var i=0;i<J;i++){var m=new THREE.Mesh(jg,jm);m.castShadow=true;scene.add(m);jts.push(m);}
    var bm=new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.8});
    var bns=edges.map(function(){
      var g=new THREE.BufferGeometry();
      g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(6),3));
      var l=new THREE.Line(g,bm); scene.add(l); return l;
    });
    return {jts:jts,bns:bns,normed:sk.normed};
  });
  function loop(){requestAnimationFrame(loop);orb.update();rdr.render(scene,cam);}
  loop();
  function setFrame(f){
    for(var oi=0;oi<objs.length;oi++){
      var o=objs[oi],pts=o.normed[f];
      for(var j=0;j<J;j++) o.jts[j].position.set(pts[j][0],pts[j][1],pts[j][2]);
      for(var e=0;e<edges.length;e++){
        var ab=edges[e],pos=o.bns[e].geometry.attributes.position;
        pos.setXYZ(0,pts[ab[0]][0],pts[ab[0]][1],pts[ab[0]][2]);
        pos.setXYZ(1,pts[ab[1]][0],pts[ab[1]][1],pts[ab[1]][2]);
        pos.needsUpdate=true;
      }
    }
  }
  function resize(w,h){rdr.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix();}
  return {setFrame:setFrame,resize:resize};
}

// ── 2-D panel ─────────────────────────────────────────────────────────────────
function init2DPanel(cell,pd){
  var cv=document.createElement('canvas'); cell.appendChild(cv);
  var W=cell.clientWidth||400,H=cell.clientHeight||CELL_HEIGHT;
  cv.width=W; cv.height=H;
  var ctx=cv.getContext('2d');
  var ns=normSkels2D(pd.skeletons,W,H);
  function setFrame(f){
    var items=ns.map(function(s){return{color:s.color,pts:s.normed[f]};});
    draw2D(ctx,cv.width,cv.height,items,EDGES);
  }
  function resize(w,h){cv.width=w;cv.height=h;ns=normSkels2D(pd.skeletons,w,h);}
  return {setFrame:setFrame,resize:resize};
}

// ── Image overlay panel ───────────────────────────────────────────────────────
function initImageOverlayPanel(cell,pd){
  var cv=document.createElement('canvas'); cell.appendChild(cv);
  var W=cell.clientWidth||400,H=cell.clientHeight||CELL_HEIGHT;
  cv.width=W; cv.height=H;
  var ctx=cv.getContext('2d');
  var T=pd.images.length;
  var imgs=pd.images.map(function(src){var i=new Image();i.src=src;return i;});
  var ns=pd.skeletons.length?normSkels2D(pd.skeletons,W,H):[];
  function draw(f){
    var cW=cv.width,cH=cv.height; ctx.clearRect(0,0,cW,cH);
    var img=imgs[f%T];
    if(img.complete&&img.naturalWidth){
      var sc=Math.min(cW/img.naturalWidth,cH/img.naturalHeight);
      var dw=img.naturalWidth*sc,dh=img.naturalHeight*sc;
      ctx.drawImage(img,(cW-dw)/2,(cH-dh)/2,dw,dh);
    } else { img.onload=function(){draw(f);}; ctx.fillStyle='#1a1a2e'; ctx.fillRect(0,0,cW,cH); }
    if(ns.length){
      var items=ns.map(function(s){return{color:s.color,pts:s.normed[f]};});
      draw2D(ctx,cW,cH,items,EDGES);
    }
  }
  function setFrame(f){draw(f);}
  function resize(w,h){
    cv.width=w;cv.height=h;
    if(pd.skeletons.length)ns=normSkels2D(pd.skeletons,w,h);
  }
  return {setFrame:setFrame,resize:resize};
}

// ── Per-row shared playback controller ───────────────────────────────────────
function initRowController(ri,T,panels){
  var bar=document.getElementById('playbar-r'+ri); if(!bar)return;
  var btnPrev=bar.querySelector('.btn-prev'),btnPlay=bar.querySelector('.btn-play'),
      btnNext=bar.querySelector('.btn-next'),btnLoop=bar.querySelector('.btn-loop'),
      slider=bar.querySelector('.slider'),lbl=bar.querySelector('.frame-lbl');
  slider.max=T-1;
  var cur=0,playing=false,looping=true;
  function goTo(f){
    cur=((f%T)+T)%T; slider.value=cur; lbl.textContent=(cur+1)+' / '+T;
    for(var pi=0;pi<panels.length;pi++) panels[pi].setFrame(cur);
  }
  goTo(0);
  var FPS=25,STEP=1/FPS,acc=0,prev=performance.now()/1000,wasP=false;
  function tick(now){
    requestAnimationFrame(tick);
    var t=now/1000,dt=Math.min(t-prev,0.1); prev=t;
    if(playing){
      if(!wasP){acc=0;wasP=true;}
      acc+=dt;
      while(acc>=STEP){
        var n=cur+1;
        if(n>=T&&!looping){playing=false;btnPlay.innerHTML='&#9654;';}else goTo(n);
        acc-=STEP;
      }
    } else wasP=false;
  }
  requestAnimationFrame(tick);
  btnPlay.addEventListener('click',function(){playing=!playing;btnPlay.innerHTML=playing?'&#9646;&#9646;':'&#9654;';});
  btnPrev.addEventListener('click',function(){goTo(cur-1);});
  btnNext.addEventListener('click',function(){goTo(cur+1);});
  btnLoop.addEventListener('click',function(){looping=!looping;btnLoop.classList.toggle('active',looping);});
  slider.addEventListener('input',function(){goTo(parseInt(slider.value));});
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
function bootstrap(){
  for(var ri=0;ri<ALL_ROWS.length;ri++){
    (function(rowIdx){
      var row=ALL_ROWS[rowIdx];
      var panels=[];
      for(var ci=0;ci<row.panels.length;ci++){
        var pd=row.panels[ci];
        var cell=document.getElementById('cell-r'+rowIdx+'-c'+ci);
        if(pd.type==='3d')            panels.push(init3DPanel(cell,pd));
        else if(pd.type==='2d')            panels.push(init2DPanel(cell,pd));
        else if(pd.type==='image_overlay') panels.push(initImageOverlayPanel(cell,pd));
        else panels.push({setFrame:function(){},resize:function(){}});
      }
      initRowController(rowIdx,row.T,panels);
    })(ri);
  }
}
if(document.readyState==='loading')
  document.addEventListener('DOMContentLoaded',bootstrap);
else
  bootstrap();
})();
"""


# ── HTML page template ────────────────────────────────────────────────────────
_THREEJS_CDN = (
    "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/objects/Reflector.js",
)

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<script src=\"{cdn0}\" defer></script>
<script src=\"{cdn1}\" defer></script>
<script src=\"{cdn2}\" defer></script>
<style>
{css}
</style>
</head>
<body>
<header>
  <h1>&#11835; {title}</h1>
  <span>{meta}</span>
</header>
{legend_html}
<main id="main-content">
{rows_html}
</main>
<script>
const __EDGES__={edges_json};
const __ALL_ROWS__={rows_json};
const __CELL_HEIGHT__={cell_height};
</script>
<script>
{js}
</script>
</body>
</html>
"""

def _playbar_html(row_index: int, T: int) -> str:
    """Return the playback bar HTML for a single row."""
    return (
        f'''<div class="row-header">
  <div class="row-label" id="label-r{row_index}"></div>
  <div class="row-playbar" id="playbar-r{row_index}">
    <button class="btn-prev" title="Previous frame">&#9664;</button>
    <button class="btn-play" title="Play/Pause">&#9654;</button>
    <button class="btn-next" title="Next frame">&#9654;&#9654;</button>
    <input type="range" class="slider" min="0" max="{T-1}" value="0"/>
    <span class="frame-lbl">1 / {T}</span>
    <button class="btn-loop active" title="Loop">&#9854;</button>
  </div>
</div>'''
    )


def _skel_badges_html(skeletons: list[dict]) -> str:
    """Return legend badge HTML for skeleton list inside a panel."""
    parts = []
    for s in skeletons:
        lbl = s.get("label", "")
        col = s.get("color", "#888")
        if lbl:
            parts.append(
                f'''<span class="skel-badge" style="background:{col}22;border:1px solid {col}66;color:{col}">'''
                + lbl + "</span>"
            )
    return '''<div class="skel-legends">'''  + "".join(parts) + "</div>" if parts else ""


def generate_html(
    rows: list[dict],
    output_path: str = "skeleton_viewer.html",
    joint_set: str = "smpl22",
    title: str = "Skeleton Viewer",
    cell_height: int = 360,
) -> str:
    """Render all rows into a self-contained HTML file and return the HTML string.

    Each entry in *rows* is::

        {
            "label": "Sample 1",   # shown in header
            "T": 80,               # number of frames
            "panels": [panel_3d(...), panel_2d(...), ...]
        }
    """
    from datetime import datetime as _dt

    edges = _JOINT_SETS.get(joint_set, SMPL22_EDGES)
    meta = f"{len(rows)} sample(s) · {_dt.now().strftime('%Y-%m-%d %H:%M')}"

    # ── legend (unique colours present in the file) ──────────────────────────
    seen: dict[str, str] = {}
    for row in rows:
        for panel in row["panels"]:
            for sk in panel.get("skeletons", []):
                if sk.get("label") and sk["color"] not in seen:
                    seen[sk["color"]] = sk["label"]
    legend_items = "".join(
        f'''<div class="legend-item"><div class="dot" style="background:{c}"></div>{l}</div>'''
        for c, l in seen.items()
    )
    legend_html = f'''<div class="legend">{legend_items}</div>''' if legend_items else ""

    # ── per-row HTML ──────────────────────────────────────────────────────────
    rows_html_parts: list[str] = []
    for ri, row in enumerate(rows):
        T = row["T"]
        ncols = len(row["panels"])
        playbar = _playbar_html(ri, T)
        # grid cells
        cells_html = ""
        for ci, panel in enumerate(row["panels"]):
            badge = f'''<div class="panel-badge">{panel.get("label","")}</div>'''
            badges = _skel_badges_html(panel.get("skeletons", []))
            cells_html += (
                f'''<div class="panel-cell" id="cell-r{ri}-c{ci}">{badge}{badges}</div>'''
            )
        rows_html_parts.append(
            f'''<div class="sample-row">
{playbar}
<div class="panels-grid" style="grid-template-columns:repeat({ncols},1fr);height:{cell_height}px;">
{cells_html}
</div>
</div>'''
        )
    rows_html = "\n".join(rows_html_parts)

    # ── strip heavy image data from rows_json payload (keep as-is — browsers handle it) ──
    import json as _json
    edges_json = _json.dumps(edges)
    rows_json  = _json.dumps(rows)

    html = _PAGE.format(
        title=title,
        css=_CSS,
        meta=meta,
        legend_html=legend_html,
        rows_html=rows_html,
        edges_json=edges_json,
        rows_json=rows_json,
        cell_height=cell_height,
        js=_JS,
        cdn0=_THREEJS_CDN[0],
        cdn1=_THREEJS_CDN[1],
        cdn2=_THREEJS_CDN[2],
    )
    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
    return html


def visualize_sample(
    pred: "np.ndarray",
    gt: "Optional[np.ndarray]" = None,
    images: "Optional[list]" = None,
    joints_2d: "Optional[np.ndarray]" = None,
    joints_2d_gt: "Optional[np.ndarray]" = None,
    label: str = "Sample",
    output_path: str = "out.html",
    joint_set: str = "smpl22",
    pred_color: str = PRED_COLOR,
    gt_color: str = GT_COLOR,
    cell_height: int = 360,
) -> str:
    """Convenience wrapper: build a single-row HTML for one sample.

    Automatically adds panels in this order:
      1. 3-D view  (always)
      2. 2-D view  (if *gt* is given)
      3. Image overlay  (if *images* is given)
    """
    panels_list: list[dict] = []
    panels_list.append(panel_3d(pred=pred, gt=gt, pred_color=pred_color,
                                 gt_color=gt_color, label="3D"))
    if gt is not None:
        panels_list.append(panel_2d(pred=pred, gt=gt, pred_color=pred_color,
                                     gt_color=gt_color, label="2D"))
    if images is not None:
        panels_list.append(panel_image_overlay(
            images=images,
            joints=joints_2d,
            joints_gt=joints_2d_gt,
            joint_color=pred_color,
            gt_color=gt_color,
            label="Overlay",
        ))
    T = len(pred)
    rows = [{"label": label, "T": T, "panels": panels_list}]
    return generate_html(rows, output_path=output_path, joint_set=joint_set,
                         title=label, cell_height=cell_height)


# ── Demo helpers ─────────────────────────────────────────────────────────────

def _demo_skeleton(T: int = 80, n_joints: int = 22, seed: int = 0) -> "np.ndarray":
    """Return a synthetic (T, J, 3) walk-like skeleton."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_joints, 3)) * 0.3
    base[:, 1] += np.linspace(0, 1, n_joints)  # rough vertical spread
    t = np.linspace(0, 2 * math.pi, T)
    traj = np.stack([np.sin(t) * 0.5, np.zeros(T), np.cos(t) * 0.1], axis=1)
    noise = rng.standard_normal((T, n_joints, 3)) * 0.02
    skel = base[None] + traj[:, None] + noise
    return skel.astype(np.float32)


def _demo_images(T: int = 80, W: int = 320, H: int = 240) -> list:
    """Return a list of T synthetic PIL Images (colour gradient per frame)."""
    try:
        from PIL import Image as _PILImage
        frames = []
        for i in range(T):
            r = int(30 + 100 * i / T)
            g = int(10 + 60 * i / T)
            frames.append(_PILImage.new("RGB", (W, H), (r, g, 80)))
        return frames
    except ImportError:
        return []


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="viz3d_utils demo")
    ap.add_argument("--out", default="./viz3d_demo.html")
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--cell_height", type=int, default=720)
    args = ap.parse_args()

    print("[demo] generating skeleton data …")
    pred1 = _demo_skeleton(T=args.T, seed=0)
    gt1   = _demo_skeleton(T=args.T, seed=1)
    pred2 = _demo_skeleton(T=args.T, seed=2)

    print("[demo] generating image data …")
    imgs = _demo_images(T=args.T)

    # 3 rows
    rows = [
        {
            "label": "Sample A – 3D + Image Overlay",
            "T": args.T,
            "panels": [
                panel_3d(pred=pred1, gt=gt1, label="3D"),
                panel_image_overlay(images=imgs, joints=pred1[:, :, :2],
                                    joints_gt=gt1[:, :, :2], label="Overlay"),
            ],
        },
        {
            "label": "Sample B – 3D + 2D",
            "T": args.T,
            "panels": [
                panel_3d(pred=pred2, label="3D only"),
                panel_2d(pred=pred2, gt=gt1, label="2D"),
            ],
        },
        {
            "label": "Sample C – three panels",
            "T": args.T,
            "panels": [
                panel_3d(pred=pred1, gt=gt1, label="3D"),
                panel_2d(pred=pred1, gt=gt1, label="2D"),
                panel_image_overlay(images=imgs, joints=pred1[:, :, :2], label="Overlay"),
            ],
        },
    ]

    path = generate_html(rows, output_path=args.out, title="viz3d demo",
                         cell_height=args.cell_height)
    sz = Path(args.out).stat().st_size / 1024
    print(f"[demo] written {sz:.0f} KB → {args.out}")
