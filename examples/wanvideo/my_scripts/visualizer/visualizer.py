"""
visualize_joints.py

Generates a self-contained HTML page visualizing 3D joints (Three.js) and 2D keypoints (Canvas)
side-by-side for N samples.

Input:
    joints_3d: np.ndarray of shape (T, J, 3)  — or list of such arrays (one per sample)
    joints_2d: np.ndarray of shape (T, J, 2)  — or list of such arrays (one per sample)

Usage:
    python visualize_joints.py          # runs demo with synthetic data
    from visualize_joints import generate_html
"""

import json
import numpy as np
from pathlib import Path

# ─── Skeleton definition (COCO-style 17 joints) ────────────────────────────────
SKELETON_EDGES = [
    [0, 1], [0, 2], [1, 3], [2, 4],          # head
    [5, 6],                                    # shoulders
    [5, 7], [7, 9],                            # left arm
    [6, 8], [8, 10],                           # right arm
    [5, 11], [6, 12],                          # torso
    [11, 12],                                  # hips
    [11, 13], [13, 15],                        # left leg
    [12, 14], [14, 16],                        # right leg
]

JOINT_COLORS_3D = [
    "#FF6B6B","#FF6B6B","#FF6B6B","#FF6B6B","#FF6B6B",  # head (red)
    "#4ECDC4","#4ECDC4",                                  # shoulders (teal)
    "#45B7D1","#45B7D1","#45B7D1","#45B7D1",              # arms (blue)
    "#96CEB4","#96CEB4",                                  # torso
    "#FFEAA7","#FFEAA7","#FFEAA7","#FFEAA7",              # legs (yellow)
]

JOINT_COLORS_2D = JOINT_COLORS_3D


# ─── Three.js viewer template (one per sample, embedded in iframe srcdoc) ──────
THREEJS_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0a0f; overflow:hidden; }}
  canvas {{ display:block; }}
  #controls {{
    position:absolute; bottom:8px; left:50%; transform:translateX(-50%);
    display:flex; align-items:center; gap:8px; z-index:10;
    background:rgba(255,255,255,0.07); backdrop-filter:blur(6px);
    border:1px solid rgba(255,255,255,0.12); border-radius:20px;
    padding:6px 14px;
  }}
  #controls button {{
    background:none; border:none; color:#e0e0e0; cursor:pointer;
    font-size:14px; padding:2px 6px; border-radius:4px; transition:background 0.2s;
  }}
  #controls button:hover {{ background:rgba(255,255,255,0.15); }}
  #frameLabel {{
    color:#aaa; font-family:monospace; font-size:11px; min-width:60px; text-align:center;
  }}
  #slider {{
    -webkit-appearance:none; width:120px; height:3px;
    background:rgba(255,255,255,0.2); border-radius:2px; outline:none; cursor:pointer;
  }}
  #slider::-webkit-slider-thumb {{
    -webkit-appearance:none; width:12px; height:12px; border-radius:50%;
    background:#4ECDC4; cursor:pointer;
  }}
</style>
</head>
<body>
<div id="controls">
  <button id="btnPrev">&#9664;</button>
  <button id="btnPlay">&#9654;</button>
  <button id="btnNext">&#9654;&#9474;</button>
  <input type="range" id="slider" min="0" max="0" value="0">
  <span id="frameLabel">0 / 0</span>
  <button id="btnLoop" title="Loop">&#8635;</button>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const JOINTS_3D = {joints_3d_json};
const EDGES = {edges_json};
const JOINT_COLORS = {joint_colors_json};

const T = JOINTS_3D.length;
const J = JOINTS_3D[0].length;

// ── Scene ──────────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({{ antialias:true, alpha:true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.01, 100);
camera.position.set(0, 0, 2.5);

// Ambient + directional light
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(1, 2, 3);
scene.add(dir);

// Grid floor
const grid = new THREE.GridHelper(2, 20, 0x222244, 0x1a1a2e);
grid.position.y = -1;
scene.add(grid);

// ── Skeleton objects ───────────────────────────────────────────────────────────
const jointMeshes = [];
const boneMeshes  = [];

for (let j = 0; j < J; j++) {{
  const color = new THREE.Color(JOINT_COLORS[j % JOINT_COLORS.length]);
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.018, 10, 10),
    new THREE.MeshStandardMaterial({{ color, emissive: color, emissiveIntensity: 0.4 }})
  );
  scene.add(mesh);
  jointMeshes.push(mesh);
}}

for (let e = 0; e < EDGES.length; e++) {{
  const mat = new THREE.LineBasicMaterial({{ color: 0x4ecdc4, linewidth: 2, transparent:true, opacity:0.7 }});
  const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const line = new THREE.Line(geom, mat);
  scene.add(line);
  boneMeshes.push({{ line, geom }});
}}

// ── Orbit-like controls (manual) ──────────────────────────────────────────────
let isDragging = false, lastX = 0, lastY = 0;
let theta = 0, phi = Math.PI/4, radius = 2.5;
const target = new THREE.Vector3(0, 0, 0);

function updateCamera() {{
  camera.position.x = target.x + radius * Math.sin(phi) * Math.sin(theta);
  camera.position.y = target.y + radius * Math.cos(phi);
  camera.position.z = target.z + radius * Math.sin(phi) * Math.cos(theta);
  camera.lookAt(target);
}}
updateCamera();

renderer.domElement.addEventListener('mousedown', e => {{ isDragging=true; lastX=e.clientX; lastY=e.clientY; }});
window.addEventListener('mouseup', () => isDragging=false);
window.addEventListener('mousemove', e => {{
  if (!isDragging) return;
  theta -= (e.clientX - lastX) * 0.01;
  phi   = Math.max(0.1, Math.min(Math.PI-0.1, phi - (e.clientY - lastY) * 0.01));
  lastX=e.clientX; lastY=e.clientY;
  updateCamera();
}});
renderer.domElement.addEventListener('wheel', e => {{
  radius = Math.max(0.5, Math.min(10, radius + e.deltaY * 0.003));
  updateCamera();
}});
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// ── Normalize skeleton to fit in [-1,1] ───────────────────────────────────────
function normalizeJoints(joints3d) {{
  let minV=[Infinity,Infinity,Infinity], maxV=[-Infinity,-Infinity,-Infinity];
  for (const frame of joints3d)
    for (const j of frame)
      for (let k=0; k<3; k++) {{ minV[k]=Math.min(minV[k],j[k]); maxV[k]=Math.max(maxV[k],j[k]); }}
  const range = Math.max(...[0,1,2].map(k => maxV[k]-minV[k])) || 1;
  const center = [0,1,2].map(k => (minV[k]+maxV[k])/2);
  return joints3d.map(frame =>
    frame.map(j => j.map((v,k) => (v-center[k])/range))
  );
}}
const normed = normalizeJoints(JOINTS_3D);

// ── Frame update ───────────────────────────────────────────────────────────────
function setFrame(f) {{
  const pts = normed[f];
  for (let j=0; j<J; j++) {{
    jointMeshes[j].position.set(pts[j][0], pts[j][1], pts[j][2]);
  }}
  for (let e=0; e<EDGES.length; e++) {{
    const [a,b] = EDGES[e];
    const pos = boneMeshes[e].geom.attributes.position;
    pos.setXYZ(0, pts[a][0], pts[a][1], pts[a][2]);
    pos.setXYZ(1, pts[b][0], pts[b][1], pts[b][2]);
    pos.needsUpdate = true;
  }}
}}

// ── Playback ───────────────────────────────────────────────────────────────────
let curFrame = 0, playing = false, looping = true;
const slider = document.getElementById('slider');
const frameLabel = document.getElementById('frameLabel');
slider.max = T - 1;

function goToFrame(f) {{
  curFrame = (f + T) % T;
  slider.value = curFrame;
  frameLabel.textContent = `${{curFrame+1}} / ${{T}}`;
  setFrame(curFrame);
}}
goToFrame(0);

document.getElementById('btnPlay').addEventListener('click', () => {{
  playing = !playing;
  document.getElementById('btnPlay').innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
}});
document.getElementById('btnPrev').addEventListener('click', () => goToFrame(curFrame - 1));
document.getElementById('btnNext').addEventListener('click', () => goToFrame(curFrame + 1));
document.getElementById('btnLoop').style.color = looping ? '#4ECDC4' : '#aaa';
document.getElementById('btnLoop').addEventListener('click', () => {{
  looping = !looping;
  document.getElementById('btnLoop').style.color = looping ? '#4ECDC4' : '#aaa';
}});
slider.addEventListener('input', () => goToFrame(parseInt(slider.value)));

let lastTime = 0;
const FPS = 25;
function animate(time) {{
  requestAnimationFrame(animate);
  if (playing && time - lastTime > 1000/FPS) {{
    lastTime = time;
    const next = curFrame + 1;
    if (next >= T && !looping) {{ playing = false; document.getElementById('btnPlay').innerHTML='&#9654;'; }}
    else {{ goToFrame(next); }}
  }}
  renderer.render(scene, camera);
}}
animate(0);
</script>
</body>
</html>"""


# ─── 2D Canvas viewer template ─────────────────────────────────────────────────
CANVAS2D_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0a0f; overflow:hidden; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; }}
  canvas {{ border-radius:6px; }}
  #ctrl2d {{
    position:absolute; bottom:8px; left:50%; transform:translateX(-50%);
    display:flex; align-items:center; gap:8px; z-index:10;
    background:rgba(255,255,255,0.07); backdrop-filter:blur(6px);
    border:1px solid rgba(255,255,255,0.12); border-radius:20px;
    padding:6px 14px;
  }}
  #ctrl2d button {{
    background:none; border:none; color:#e0e0e0; cursor:pointer;
    font-size:14px; padding:2px 6px; border-radius:4px;
  }}
  #frameLabel2d {{ color:#aaa; font-family:monospace; font-size:11px; min-width:60px; text-align:center; }}
  #slider2d {{
    -webkit-appearance:none; width:120px; height:3px;
    background:rgba(255,255,255,0.2); border-radius:2px; outline:none; cursor:pointer;
  }}
  #slider2d::-webkit-slider-thumb {{
    -webkit-appearance:none; width:12px; height:12px; border-radius:50%;
    background:#FF6B6B; cursor:pointer;
  }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="ctrl2d">
  <button id="btnPrev2">&#9664;</button>
  <button id="btnPlay2">&#9654;</button>
  <button id="btnNext2">&#9654;&#9474;</button>
  <input type="range" id="slider2d" min="0" max="0" value="0">
  <span id="frameLabel2d">0 / 0</span>
</div>
<script>
const JOINTS_2D = {joints_2d_json};
const EDGES = {edges_json};
const JOINT_COLORS = {joint_colors_json};

const T = JOINTS_2D.length;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

function resize() {{
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight - 40;
}}
resize();
window.addEventListener('resize', () => {{ resize(); drawFrame(curFrame); }});

// normalize 2d to canvas
function normalizeJoints2D(joints2d, W, H, pad=0.1) {{
  let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity;
  for (const frame of joints2d)
    for (const j of frame) {{
      minX=Math.min(minX,j[0]); maxX=Math.max(maxX,j[0]);
      minY=Math.min(minY,j[1]); maxY=Math.max(maxY,j[1]);
    }}
  const rangeX = maxX-minX || 1, rangeY = maxY-minY || 1;
  const scale = Math.min((W*(1-2*pad))/rangeX, (H*(1-2*pad))/rangeY);
  const cx = (minX+maxX)/2, cy = (minY+maxY)/2;
  return joints2d.map(frame =>
    frame.map(j => [
      W/2 + (j[0]-cx)*scale,
      H/2 + (j[1]-cy)*scale
    ])
  );
}}

function drawFrame(f) {{
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const pts = normed2d[f];

  // draw bones
  for (const [a, b] of EDGES) {{
    ctx.beginPath();
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
    ctx.strokeStyle = 'rgba(78,205,196,0.6)';
    ctx.lineWidth = 2;
    ctx.stroke();
  }}

  // draw joints
  for (let j=0; j<pts.length; j++) {{
    const [x,y] = pts[j];
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI*2);
    ctx.fillStyle = JOINT_COLORS[j % JOINT_COLORS.length];
    ctx.shadowColor = JOINT_COLORS[j % JOINT_COLORS.length];
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;
  }}
}}

const normed2d = normalizeJoints2D(JOINTS_2D, canvas.width, canvas.height);

let curFrame = 0, playing = false;
const slider = document.getElementById('slider2d');
const lbl = document.getElementById('frameLabel2d');
slider.max = T - 1;

function goToFrame(f) {{
  curFrame = (f + T) % T;
  slider.value = curFrame;
  lbl.textContent = `${{curFrame+1}} / ${{T}}`;
  drawFrame(curFrame);
}}
goToFrame(0);

document.getElementById('btnPlay2').addEventListener('click', () => {{
  playing = !playing;
  document.getElementById('btnPlay2').innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
}});
document.getElementById('btnPrev2').addEventListener('click', () => goToFrame(curFrame-1));
document.getElementById('btnNext2').addEventListener('click', () => goToFrame(curFrame+1));
slider.addEventListener('input', () => goToFrame(parseInt(slider.value)));

const FPS = 25;
let last = 0;
function animate(t) {{
  requestAnimationFrame(animate);
  if (playing && t - last > 1000/FPS) {{ last = t; goToFrame(curFrame+1); }}
}}
animate(0);
</script>
</body>
</html>"""


# ─── Main HTML wrapper ─────────────────────────────────────────────────────────
MAIN_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pose Viewer — {n_samples} Samples</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  :root {{
    --bg: #07070d;
    --surface: #0f0f1a;
    --border: rgba(255,255,255,0.08);
    --accent: #4ECDC4;
    --accent2: #FF6B6B;
    --text: #e8e8f0;
    --muted: #6b6b88;
  }}

  * {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
  }}

  header {{
    padding: 24px 32px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: baseline; gap: 16px;
  }}
  header h1 {{
    font-family: 'Space Mono', monospace;
    font-size: 18px; font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  header span {{
    font-size: 12px; color: var(--muted);
    font-family: 'Space Mono', monospace;
  }}

  .grid-header {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }}
  .col-label {{
    padding: 10px 20px;
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
  }}
  .col-label:first-child {{
    border-right: 1px solid var(--border);
    color: var(--accent);
  }}
  .col-label:last-child {{ color: var(--accent2); }}

  .sample-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-bottom: 1px solid var(--border);
    min-height: 320px;
  }}
  .sample-row:last-child {{ border-bottom: none; }}

  .cell {{
    position: relative;
    border-right: 1px solid var(--border);
  }}
  .cell:last-child {{ border-right: none; }}

  .sample-badge {{
    position: absolute; top: 10px; left: 12px; z-index: 20;
    font-family: 'Space Mono', monospace;
    font-size: 10px; font-weight: 700;
    background: rgba(0,0,0,0.6); backdrop-filter:blur(4px);
    border: 1px solid var(--border);
    padding: 3px 8px; border-radius: 10px;
    color: var(--muted);
  }}

  iframe {{
    width: 100%; height: 320px;
    border: none; display: block;
  }}

  /* subtle row hover */
  .sample-row:hover .cell {{
    background: rgba(78,205,196,0.015);
  }}
</style>
</head>
<body>

<header>
  <h1>Pose Visualizer</h1>
  <span>{n_samples} samples · {n_frames} frames · {n_joints} joints</span>
</header>

<div class="grid-header">
  <div class="col-label">3D Skeleton · Three.js</div>
  <div class="col-label">2D Keypoints · Canvas</div>
</div>

{rows_html}

</body>
</html>"""


def _escape_for_srcdoc(html: str) -> str:
    """Escape HTML string to be safely embedded in srcdoc attribute."""
    return html.replace("&", "&amp;").replace('"', "&quot;")


def generate_html(
    joints_3d_list: list,   # list of np.ndarray (T, J, 3)
    joints_2d_list: list,   # list of np.ndarray (T, J, 2)
    edges: list = None,
    joint_colors: list = None,
    output_path: str = "pose_viewer.html",
) -> str:
    """
    Generate a self-contained HTML page with N rows, each showing
    a 3D skeleton viewer (Three.js) and a 2D keypoints viewer (Canvas).

    Returns the output path.
    """
    assert len(joints_3d_list) == len(joints_2d_list), "Mismatched sample counts"

    if edges is None:
        edges = SKELETON_EDGES
    if joint_colors is None:
        joint_colors = JOINT_COLORS_3D

    edges_json       = json.dumps(edges)
    joint_colors_json = json.dumps(joint_colors)

    n_samples = len(joints_3d_list)
    n_frames  = int(np.array(joints_3d_list[0]).shape[0])
    n_joints  = int(np.array(joints_3d_list[0]).shape[1])

    rows_html_parts = []

    for i, (j3d, j2d) in enumerate(zip(joints_3d_list, joints_2d_list)):
        j3d = np.array(j3d).tolist()   # (T, J, 3)
        j2d = np.array(j2d).tolist()   # (T, J, 2)

        # Build 3D iframe srcdoc
        viewer3d_html = THREEJS_VIEWER_TEMPLATE.format(
            joints_3d_json=json.dumps(j3d),
            edges_json=edges_json,
            joint_colors_json=joint_colors_json,
        )

        # Build 2D iframe srcdoc
        viewer2d_html = CANVAS2D_TEMPLATE.format(
            joints_2d_json=json.dumps(j2d),
            edges_json=edges_json,
            joint_colors_json=joint_colors_json,
        )

        srcdoc3d = _escape_for_srcdoc(viewer3d_html)
        srcdoc2d = _escape_for_srcdoc(viewer2d_html)

        row = f"""
  <div class="sample-row">
    <div class="cell">
      <span class="sample-badge">S{i+1}</span>
      <iframe srcdoc="{srcdoc3d}" sandbox="allow-scripts"></iframe>
    </div>
    <div class="cell">
      <span class="sample-badge">S{i+1}</span>
      <iframe srcdoc="{srcdoc2d}" sandbox="allow-scripts"></iframe>
    </div>
  </div>"""
        rows_html_parts.append(row)

    rows_html = "\n".join(rows_html_parts)

    final_html = MAIN_HTML_TEMPLATE.format(
        n_samples=n_samples,
        n_frames=n_frames,
        n_joints=n_joints,
        rows_html=rows_html,
    )

    Path(output_path).write_text(final_html, encoding="utf-8")
    print(f"✓ Saved → {output_path}  ({len(final_html)//1024} KB)")
    return output_path


# ─── Demo / CLI ────────────────────────────────────────────────────────────────
def _make_synthetic_data(n_samples=3, T=60, J=17):
    """Generate synthetic walking-like skeleton data."""
    joints_3d_list = []
    joints_2d_list = []

    # rough COCO joint layout in 3D
    base_3d = np.array([
        [0, 0.9, 0],   # 0 nose
        [-0.05, 0.95, 0], # 1 left eye
        [0.05, 0.95, 0],  # 2 right eye
        [-0.1, 0.9, 0],   # 3 left ear
        [0.1, 0.9, 0],    # 4 right ear
        [-0.2, 0.6, 0],   # 5 left shoulder
        [0.2, 0.6, 0],    # 6 right shoulder
        [-0.35, 0.3, 0],  # 7 left elbow
        [0.35, 0.3, 0],   # 8 right elbow
        [-0.3, 0.0, 0],   # 9 left wrist
        [0.3, 0.0, 0],    # 10 right wrist
        [-0.15, 0.0, 0],  # 11 left hip
        [0.15, 0.0, 0],   # 12 right hip
        [-0.18, -0.4, 0], # 13 left knee
        [0.18, -0.4, 0],  # 14 right knee
        [-0.15, -0.8, 0], # 15 left ankle
        [0.15, -0.8, 0],  # 16 right ankle
    ], dtype=np.float32)

    for s in range(n_samples):
        phase = s * 0.5
        frames_3d, frames_2d = [], []
        for t in range(T):
            theta = 2 * np.pi * t / T + phase
            jitter = np.random.randn(J, 3) * 0.01
            pose = base_3d.copy() + jitter
            # animate legs
            pose[13, 2] += 0.2 * np.sin(theta)
            pose[15, 2] += 0.25 * np.sin(theta)
            pose[14, 2] += 0.2 * np.sin(theta + np.pi)
            pose[16, 2] += 0.25 * np.sin(theta + np.pi)
            # animate arms
            pose[7, 2]  += 0.15 * np.sin(theta + np.pi)
            pose[9, 2]  += 0.18 * np.sin(theta + np.pi)
            pose[8, 2]  += 0.15 * np.sin(theta)
            pose[10, 2] += 0.18 * np.sin(theta)

            frames_3d.append(pose.tolist())
            # 2d: project orthographically (x, y) with some scale
            xy = pose[:, :2] * 200 + np.array([320, 240])
            frames_2d.append(xy.tolist())

        joints_3d_list.append(frames_3d)
        joints_2d_list.append(frames_2d)

    return joints_3d_list, joints_2d_list


if __name__ == "__main__":
    print("Generating synthetic demo data...")
    j3d_list, j2d_list = _make_synthetic_data(n_samples=3, T=80, J=17)

    generate_html(
        joints_3d_list=j3d_list,
        joints_2d_list=j2d_list,
        output_path="pose_viewer.html",
    )