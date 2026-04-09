import os
import json
import re
import threading
import time
import subprocess
from flask import Flask, render_template, jsonify, abort, send_from_directory, request

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
RENDER_ROOT   = os.environ.get("RENDER_ROOT",   "/data/mint/Motion_Dataset/Mixamo/render_fbx")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "30"))
STATIC_PORT   = int(os.environ.get("STATIC_PORT",   "5001"))

# STATIC_BACKEND: "node" | "python" | "nginx" | "flask"
#   node   — fastest overall, uses Node.js (node_static.js must be alongside app.py)
#   python — pure Python, no extra deps, nearly as fast
#   nginx  — fastest possible, requires Nginx configured externally
#   flask  — slowest, no extra process needed
import shutil as _shutil
_NODE_BIN = _shutil.which("node") or _shutil.which("nodejs")
STATIC_BACKEND = os.environ.get("STATIC_BACKEND", "node" if _NODE_BIN else "python")

USE_NGINX    = os.environ.get("USE_NGINX", "0") == "1"  # legacy alias for nginx backend
NGINX_PREFIX = os.environ.get("NGINX_INTERNAL_PREFIX", "/renders")

FRAME_RE = re.compile(r"^frame\d{4}\.png$", re.IGNORECASE)
CAM_RE   = re.compile(r"^cam-(.+)$",        re.IGNORECASE)


if USE_NGINX:
    STATIC_BACKEND = "nginx"


# ── In-memory cache ────────────────────────────────────────────────────────────
_cache_lock   = threading.Lock()
_jobs_cache   = []
_frames_cache = {}
_last_scan    = 0.0
_scan_running = False


# ── Scanner ────────────────────────────────────────────────────────────────────

def parse_frame_range(frame_range):
    if isinstance(frame_range, (list, tuple)) and len(frame_range) >= 2:
        return int(frame_range[0]), int(frame_range[1])
    if isinstance(frame_range, str) and "-" in frame_range:
        parts = frame_range.split("-")
        return int(parts[0]), int(parts[1])
    if isinstance(frame_range, (int, float)):
        return 0, int(frame_range) - 1
    return None, None


def _scan_once():
    jobs   = []
    frames = {}
    if not os.path.isdir(RENDER_ROOT):
        return jobs, frames

    for char_name in sorted(os.listdir(RENDER_ROOT)):
        char_path = os.path.join(RENDER_ROOT, char_name)
        if not os.path.isdir(char_path):
            continue
        for motion_name in sorted(os.listdir(char_path)):
            motion_path = os.path.join(char_path, motion_name)
            if not os.path.isdir(motion_path):
                continue
            for cam_dir in sorted(os.listdir(motion_path)):
                cam_path = os.path.join(motion_path, cam_dir)
                if not os.path.isdir(cam_path):
                    continue
                m = CAM_RE.match(cam_dir)
                if not m:
                    continue
                cam_id = m.group(1)

                json_path = os.path.join(cam_path, f"skeleton_cam-{cam_id}.json")
                if not os.path.isfile(json_path):
                    continue
                try:
                    with open(json_path) as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}

                frame_range    = meta.get("frame_range")
                start, end     = parse_frame_range(frame_range)
                total_expected = max(0, end - start + 1) if (start is not None and end is not None) else None
                frame_files    = sorted(f for f in os.listdir(cam_path) if FRAME_RE.match(f))
                rendered_count = len(frame_files)

                if total_expected is not None:
                    remaining = max(0, total_expected - rendered_count)
                    pct       = min(100.0, rendered_count / total_expected * 100) if total_expected else 0.0
                    done      = rendered_count >= total_expected
                else:
                    remaining = pct = None
                    done = False

                job_id   = f"{char_name}|{motion_name}|{cam_id}"
                rel_path = os.path.join(char_name, motion_name, cam_dir)

                jobs.append({
                    "job_id":         job_id,
                    "char_name":      char_name,
                    "motion_name":    motion_name,
                    "cam_id":         cam_id,
                    "cam_dir":        cam_dir,
                    "cam_path":       cam_path,
                    "rel_path":       rel_path,
                    "total_expected": total_expected,
                    "rendered_count": rendered_count,
                    "remaining":      remaining,
                    "pct":            round(pct, 1) if pct is not None else None,
                    "done":           done,
                    "frame_range":    frame_range,
                    "json_meta":      meta,
                })
                frames[job_id] = frame_files

    return jobs, frames


def _refresh_cache():
    global _jobs_cache, _frames_cache, _last_scan, _scan_running
    _scan_running = True
    try:
        jobs, frames = _scan_once()
        with _cache_lock:
            _jobs_cache   = jobs
            _frames_cache = frames
            _last_scan    = time.time()
    except Exception as e:
        print(f"[render_monitor] scan error: {e}")
    finally:
        _scan_running = False


def _background_loop():
    while True:
        time.sleep(SCAN_INTERVAL)
        _refresh_cache()


def get_cached_jobs():
    with _cache_lock:
        return list(_jobs_cache)

def get_cached_frames(job_id):
    with _cache_lock:
        return _frames_cache.get(job_id)

def find_job(job_id):
    with _cache_lock:
        for j in _jobs_cache:
            if j["job_id"] == job_id:
                return j
    return None


# ── Startup scan ───────────────────────────────────────────────────────────────
print(f"[render_monitor] Initial scan of {RENDER_ROOT} ...")
_refresh_cache()
print(f"[render_monitor] Found {len(_jobs_cache)} jobs. Refresh every {SCAN_INTERVAL}s.")
threading.Thread(target=_background_loop, daemon=True).start()


# ── Static file backend startup ────────────────────────────────────────────────
_static_proc = None   # Node subprocess handle

if STATIC_BACKEND == "node":
    node_script = os.path.join(os.path.dirname(__file__), "node_static.js")
    if os.path.isfile(node_script):
        _static_proc = subprocess.Popen(
            [_NODE_BIN, node_script],
            env={**os.environ, "RENDER_ROOT": RENDER_ROOT, "STATIC_PORT": str(STATIC_PORT)},
        )
        print(f"[render_monitor] Node.js static server started on port {STATIC_PORT}")
    else:
        print(f"[render_monitor] node_static.js not found, falling back to Python server")
        STATIC_BACKEND = "python"

if STATIC_BACKEND == "python":
    import static_server
    static_server.start(port=STATIC_PORT, render_root=RENDER_ROOT, daemon=True)

if STATIC_BACKEND == "nginx":
    print(f"[render_monitor] Using Nginx at prefix {NGINX_PREFIX}")

if STATIC_BACKEND == "flask":
    print(f"[render_monitor] Using Flask for frame serving (slow — set STATIC_BACKEND=node for best speed)")

print(f"[render_monitor] Static backend: {STATIC_BACKEND.upper()}")

# ── GIF preview worker ─────────────────────────────────────────────────────────
import gif_worker
GIF_WORKERS = int(os.environ.get("GIF_WORKERS", "4"))
gif_worker.start(n_workers=GIF_WORKERS)

def _queue_gif_jobs(jobs):
    for job in jobs:
        if job["rendered_count"] > 0:
            frames = _frames_cache.get(job["job_id"]) or []
            gif_worker.queue_job({**job, "frames": frames})

# Queue initial GIF batch right after first scan
_queue_gif_jobs(_jobs_cache)

# Hook into background refresh so new frames trigger GIF regeneration
_orig_bg = _background_loop
def _background_loop_with_gif():
    while True:
        import time as _t; _t.sleep(SCAN_INTERVAL)
        _refresh_cache()
        with _cache_lock:
            _queue_gif_jobs(_jobs_cache)
# Replace the background thread (already started above — restart with new fn)
import threading as _threading
_threading.Thread(target=_background_loop_with_gif, daemon=True).start()



def _frame_base_url(job):
    host = request.host.split(":")[0]
    if STATIC_BACKEND == "nginx":
        return f"{NGINX_PREFIX}/{job['rel_path']}/"
    elif STATIC_BACKEND in ("node", "python"):
        return f"http://{host}:{STATIC_PORT}/{job['rel_path']}/"
    else:  # flask
        return f"/frame/{job['job_id']}/"


# ── Cleanup on exit ────────────────────────────────────────────────────────────
import atexit
def _cleanup():
    if _static_proc:
        _static_proc.terminate()
atexit.register(_cleanup)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    jobs = get_cached_jobs()
    age  = int(time.time() - _last_scan)
    return render_template("index.html",
                           jobs=jobs,
                           render_root=RENDER_ROOT,
                           scan_interval=SCAN_INTERVAL,
                           cache_age=age)


@app.route("/api/jobs")
def api_jobs():
    jobs = get_cached_jobs()
    age  = int(time.time() - _last_scan)
    light = [{k: v for k, v in j.items() if k not in ("json_meta", "cam_path", "rel_path")}
             for j in jobs]
    return jsonify({"jobs": light, "cache_age": age, "scan_interval": SCAN_INTERVAL})


@app.route("/api/status")
def api_status():
    return jsonify({
        "scan_running":   _scan_running,
        "last_scan_ago":  int(time.time() - _last_scan),
        "scan_interval":  SCAN_INTERVAL,
        "job_count":      len(_jobs_cache),
        "static_backend": STATIC_BACKEND,
        "static_port":    STATIC_PORT,
    })


@app.route("/job/<path:job_id>")
def job_detail(job_id):
    job = find_job(job_id)
    if job is None:
        abort(404)
    return render_template("detail.html", job=job)


@app.route("/api/job/<path:job_id>/frames")
def api_frames(job_id):
    frames = get_cached_frames(job_id)
    if frames is None:
        abort(404)
    job      = find_job(job_id)
    base_url = _frame_base_url(job)
    return jsonify({"frames": frames, "count": len(frames), "base_url": base_url})



@app.route("/gif/<path:job_id>")
def serve_gif(job_id):
    """Serve the pre-generated GIF preview for a job."""
    gif_path = gif_worker.get_gif_path(job_id)
    if not gif_path or not os.path.isfile(gif_path):
        abort(404)
    return send_from_directory(os.path.dirname(gif_path),
                               os.path.basename(gif_path),
                               mimetype="image/gif")


@app.route("/api/gif_status")
def api_gif_status():
    """Returns gif readiness for all jobs — polled by the grid view."""
    jobs = get_cached_jobs()
    result = {}
    for j in jobs:
        jid = j["job_id"]
        result[jid] = {
            "ready":   gif_worker.get_gif_url(jid) is not None,
            "pending": gif_worker.is_pending(jid),
            "url":     gif_worker.get_gif_url(jid),
        }
    return jsonify(result)

@app.route("/frame/<path:job_id>/<filename>")
def serve_frame(job_id, filename):
    """Fallback when STATIC_BACKEND=flask."""
    if not FRAME_RE.match(filename):
        abort(400)
    job = find_job(job_id)
    if job is None:
        abort(404)
    return send_from_directory(job["cam_path"], filename)


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True, host="0.0.0.0")
