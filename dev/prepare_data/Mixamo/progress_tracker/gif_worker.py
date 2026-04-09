"""
GIF preview generator — runs as a background thread pool inside app.py.

For each completed (or partial) render job:
  - Checks if a GIF already exists at <cam_path>/.preview.gif
  - If not (or if more frames have been added since last generation), creates one
  - GIF is a thumbnail-sized (~200px wide) looping animation sampled from rendered frames
  - Uses a thread pool so multiple GIFs are generated concurrently
  - Never blocks the Flask request threads

Public API (called from app.py):
  gif_worker.start(n_workers=4)
  gif_worker.queue_job(job)        # enqueue a single job for (re)generation
  gif_worker.get_gif_path(job_id)  # returns path if ready, None if pending/missing
"""

import os
import threading
import queue
import time
import hashlib
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
GIF_FILENAME   = ".preview.gif"   # hidden file inside each cam folder
GIF_WIDTH      = 200              # thumbnail width (height auto)
GIF_MAX_FRAMES = 24               # max frames sampled into the GIF
GIF_FPS        = 12               # playback speed
GIF_DURATION   = int(1000 / GIF_FPS)  # ms per frame

# ── State ──────────────────────────────────────────────────────────────────────
_queue     = queue.Queue()
_gif_cache = {}        # job_id -> gif_path (once ready)
_gif_lock  = threading.Lock()
_in_flight = set()     # job_ids currently being processed
_started   = False


# ── Worker ─────────────────────────────────────────────────────────────────────

def _make_gif(job):
    """Generate (or skip) a GIF preview for one job. Called in worker thread."""
    job_id   = job["job_id"]
    cam_path = job["cam_path"]
    frames   = job["frames"]   # sorted list of filenames

    if not frames:
        return

    gif_path = os.path.join(cam_path, GIF_FILENAME)

    # Build a hash of the current frame list to detect if we need to regenerate
    frame_sig = hashlib.md5("|".join(frames).encode()).hexdigest()[:12]
    sig_file  = gif_path + ".sig"

    # Check if existing GIF is still valid
    if os.path.isfile(gif_path) and os.path.isfile(sig_file):
        try:
            if open(sig_file).read().strip() == frame_sig:
                with _gif_lock:
                    _gif_cache[job_id] = gif_path
                return  # already up to date
        except Exception:
            pass

    # Sample frames evenly (max GIF_MAX_FRAMES)
    step    = max(1, len(frames) // GIF_MAX_FRAMES)
    sampled = frames[::step][:GIF_MAX_FRAMES]

    pil_frames = []
    for fname in sampled:
        fpath = os.path.join(cam_path, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            # Resize maintaining aspect ratio
            w, h   = img.size
            new_h  = max(1, int(h * GIF_WIDTH / w))
            img    = img.resize((GIF_WIDTH, new_h), Image.LANCZOS)
            pil_frames.append(img)
        except Exception:
            continue  # skip corrupt/missing frame

    if not pil_frames:
        return

    try:
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=True,
            loop=0,
            duration=GIF_DURATION,
        )
        # Write signature
        with open(sig_file, "w") as f:
            f.write(frame_sig)

        with _gif_lock:
            _gif_cache[job_id] = gif_path

    except Exception as e:
        print(f"[gif_worker] Failed to write GIF for {job_id}: {e}")


def _worker_loop():
    while True:
        job = _queue.get()
        try:
            _make_gif(job)
        except Exception as e:
            print(f"[gif_worker] Error: {e}")
        finally:
            with _gif_lock:
                _in_flight.discard(job["job_id"])
            _queue.task_done()


# ── Public API ─────────────────────────────────────────────────────────────────

def start(n_workers=4):
    global _started
    if _started:
        return
    _started = True
    for _ in range(n_workers):
        t = threading.Thread(target=_worker_loop, daemon=True)
        t.start()
    print(f"[gif_worker] Started {n_workers} worker threads")


def queue_job(job):
    """Enqueue a job for GIF generation. Safe to call repeatedly — deduplicates."""
    job_id = job["job_id"]
    with _gif_lock:
        if job_id in _in_flight:
            return
        _in_flight.add(job_id)
    _queue.put(job)


def get_gif_url(job_id):
    """Returns the Flask URL to serve the GIF, or None if not ready."""
    with _gif_lock:
        path = _gif_cache.get(job_id)
    if path and os.path.isfile(path):
        return f"/gif/{job_id}"
    return None


def get_gif_path(job_id):
    with _gif_lock:
        return _gif_cache.get(job_id)


def is_pending(job_id):
    with _gif_lock:
        return job_id in _in_flight
