#!/usr/bin/env python3
"""
Seed script — creates a fake render_fbx directory tree for testing.
Usage:  python seed_demo.py [root_path]
"""
import sys, os, json, struct, zlib

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/render_fbx_demo"

JOBS = [
    ("Xbot",    "Walking",  "1", (0,  47),  30),
    ("Xbot",    "Running",  "1", (0,  89),  45),
    ("Xbot",    "Running",  "2", (0,  89),   0),   # not started
    ("Ybot",    "Jump",     "1", (10, 59),  50),
    ("Ybot",    "Idle",     "1", (0, 119), 120),   # 100% done
    ("Paladin", "Attack",   "1", (0,  35),  10),
]

def make_png(w=4, h=4):
    """Minimal valid 4x4 black PNG."""
    def chunk(tag, data):
        c = zlib.crc32(tag + data) & 0xffffffff
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', c)
    sig   = b'\x89PNG\r\n\x1a\n'
    ihdr  = chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    raw   = b''.join(b'\x00' + b'\x00\x00\x00' * w for _ in range(h))
    idat  = chunk(b'IDAT', zlib.compress(raw))
    iend  = chunk(b'IEND', b'')
    return sig + ihdr + idat + iend

PNG = make_png()

for char, motion, cam_id, frame_range, rendered_count in JOBS:
    cam_dir = os.path.join(ROOT, char, motion, f"cam_{cam_id}")
    os.makedirs(cam_dir, exist_ok=True)

    # JSON
    meta = {"frame_range": list(frame_range), "fps": 24, "camera": int(cam_id)}
    with open(os.path.join(cam_dir, f"skeleton_cam-{cam_id}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Frames
    start = frame_range[0]
    for i in range(rendered_count):
        name = f"frame{start + i:04d}.png"
        path = os.path.join(cam_dir, name)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(PNG)

print(f"Demo data created at: {ROOT}")
print(f"Run:  RENDER_ROOT={ROOT} python app.py")
