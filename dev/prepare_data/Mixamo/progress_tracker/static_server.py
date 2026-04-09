"""
Fast static file server for render frames — no Nginx needed.
Runs as a thread inside app.py, or standalone for debugging.

Uses Python's built-in http.server with:
  - ThreadingMixIn  → handles multiple requests concurrently
  - Range headers   → browsers can resume/seek without re-downloading
  - ETags           → browser caches frames; repeat views are instant
  - CORS header     → Flask (port 5000) can fetch from this server (port 5001)

Start automatically via app.py, or manually:
    python static_server.py
"""

import os
import hashlib
import threading
import email.utils
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn


RENDER_ROOT  = os.environ.get("RENDER_ROOT", "/data/mint/Motion_Dataset/Mixamo/render_fbx")
STATIC_PORT  = int(os.environ.get("STATIC_PORT", "5001"))
ALLOWED_EXT  = {".png", ".jpg", ".jpeg", ".exr", ".webp"}


class FrameHandler(BaseHTTPRequestHandler):
    """
    Serves files under RENDER_ROOT at URL path /<relative_path>.
    Only allows image extensions. Supports Range, ETag, If-None-Match.
    """

    def log_message(self, fmt, *args):
        pass  # silence per-request logs; errors still print

    def do_GET(self):
        self._serve(head_only=False)

    def do_HEAD(self):
        self._serve(head_only=True)

    def _serve(self, head_only=False):
        # Strip query string
        path = self.path.split("?", 1)[0]

        # Security: block path traversal
        rel = os.path.normpath(path.lstrip("/"))
        if rel.startswith(".."):
            self._send_error(403)
            return

        # Block non-image extensions
        _, ext = os.path.splitext(rel)
        if ext.lower() not in ALLOWED_EXT:
            self._send_error(403)
            return

        full_path = os.path.join(RENDER_ROOT, rel)
        if not os.path.isfile(full_path):
            self._send_error(404)
            return

        stat   = os.stat(full_path)
        size   = stat.st_size
        mtime  = stat.st_mtime

        # ETag based on path + mtime + size (cheap, no file read needed)
        etag_raw = f"{full_path}:{mtime}:{size}"
        etag = '"' + hashlib.md5(etag_raw.encode()).hexdigest()[:16] + '"'

        # 304 Not Modified
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.end_headers()
            return

        # Range request support (browser seeks in a video-like slider)
        range_header = self.headers.get("Range")
        start, end = 0, size - 1

        if range_header and range_header.startswith("bytes="):
            try:
                r = range_header[6:].split("-")
                start = int(r[0]) if r[0] else 0
                end   = int(r[1]) if r[1] else size - 1
                end   = min(end, size - 1)
                status = 206
            except (ValueError, IndexError):
                status = 200
        else:
            status = 200

        content_length = end - start + 1

        self.send_response(status)
        self.send_header("Content-Type",   _mime(ext))
        self.send_header("Content-Length", str(content_length))
        self.send_header("ETag",           etag)
        self.send_header("Accept-Ranges",  "bytes")
        self.send_header("Cache-Control",  "public, max-age=3600, immutable")
        self.send_header("Last-Modified",  email.utils.formatdate(mtime, usegmt=True))
        self.send_header("Access-Control-Allow-Origin", "*")   # allow Flask page to fetch
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if head_only:
            return

        with open(full_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            chunk = 64 * 1024  # 64 KB chunks
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(data)

    def _send_error(self, code):
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _mime(ext):
    return {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".exr":  "image/x-exr",
    }.get(ext.lower(), "application/octet-stream")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in its own thread."""
    daemon_threads = True
    allow_reuse_address = True


def start(port=None, render_root=None, daemon=True):
    """Start the static server. Call from app.py."""
    global RENDER_ROOT, STATIC_PORT
    if render_root:
        RENDER_ROOT = render_root
    if port:
        STATIC_PORT = port

    server = ThreadedHTTPServer(("0.0.0.0", STATIC_PORT), FrameHandler)
    t = threading.Thread(target=server.serve_forever, name="static-server", daemon=daemon)
    t.start()
    print(f"[static_server] Serving {RENDER_ROOT} on port {STATIC_PORT}")
    return server


if __name__ == "__main__":
    start(daemon=False)
