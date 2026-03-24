import os
import re
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, abort, render_template_string, send_file

app = Flask(__name__)

# Change this to your dataset root
ROOT_DIR = Path("/data/mint/Motion_Dataset/Mixamo/render_fbx").resolve()

FRAME_PATTERN = re.compile(r"^frame(\d+)\.png$", re.IGNORECASE)
CAM_DIR_PATTERN = re.compile(r"^cam_(\d+)$")
SKELETON_FILE_PATTERN = "skeleton_cam-{}.json"


def safe_resolve_under_root(rel_path: str) -> Path:
    path = (ROOT_DIR / rel_path).resolve()
    if ROOT_DIR not in path.parents and path != ROOT_DIR:
        raise ValueError("Path escapes ROOT_DIR")
    return path


def parse_frame_range(frame_range: Any) -> Optional[int]:
    """
    Try to infer total frame count from various possible frame_range formats.

    Supported examples:
    - [0, 119]              -> 120
    - [1, 120]              -> 120
    - {"start": 0, "end": 119} -> 120
    - {"start_frame": 1, "end_frame": 120} -> 120
    - [0, 1, 2, 3, ...]     -> len(list)
    """
    print(frame_range)
    if frame_range is None:
        return None

    if isinstance(frame_range, dict):
        start = (
            frame_range.get("start")
            if "start" in frame_range
            else frame_range.get("start_frame")
        )
        end = (
            frame_range.get("end")
            if "end" in frame_range
            else frame_range.get("end_frame")
        )
        if isinstance(start, int) and isinstance(end, int):
            return max(0, end - start + 1)
        return None

    if isinstance(frame_range, list):
        if len(frame_range) == 2 and all(isinstance(x, int) for x in frame_range):
            start, end = frame_range
            return max(0, end - start + 1)

        if all(isinstance(x, int) for x in frame_range):
            return len(frame_range)

    return None


def find_skeleton_json(cam_dir: Path, cam_id: str) -> Optional[Path]:
    expected = cam_dir / SKELETON_FILE_PATTERN.format(cam_id)
    if expected.exists():
        return expected

    # fallback: find any skeleton_cam-*.json
    candidates = sorted(cam_dir.glob("skeleton_cam-*.json"))
    return candidates[0] if candidates else None


def read_total_expected_frames(cam_dir: Path, cam_id: str) -> Optional[int]:
    skeleton_path = find_skeleton_json(cam_dir, cam_id)
    if skeleton_path is None:
        return None

    try:
        with open(skeleton_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return parse_frame_range(data.get("frame_range"))
    except Exception:
        return None


def list_rendered_frames(cam_dir: Path) -> List[Dict[str, Any]]:
    frames = []
    for entry in cam_dir.iterdir():
        if not entry.is_file():
            continue
        m = FRAME_PATTERN.match(entry.name)
        if m:
            frame_idx = int(m.group(1))
            frames.append(
                {
                    "name": entry.name,
                    "frame_idx": frame_idx,
                    "rel_path": str(entry.relative_to(ROOT_DIR)),
                }
            )
    frames.sort(key=lambda x: x["frame_idx"])
    return frames


def collect_progress(root_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    if not root_dir.exists():
        return items

    for character_dir in sorted(root_dir.iterdir()):
        if not character_dir.is_dir():
            continue

        for motion_dir in sorted(character_dir.iterdir()):
            if not motion_dir.is_dir():
                continue

            for cam_dir in sorted(motion_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue

                cam_match = CAM_DIR_PATTERN.match(cam_dir.name)
                if not cam_match:
                    continue

                cam_id = cam_match.group(1)

                rendered_frames = list_rendered_frames(cam_dir)
                rendered_count = len(rendered_frames)
                total_expected = read_total_expected_frames(cam_dir, cam_id)

                if total_expected is None:
                    remaining = None
                    percent = None
                else:
                    remaining = max(0, total_expected - rendered_count)
                    percent = 0.0 if total_expected == 0 else (rendered_count / total_expected) * 100.0

                items.append(
                    {
                        "character": character_dir.name,
                        "motion": motion_dir.name,
                        "camera": cam_dir.name,
                        "camera_id": cam_id,
                        "relative_dir": str(cam_dir.relative_to(ROOT_DIR)),
                        "rendered": rendered_count,
                        "total": total_expected,
                        "remaining": remaining,
                        "percent": round(percent, 2) if percent is not None else None,
                        "frames": rendered_frames,
                        "first_frame": rendered_frames[0]["rel_path"] if rendered_frames else None,
                        "last_frame": rendered_frames[-1]["rel_path"] if rendered_frames else None,
                    }
                )

    # show unfinished first, then most complete
    items.sort(
        key=lambda x: (
            x["percent"] is not None and x["percent"] >= 100.0,
            -(x["percent"] or 0.0),
            x["character"],
            x["motion"],
            x["camera"],
        )
    )
    return items


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Render FBX Progress Tracker</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #111827;
            color: #f9fafb;
        }
        h1 {
            margin-bottom: 8px;
        }
        .sub {
            color: #9ca3af;
            margin-bottom: 16px;
        }
        .toolbar {
            display: flex;
            gap: 12px;
            align-items: center;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        input, button, select {
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid #374151;
            background: #1f2937;
            color: #f9fafb;
        }
        button {
            cursor: pointer;
        }
        .stats {
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        .card {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 12px 16px;
            min-width: 180px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #1f2937;
            border-radius: 12px;
            overflow: hidden;
        }
        th, td {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid #374151;
            vertical-align: middle;
        }
        th {
            background: #111827;
            position: sticky;
            top: 0;
        }
        tr:hover {
            background: #243041;
        }
        tr.clickable {
            cursor: pointer;
        }
        .progress-wrap {
            width: 220px;
        }
        .progress-bar {
            height: 10px;
            width: 100%;
            background: #374151;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 4px;
        }
        .progress-fill {
            height: 100%;
            background: #22c55e;
        }
        .muted {
            color: #9ca3af;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 999px;
            background: #374151;
            font-size: 12px;
        }
        .modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
        }
        .modal-content {
            width: min(1100px, 95vw);
            margin: 3vh auto;
            background: #111827;
            border: 1px solid #374151;
            border-radius: 16px;
            padding: 16px;
        }
        .viewer-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }
        .viewer-controls {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }
        .frame-box {
            display: flex;
            justify-content: center;
            align-items: center;
            background: black;
            border-radius: 12px;
            min-height: 400px;
            overflow: hidden;
        }
        .frame-box img {
            max-width: 100%;
            max-height: 75vh;
            display: block;
        }
        .slider-row {
            margin-top: 12px;
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        input[type="range"] {
            flex: 1;
            min-width: 240px;
        }
        .small {
            font-size: 12px;
            color: #9ca3af;
        }
    </style>
</head>
<body>
    <h1>Render FBX Progress Tracker</h1>
    <div class="sub">Root: <code>{{ root_dir }}</code></div>

    <div class="toolbar">
        <input id="searchBox" type="text" placeholder="Filter by character / motion / cam" />
        <label>
            Auto refresh
            <select id="refreshSelect">
                <option value="5000">5 sec</option>
                <option value="10000" selected>10 sec</option>
                <option value="30000">30 sec</option>
                <option value="0">Off</option>
            </select>
        </label>
        <button onclick="refreshData()">Refresh now</button>
    </div>

    <div class="stats">
        <div class="card"><div class="muted">Total camera jobs</div><div id="statJobs">-</div></div>
        <div class="card"><div class="muted">Completed</div><div id="statCompleted">-</div></div>
        <div class="card"><div class="muted">In progress</div><div id="statInProgress">-</div></div>
        <div class="card"><div class="muted">Unknown total</div><div id="statUnknown">-</div></div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Character</th>
                <th>Motion</th>
                <th>Camera</th>
                <th>Rendered</th>
                <th>Total</th>
                <th>Remaining</th>
                <th>Progress</th>
                <th>Preview</th>
            </tr>
        </thead>
        <tbody id="tableBody"></tbody>
    </table>

    <div id="viewerModal" class="modal" onclick="closeViewer(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="viewer-top">
                <div>
                    <h2 id="viewerTitle" style="margin:0;"></h2>
                    <div id="viewerMeta" class="small"></div>
                </div>
                <div class="viewer-controls">
                    <button onclick="prevFrame()">Prev</button>
                    <button id="playBtn" onclick="togglePlay()">Play</button>
                    <button onclick="nextFrame()">Next</button>
                    <label>FPS <input id="fpsInput" type="number" value="12" min="1" max="60" style="width:70px;"></label>
                    <button onclick="closeViewer()">Close</button>
                </div>
            </div>

            <div class="frame-box">
                <img id="viewerImage" src="" alt="frame preview">
            </div>

            <div class="slider-row">
                <input id="frameSlider" type="range" min="0" max="0" value="0" oninput="onSliderChange()">
                <div id="frameInfo">Frame - / -</div>
            </div>
        </div>
    </div>

    <script>
        let allItems = [];
        let filteredItems = [];
        let refreshTimer = null;

        let viewerFrames = [];
        let viewerIndex = 0;
        let playTimer = null;

        function escapeHtml(text) {
            return text.replace(/[&<>"']/g, function(m) {
                return ({
                    '&': '&amp;',
                    '<': '&lt;',
                    '>': '&gt;',
                    '"': '&quot;',
                    "'": '&#039;'
                })[m];
            });
        }

        function formatPercent(v) {
            if (v === null || v === undefined) return "Unknown";
            return v.toFixed(2) + "%";
        }

        function renderStats(items) {
            const totalJobs = items.length;
            const completed = items.filter(x => x.total !== null && x.rendered >= x.total).length;
            const inProgress = items.filter(x => x.total !== null && x.rendered < x.total).length;
            const unknown = items.filter(x => x.total === null).length;

            document.getElementById("statJobs").textContent = totalJobs;
            document.getElementById("statCompleted").textContent = completed;
            document.getElementById("statInProgress").textContent = inProgress;
            document.getElementById("statUnknown").textContent = unknown;
        }

        function renderTable(items) {
            const tbody = document.getElementById("tableBody");
            tbody.innerHTML = "";

            for (const item of items) {
                const tr = document.createElement("tr");
                tr.className = item.frames.length ? "clickable" : "";

                if (item.frames.length) {
                    tr.onclick = () => openViewer(item);
                }

                const totalText = item.total === null ? "Unknown" : item.total;
                const remainingText = item.remaining === null ? "Unknown" : item.remaining;
                const progressWidth = item.percent === null ? 0 : Math.max(0, Math.min(100, item.percent));

                tr.innerHTML = `
                    <td>${escapeHtml(item.character)}</td>
                    <td>${escapeHtml(item.motion)}</td>
                    <td><span class="badge">${escapeHtml(item.camera)}</span></td>
                    <td>${item.rendered}</td>
                    <td>${totalText}</td>
                    <td>${remainingText}</td>
                    <td>
                        <div class="progress-wrap">
                            <div class="progress-bar">
                                <div class="progress-fill" style="width:${progressWidth}%"></div>
                            </div>
                            <div class="small">${formatPercent(item.percent)}</div>
                        </div>
                    </td>
                    <td>${item.first_frame ? "Open viewer" : '<span class="muted">No frames</span>'}</td>
                `;
                tbody.appendChild(tr);
            }
        }

        function applyFilter() {
            const q = document.getElementById("searchBox").value.trim().toLowerCase();
            if (!q) {
                filteredItems = allItems;
            } else {
                filteredItems = allItems.filter(item => {
                    const text = `${item.character} ${item.motion} ${item.camera}`.toLowerCase();
                    return text.includes(q);
                });
            }
            renderStats(filteredItems);
            renderTable(filteredItems);
        }

        async function refreshData() {
            try {
                const res = await fetch("/api/progress");
                const data = await res.json();
                allItems = data.items || [];
                applyFilter();
            } catch (err) {
                console.error("Failed to refresh data:", err);
            }
        }

        function restartAutoRefresh() {
            if (refreshTimer) {
                clearInterval(refreshTimer);
                refreshTimer = null;
            }
            const interval = parseInt(document.getElementById("refreshSelect").value, 10);
            if (interval > 0) {
                refreshTimer = setInterval(refreshData, interval);
            }
        }

        function openViewer(item) {
            viewerFrames = item.frames || [];
            viewerIndex = 0;

            document.getElementById("viewerTitle").textContent =
                `${item.character} / ${item.motion} / ${item.camera}`;
            document.getElementById("viewerMeta").textContent =
                `${item.rendered} rendered frames` + (item.total !== null ? ` / expected ${item.total}` : "");

            const slider = document.getElementById("frameSlider");
            slider.min = 0;
            slider.max = Math.max(0, viewerFrames.length - 1);
            slider.value = 0;

            document.getElementById("viewerModal").style.display = "block";
            showFrame(0);
        }

        function closeViewer(event) {
            if (event && event.target && event.target.id !== "viewerModal") return;
            stopPlay();
            document.getElementById("viewerModal").style.display = "none";
        }

        function showFrame(idx) {
            if (!viewerFrames.length) return;
            viewerIndex = Math.max(0, Math.min(idx, viewerFrames.length - 1));

            const frame = viewerFrames[viewerIndex];
            document.getElementById("viewerImage").src = "/image?path=" + encodeURIComponent(frame.rel_path);
            document.getElementById("frameSlider").value = viewerIndex;
            document.getElementById("frameInfo").textContent =
                `Frame ${viewerIndex + 1} / ${viewerFrames.length} (file: ${frame.name})`;
        }

        function prevFrame() {
            showFrame(viewerIndex - 1);
        }

        function nextFrame() {
            showFrame((viewerIndex + 1) % viewerFrames.length);
        }

        function onSliderChange() {
            const idx = parseInt(document.getElementById("frameSlider").value, 10);
            showFrame(idx);
        }

        function stopPlay() {
            if (playTimer) {
                clearInterval(playTimer);
                playTimer = null;
            }
            document.getElementById("playBtn").textContent = "Play";
        }

        function togglePlay() {
            if (playTimer) {
                stopPlay();
                return;
            }
            const fps = Math.max(1, Math.min(60, parseInt(document.getElementById("fpsInput").value, 10) || 12));
            const interval = Math.round(1000 / fps);
            playTimer = setInterval(() => {
                if (!viewerFrames.length) return;
                nextFrame();
            }, interval);
            document.getElementById("playBtn").textContent = "Pause";
        }

        document.getElementById("searchBox").addEventListener("input", applyFilter);
        document.getElementById("refreshSelect").addEventListener("change", restartAutoRefresh);

        refreshData();
        restartAutoRefresh();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, root_dir=str(ROOT_DIR))


@app.route("/api/progress")
def api_progress():
    items = collect_progress(ROOT_DIR)
    return jsonify({"root_dir": str(ROOT_DIR), "items": items})


@app.route("/image")
def image():
    rel_path = request.args.get("path", "").strip()
    if not rel_path:
        abort(400, "Missing path")

    try:
        img_path = safe_resolve_under_root(rel_path)
    except ValueError:
        abort(400, "Invalid path")

    if not img_path.exists() or not img_path.is_file():
        abort(404, "Image not found")

    return send_file(img_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(ROOT_DIR), help="Root render_fbx directory")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ROOT_DIR = Path(args.root).resolve()
    print(f"Using ROOT_DIR = {ROOT_DIR}")
    app.run(host=args.host, port=args.port, debug=args.debug)