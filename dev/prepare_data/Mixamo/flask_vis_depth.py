import argparse
import re
from pathlib import Path

from flask import Flask, render_template_string, send_from_directory, abort

app = Flask(__name__)

DATA_ROOT = None  # will be set from --path

FRAME_REGEX = re.compile(r"frame(\d{4})\.png$")


def scan_sequences():
    """
    Recursively scan DATA_ROOT for directories that contain frame%04d.png.
    Returns a list of dicts:
      {
        "name": relative path used in URL (e.g. "walking", "running/sub1"),
        "id": safe id for HTML (slashes replaced),
        "num_frames": N,
        "first_index": first frame number,
        "last_index": last frame number,
      }
    """
    if DATA_ROOT is None:
        return []

    sequences = []
    root = Path(DATA_ROOT)
    if not root.exists():
        return []

    # All directories that contain at least one frame*.png
    candidate_dirs = sorted({p.parent for p in root.rglob("frame*.png")})

    for seq_dir in candidate_dirs:
        frame_indices = []
        for img_path in seq_dir.glob("frame*.png"):
            m = FRAME_REGEX.match(img_path.name)
            if m:
                frame_indices.append(int(m.group(1)))

        if not frame_indices:
            continue

        frame_indices.sort()
        first_idx = frame_indices[0]
        last_idx = frame_indices[-1]
        num_frames = len(frame_indices)

        rel_path = seq_dir.relative_to(root).as_posix()  # "walking" or "running/sub1"
        safe_id = rel_path.replace("/", "__")

        sequences.append(
            {
                "name": rel_path,
                "id": safe_id,
                "num_frames": num_frames,
                "first_index": first_idx,
                "last_index": last_idx,
            }
        )

    return sequences


@app.route("/")
def index():
    sequences = scan_sequences()

    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Frame Sequences Viewer</title>
    <style>
    body {
        font-family: sans-serif;
        margin: 20px;
        background: #ffffff;
        color: #000000;
    }
    .samples-container {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: flex-start;
    }
    .sequence {
        width: 260px;  /* adjust to control how many per row */
        font-size: 13px;
    }
    .sequence h2 {
        margin: 0 0 6px 0;
        font-weight: 600;
        font-size: 13px;
        word-break: break-all; /* for long nested paths */
    }
    .image-wrapper {
        position: relative;
        display: inline-block;
        width: 100%;
    }
    .frame-image,
    .overlay-image {
        display: block;
        width: 100%;
        height: auto;
    }
    .overlay-image {
        position: absolute;
        top: 0;
        left: 0;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.1s linear;
    }
    /* On hover, fade in overlay at ~50% */
    .image-wrapper:hover .overlay-image {
        opacity: 0.5;
    }
    .controls {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 4px 0;
    }
    input[type=range] {
        width: 160px;
    }
    button {
        padding: 4px 10px;
        border-radius: 4px;
        border: none;
        cursor: pointer;
        background: #4caf50;
        color: white;
        font-size: 0.8rem;
    }
    button.paused {
        background: #e53935;
    }
    .info {
        font-size: 0.8rem;
        color: #333333;
    }
    </style>
</head>
<body>
    {% if not sequences %}
        <p>No sequences found in the specified path.</p>
    {% else %}
    <div class="samples-container">
        {% for seq in sequences %}
        <div class="sequence" id="seq-{{ seq.id }}">
            <h2>{{ seq.name }}</h2>
            <div class="image-wrapper">
                <img id="img-{{ seq.id }}" class="frame-image"
                     src="/frame/{{ seq.name }}/{{ seq.first_index }}">
                <img id="overlay-{{ seq.id }}" class="overlay-image"
                     src="/depth/{{ seq.name }}/{{ seq.first_index }}">
            </div>
            <div class="controls">
                <button id="play-{{ seq.id }}">Play</button>
                <input type="range" id="slider-{{ seq.id }}"
                       min="0" max="{{ seq.num_frames - 1 }}" value="0">
            </div>
            <div class="info">
                <span id="label-{{ seq.id }}">
                    Frame: {{ seq.first_index }} / {{ seq.last_index }}
                </span>
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

<script>
const sequences={{ sequences|tojson }},FPS=30,INTERVAL=1000/FPS,players={};
function setupSequence(seq){
    const name=seq.name, id=seq.id;
    const img=document.getElementById("img-"+id),
          overlay=document.getElementById("overlay-"+id),
          slider=document.getElementById("slider-"+id),
          label=document.getElementById("label-"+id),
          btn=document.getElementById("play-"+id);
    if(!img||!slider||!label||!btn||!overlay)return;

    const state={
        name, id,
        numFrames:seq.num_frames,
        firstIndex:seq.first_index,
        lastIndex:seq.last_index,
        currentIdx:0,
        timerId:null,
        playing:false,
        img, overlay, slider, label, btn
    };
    players[id]=state;

    const updateImage=()=>{
        const f=state.firstIndex+state.currentIdx;
        img.src="/frame/"+encodeURIComponent(state.name)+"/"+f;
        overlay.src="/depth/"+encodeURIComponent(state.name)+"/"+f;
        label.textContent="Frame: "+f+" / "+state.lastIndex;
    };
    const setFrame=i=>{
        state.currentIdx=Math.max(0,Math.min(state.numFrames-1,i));
        slider.value=state.currentIdx;
        updateImage();
    };
    const startLoop=()=>{
        if(state.timerId!==null)clearInterval(state.timerId);
        state.timerId=setInterval(()=>{
            state.currentIdx=(state.currentIdx+1)%state.numFrames;
            slider.value=state.currentIdx;
            updateImage();
        },INTERVAL);
    };
    const play=()=>{
        if(state.playing||state.numFrames<=0)return;
        state.playing=true;
        btn.textContent="Pause";
        btn.classList.add("paused");
        startLoop();
    };
    const pause=()=>{
        state.playing=false;
        btn.textContent="Play";
        btn.classList.remove("paused");
        if(state.timerId!==null){
            clearInterval(state.timerId);
            state.timerId=null;
        }
    };

    slider.oninput=()=>{pause(); setFrame(+slider.value);};
    btn.onclick=()=>{state.playing?pause():play();};

    setFrame(0);
    play(); // autoplay
}
sequences.forEach(setupSequence);
</script>
</body>
</html>
        """,
        sequences=sequences,
    )


@app.route("/frame/<path:seq_name>/<int:frame_number>")
def serve_frame(seq_name, frame_number):
    """
    Serve frame%04d.png from DATA_ROOT/seq_name
    """
    if DATA_ROOT is None:
        abort(404)

    seq_dir = Path(DATA_ROOT) / seq_name
    if not seq_dir.is_dir():
        abort(404)

    filename = f"frame{frame_number:04d}.png"
    file_path = seq_dir / filename
    if not file_path.is_file():
        abort(404)

    return send_from_directory(seq_dir, filename)


@app.route("/depth/<path:seq_name>/<int:frame_number>")
def serve_depth(seq_name, frame_number):
    """
    Serve depth%04d.png from DATA_ROOT/seq_name (for overlay).
    If depth file is missing, this will 404; overlay just won't show.
    """
    if DATA_ROOT is None:
        abort(404)

    seq_dir = Path(DATA_ROOT) / seq_name
    if not seq_dir.is_dir():
        abort(404)

    filename = f"depth{frame_number:04d}.png"
    file_path = seq_dir / filename
    if not file_path.is_file():
        abort(404)

    return send_from_directory(seq_dir, filename)


def main():
    global DATA_ROOT

    parser = argparse.ArgumentParser(description="View frame sequences in the browser.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Root path containing sequence subfolders (e.g., ./outpath)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    DATA_ROOT = Path(args.path).resolve()
    print(f"[INFO] Using data root: {DATA_ROOT}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
