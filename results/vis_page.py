import os
from pathlib import Path
from flask import Flask, send_from_directory, render_template_string

app = Flask(__name__)

VIS_DIR = Path("./rdy_to_vis").resolve()


def stem_key(p: Path) -> str:
    stem = p.stem
    for s in [
        "_animation",
        "_with_skeleton",
        "_skeleton",
        "_video",
        "_vis",
    ]:
        if stem.endswith(s):
            return stem[:-len(s)]
    return stem


def discover_items():
    htmls = list(VIS_DIR.glob("*.html"))
    mp4s = list(VIS_DIR.glob("*.mp4"))

    items = {}

    for h in htmls:
        k = stem_key(h)
        items.setdefault(k, {})["html"] = h.name

    for v in mp4s:
        k = stem_key(v)
        items.setdefault(k, {})["mp4"] = v.name

    return dict(sorted(items.items(), key=lambda x: x[0].lower()))


@app.route("/")
def index():
    items = discover_items()

    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Visualization</title>
  <style>
    body {
      background: white;
      color: #eee;
      font-family: sans-serif;
      margin: 0;
      padding: 10px;
    }
    .item {
      margin-bottom: 40px;
      border-bottom: 1px solid #333;
      padding-bottom: 30px;
    }
    h3 {
      margin: 5px 0 10px;
      font-weight: normal;
      color: #ccc;
    }
    iframe {
      width: 100%;
      height: 720px;
      border: none;
      background: white;
      margin-bottom: 10px;
    }
    video {
      width: 100%;
      max-height: 480px;
      background: black;
    }
    .missing {
      color: #888;
      font-style: italic;
      margin-bottom: 10px;
    }
  </style>
</head>
<body>

{% for key, item in items.items() %}
  <div class="item">
    <h3>{{ key }}</h3>

    {% if item.html %}
      <iframe src="/vis/{{ item.html }}"></iframe>
    {% else %}
      <div class="missing">No HTML found</div>
    {% endif %}

    {% if item.mp4 %}
      <video muted autoplay loop>
        <source src="/vis/{{ item.mp4 }}" type="video/mp4">
      </video>
    {% else %}
      <div class="missing">No video found</div>
    {% endif %}
  </div>
{% endfor %}

</body>
</html>
        """,
        items=items,
    )


@app.route("/vis/<path:filename>")
def vis_file(filename):
    return send_from_directory(VIS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
