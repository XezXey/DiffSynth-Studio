# Render Monitor

A Flask web app to monitor FBX render job progress.

## Directory Structure Expected

```
<RENDER_ROOT>/
  <character_name>/
    <motion_name>/
      cam_<id>/
        skeleton_cam-<id>.json   ← must contain "frame_range": [start, end]
        frame0000.png
        frame0001.png
        ...
```

---

## Quick Start (Flask only, no Nginx)

```bash
pip install flask

# Point at your render root
RENDER_ROOT=/data/mint/Motion_Dataset/Mixamo/render_fbx python app.py
```

Open **http://localhost:5000**

---

## Fast Setup with Nginx (recommended for large frame sets)

Nginx serves PNG frames directly using the kernel's `sendfile()` syscall —
bypassing Python entirely. This is 10–50x faster per frame than Flask.

### 1. Install Nginx

```bash
sudo apt install nginx      # Ubuntu/Debian
# or
sudo yum install nginx      # RHEL/CentOS
```

### 2. Install the config

```bash
# Edit nginx.conf first — set the alias path to match your RENDER_ROOT
sudo cp nginx.conf /etc/nginx/sites-available/render_monitor
sudo ln -s /etc/nginx/sites-available/render_monitor /etc/nginx/sites-enabled/render_monitor

# Remove the default site if needed
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t                       # test config
sudo systemctl reload nginx
```

### 3. Run Flask with Nginx mode enabled

```bash
USE_NGINX=1 RENDER_ROOT=/data/mint/Motion_Dataset/Mixamo/render_fbx python app.py
```

Open **http://localhost** (port 80 via Nginx, which proxies API/HTML to Flask on 5000).

### How it works

```
Browser → Nginx :80
  ├─ /renders/.../*.png  →  served directly by Nginx (sendfile, zero Python)
  ├─ /api/...            →  proxied to Flask :5000
  └─ /job/...            →  proxied to Flask :5000
```

---

## Configuration

| Variable               | Default                                       | Description                          |
|------------------------|-----------------------------------------------|--------------------------------------|
| `RENDER_ROOT`          | `/data/mint/Motion_Dataset/Mixamo/render_fbx` | Root render directory                |
| `SCAN_INTERVAL`        | `30`                                          | Seconds between background rescans  |
| `USE_NGINX`            | `0`                                           | Set to `1` when Nginx is configured  |
| `NGINX_INTERNAL_PREFIX`| `/renders`                                    | Must match nginx.conf location block |
| `USE_XSENDFILE`        | `0`                                           | Set to `1` for Apache/mod_wsgi       |

---

## Performance options (fastest → slowest)

| Setup | Frame serving | Notes |
|---|---|---|
| **Nginx + sendfile** | ~1ms/frame | Best. Kernel-level, zero Python overhead |
| **Apache + X-Sendfile** | ~2ms/frame | Set `USE_XSENDFILE=1` |
| **Flask `send_from_directory`** | ~10–30ms/frame | Default, no extra setup needed |

---

## Files

```
render_monitor/
├── app.py            ← Flask backend + background cache
├── nginx.conf        ← Nginx config for fast frame serving
├── requirements.txt
├── seed_demo.py      ← test data generator
└── templates/
    ├── index.html    ← dashboard (auto-refresh, filter, search)
    └── detail.html   ← image sequence player (lazy loaded)
```
