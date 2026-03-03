# Parallel Rendering Guide

## How It Works

Your script supports **true parallel rendering** by spawning multiple Blender processes. Each process runs independently and renders a subset of the work.

### Why Multiple Blender Processes?

You **cannot** use Python's `multiprocessing` within a single Blender process because:
- Blender's rendering engine is not thread-safe
- The `bpy` module is tied to a single Blender instance
- Each render operation needs exclusive access to Blender's context

The solution: **spawn multiple Blender subprocesses**, each handling a portion of the work.

## Usage Examples

### Example 1: Basic Parallel Rendering (5 cameras in parallel)

```bash
blender --background --python render_fbx.py -- \
    --fbx_path ./mixamo_fbx/character.fbx \
    --out_dir ./output/ \
    --n_cam 5 \
    --blender_exe /usr/bin/blender \
    --render_engine cycles \
    --render_samples 32 \
    --use_gpu
```

This will spawn **5 parallel Blender processes**, one per camera.

### Example 2: Parallel Cameras + Frame Chunks

If you have a long animation and want even more parallelization:

```bash
blender --background --python render_fbx.py -- \
    --fbx_path ./mixamo_fbx/character.fbx \
    --out_dir ./output/ \
    --n_cam 5 \
    --n_frame_workers 4 \
    --blender_exe /usr/bin/blender \
    --render_engine cycles \
    --render_samples 32 \
    --use_gpu
```

This spawns **5 cameras × 4 frame workers = 20 parallel Blender processes**. Each process renders 1/4 of the frames for one camera.

### Example 3: Control Parallelism

Limit concurrent processes to avoid overwhelming your system:

```bash
blender --background --python render_fbx.py -- \
    --fbx_path ./mixamo_fbx/character.fbx \
    --out_dir ./output/ \
    --n_cam 8 \
    --n_frame_workers 2 \
    --n_workers 8 \
    --blender_exe /usr/bin/blender \
    --render_engine cycles \
    --use_gpu
```

This creates **16 total jobs** (8 cams × 2 frame workers) but only runs **8 at a time**.

### Example 4: Fast Preview (Cycles with Low Samples)

```bash
blender --background --python render_fbx.py -- \
    --fbx_path ./mixamo_fbx/character.fbx \
    --out_dir ./output/ \
    --n_cam 4 \
    --blender_exe /usr/bin/blender \
    --render_engine cycles \
    --render_samples 16 \
    --use_gpu
```

For fast previews, use Cycles with low samples (16-32) and GPU. This is faster than Eevee in headless mode.

**Note:** If you're rendering interactively (not in background), you can use Eevee:
```bash
# Only for interactive/GUI mode (no --background flag)
blender --python render_fbx.py -- \
    --fbx_path ./mixamo_fbx/character.fbx \
    --render_engine eevee \
    --render_samples 8
```

### Example 5: High Quality (Cycles + More Samples)

```bash
blender --background --python render_fbx.py -- \
    --fbx_path ./mixamo_fbx/character.fbx \
    --out_dir ./output/ \
    --n_cam 4 \
    --blender_exe /usr/bin/blender \
    --render_engine cycles \
    --render_samples 128 \
    --use_gpu
```

## Understanding the Arguments

### Required for Parallel Mode

- `--blender_exe`: Path to Blender executable (e.g., `/usr/bin/blender`, `C:\Program Files\Blender Foundation\Blender\blender.exe`)
  - Without this, the script runs in sequential mode (single Blender process)

### Parallelism Control

- `--n_cam`: Number of camera viewpoints (default: 5)
  - Each camera renders a full 360° view of the character
  - More cameras = more viewpoints = more parallel jobs

- `--n_frame_workers`: Split frames into chunks (default: 1)
  - 1 = each camera renders all frames sequentially
  - 2 = split frames in half, run 2 jobs per camera
  - 4 = split frames into quarters, run 4 jobs per camera

- `--n_workers`: Max concurrent Blender processes (default: n_cam × n_frame_workers)
  - Limits how many Blender instances run simultaneously
  - Set this based on your CPU/GPU capacity

### Rendering Quality

- `--render_engine`: `eevee` (fast) or `cycles` (high quality)
  - Eevee: Real-time engine, very fast, good for previews
  - Cycles: Ray-tracing engine, slower, photorealistic

- `--render_samples`: Number of samples per pixel (default: 16)
  - Lower = faster but noisier
  - Higher = slower but cleaner
  - Eevee: 8-32 is usually enough
  - Cycles: 64-256 for good quality

- `--use_gpu`: Enable GPU acceleration
  - Works with both Eevee and Cycles
  - Significantly faster if you have a compatible GPU

## Verifying Output

After rendering, use the verification script:

```bash
python verify_parallel_render.py output/character_name/ \
    --n_cams 5 \
    --n_frames 240 \
    --check_json
```

This checks:
- All expected frames were rendered
- No missing frames
- Skeleton JSON files exist (if applicable)

## Output Structure

```
output/
└── character_name/
    ├── cam_0/
    │   ├── frame0000.png
    │   ├── frame0001.png
    │   ├── ...
    │   └── skeleton_cam_0.json  # only if n_frame_workers=1
    ├── cam_1/
    │   ├── frame0000.png
    │   ├── ...
    └── cam_4/
        └── ...
```

## Performance Tips

### 1. GPU Acceleration

Always use `--use_gpu` if you have a compatible GPU. It's typically 5-10× faster.

Check GPU support:
```bash
blender --background --python-expr "import bpy; prefs = bpy.context.preferences.addons['cycles'].preferences; prefs.get_devices(); print([(d.name, d.type) for d in prefs.devices])"
```

### 2. Render Engine Choice

**Important: Eevee in Background Mode**
Eevee requires an OpenGL context, which can cause EGL errors in headless/background rendering:
```
EGL Error (0x3009): EGL_BAD_MATCH
```

The script now uses **software OpenGL** (llvmpipe) for Eevee in background mode, but this can be **slower than Cycles with GPU**!

**Use Eevee for:**
- Interactive rendering (Blender GUI)
- When you have a display/X server available
- Quick previews on a desktop machine

**Use Cycles for:**
- **Parallel/headless rendering (RECOMMENDED)**
- Server environments without display
- When you need photorealistic lighting
- Background rendering with GPU acceleration

### 3. Optimal Worker Count

**Rule of thumb:**
- CPU cores: Set `--n_workers` to number of **physical cores** (not threads)
- GPU rendering: Set to number of GPUs (usually 1)
- Memory: Each Blender process uses ~2-4GB RAM

Example for 8-core CPU:
```bash
--n_cam 8 --n_frame_workers 1 --n_workers 8
```

### 4. Frame Workers

Only use `--n_frame_workers > 1` if:
- Animation is very long (>500 frames)
- You want maximum parallelism
- You have enough CPU/GPU capacity

**Note:** Frame workers don't create skeleton JSON files. If you need JSON data, use `--n_frame_workers 1`.

## Troubleshooting

### EGL Error with Eevee

```
EGL Error (0x3009): EGL_BAD_MATCH: Arguments are inconsistent
```

**Cause:** Eevee requires OpenGL context, which doesn't work properly in headless/background mode.

**Solutions:**
1. **Use Cycles instead (RECOMMENDED):**
   ```bash
   --render_engine cycles --use_gpu
   ```
   Cycles with GPU is often faster than Eevee in headless mode anyway!

2. **Use Xvfb (virtual display):**
   ```bash
   xvfb-run -a blender --background --python render_fbx.py -- \
       --render_engine eevee ...
   ```

3. **Render without --background flag** (if you have a display):
   ```bash
   blender --python render_fbx.py -- --render_engine eevee ...
   ```

### "Subprocess failed"

Check the error output. Common issues:
- Blender path incorrect: verify `--blender_exe` points to actual Blender
- Out of memory: reduce `--n_workers`
- GPU issues: try without `--use_gpu`

### Slow rendering

- Use Cycles with GPU (faster than Eevee in headless mode)
- Reduce `--render_samples` (try 16-32 for previews)
- Enable GPU: `--use_gpu`
- Reduce resolution: `--img_width 256 --img_height 256`
- Increase parallelism: `--n_frame_workers 2` for long animations

### Missing frames

Run verification:
```bash
python verify_parallel_render.py output/character_name/ --n_cams 5 --n_frames 240
```

Then re-render missing sections if needed.

## Finding Your Blender Path

**Linux:**
```bash
which blender
# Usually: /usr/bin/blender or /snap/bin/blender
```

**macOS:**
```bash
which blender
# or: /Applications/Blender.app/Contents/MacOS/Blender
```

**Windows:**
```cmd
where blender
# Usually: C:\Program Files\Blender Foundation\Blender\blender.exe
```

## Example Workflow

1. **Quick test** (single camera, low quality):
```bash
blender --background --python render_fbx.py -- \
    --fbx_path ./test.fbx \
    --out_dir ./test_output/ \
    --n_cam 1 \
    --render_engine cycles \
    --render_samples 16 \
    --use_gpu \
    --blender_exe $(which blender)
```

2. **Full render** (all cameras, parallel):
```bash
blender --background --python render_fbx.py -- \
    --fbx_path ./character.fbx \
    --out_dir ./output/ \
    --n_cam 8 \
    --render_engine cycles \
    --render_samples 32 \
    --use_gpu \
    --blender_exe $(which blender)
```

3. **Verify**:
```bash
python verify_parallel_render.py output/character/ --n_cams 8 --n_frames <YOUR_FRAME_COUNT>
```

## Summary

✅ **DO use:** `--blender_exe` for parallel rendering
✅ **DO use:** `--use_gpu` for speed
✅ **DO use:** Cycles for parallel/headless rendering (recommended)
✅ **DO use:** `--n_cam` for more viewpoints
✅ **DO use:** Low samples (16-32) for fast previews

⚠️ **CAREFUL with:** `--n_workers` (don't exceed CPU cores)
⚠️ **CAREFUL with:** `--n_frame_workers` (only for long animations)
⚠️ **CAREFUL with:** Eevee in background mode (causes EGL errors, use Cycles instead)

❌ **DON'T:** Try to use multiprocessing inside Blender (won't work)
❌ **DON'T:** Set too many workers (will slow down or crash)
❌ **DON'T:** Use Eevee for parallel rendering (use Cycles with GPU instead)
