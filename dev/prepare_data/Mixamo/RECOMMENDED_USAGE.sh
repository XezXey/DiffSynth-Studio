#!/bin/bash
# Recommended usage for parallel rendering with DiffSynth-Studio
# This avoids the EGL errors with Eevee by using Cycles instead

# Find your Blender executable
BLENDER_EXE=$(which blender)
# Or set it manually:
# BLENDER_EXE="/usr/bin/blender"
# BLENDER_EXE="/snap/bin/blender"
# BLENDER_EXE="/Applications/Blender.app/Contents/MacOS/Blender"

# Your FBX file
FBX_PATH="./mixamo_fbx/character.fbx"
# Or a directory of FBX files:
# FBX_PATH="./mixamo_fbx/"

# Output directory
OUT_DIR="./output/"

# === RECOMMENDED: Fast parallel rendering with Cycles ===
# This uses Cycles with GPU, which is FASTER than Eevee in headless mode
# and doesn't have EGL errors

blender --background --python render_fbx.py -- \
    --fbx_path "$FBX_PATH" \
    --out_dir "$OUT_DIR" \
    --n_cam 5 \
    --render_engine cycles \
    --render_samples 32 \
    --use_gpu \
    --blender_exe "$BLENDER_EXE"

# Parameters explained:
# --n_cam 5               : 5 camera viewpoints (360° coverage)
# --render_engine cycles  : Use Cycles (works in headless, GPU accelerated)
# --render_samples 32     : 32 samples (good quality, reasonable speed)
# --use_gpu               : Enable GPU acceleration
# --blender_exe           : Path to blender (enables parallel rendering)

# === For even faster previews ===
# Reduce samples to 16:
# --render_samples 16

# === For maximum parallelism (long animations) ===
# Split frames into chunks:
# --n_frame_workers 4 --n_workers 8

# === After rendering, verify the output ===
# python verify_parallel_render.py "$OUT_DIR/character_name/" \
#     --n_cams 5 \
#     --n_frames 240 \
#     --check_json
