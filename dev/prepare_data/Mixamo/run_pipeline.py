"""
run_pipeline.py
───────────────
End-to-end preprocessing pipeline for Mixamo → SkelAg training data.

Steps
  1  run_multiple_chars.py             – render FBX files per character
  2  gen_data_format_multiple_chars.py – reformat raw render output
  3  chunk_data_multiple_chars.py      – split sequences into N-frame chunks
  4  combined_output.py                – merge all character folders → all/
  5  precompute_features.py            – precompute VAE latents + DIT features

Usage (working directory must be .../DiffSynth-Studio/dev/prepare_data/Mixamo)
  python run_pipeline.py \
      --input_dir ./testset_motion/ \
      --render_output_dir /data2/mint/Motion_Dataset/Mixamo/testset_raw \
      --format_output_dir /data2/mint/Motion_Dataset/Mixamo/testset_fmt \
      --chunk_output_dir  /data2/mint/Motion_Dataset/Mixamo/testset_5f \
      --host_prefix /host \
      --vae_output_path /host/data2/mint/Motion_Dataset/SkelAg/testset_5f/latents \
      --wan_output_path /host/data2/mint/Motion_Dataset/SkelAg/testset_5f/dit_features \
      --use_gpu --run_projection --skip_plot_map --only_body_joints
"""

import argparse
import os
import subprocess
import sys

# ── locate repo root (3 levels up from this file) ─────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "../../../"))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run(cmd: str, *, cwd: str | None = None, step: str) -> None:
    """Run *cmd* as a shell command; exit the whole pipeline on failure."""
    print(f"\n{'='*70}")
    print(f"  STEP: {step}")
    print(f"  CWD : {cwd or os.getcwd()}")
    print(f"  CMD : {cmd}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    print(f"\n[OK] Step '{step}' completed.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full Mixamo → SkelAg preprocessing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── global skip flags ─────────────────────────────────────────────────────
    skip = p.add_argument_group("skip flags  (skip individual steps)")
    skip.add_argument("--skip_render",     action="store_true", default=False, help="Skip step 1: render FBX files.")
    skip.add_argument("--skip_format",     action="store_true", default=False, help="Skip step 2: reformat render output.")
    skip.add_argument("--skip_chunk",      action="store_true", default=False, help="Skip step 3: chunk into N-frame clips.")
    skip.add_argument("--skip_combine",    action="store_true", default=False, help="Skip step 4: combine character folders.")
    skip.add_argument("--skip_precompute", action="store_true", default=False, help="Skip step 5: precompute VAE/DIT features.")

    # ── paths ─────────────────────────────────────────────────────────────────
    paths = p.add_argument_group("paths")
    paths.add_argument("--input_dir", required=True,
                       help="Input directory with character subdirs containing FBX files. (step 1)")
    paths.add_argument("--render_output_dir", required=True,
                       help="Output of step 1 / input to step 2 (raw render data).")
    paths.add_argument("--format_output_dir", required=True,
                       help="Output of step 2 / input to step 3 (formatted data).")
    paths.add_argument("--chunk_output_dir", required=True,
                       help="Output of step 3 / input to steps 4-5 (chunked data).")
    paths.add_argument("--host_prefix", default="",
                       help="Path prefix prepended to chunk_output_dir for steps 4 and 5 "
                            "(e.g. '/host' when running inside a Docker container that mounts "
                            "/data2 at /host/data2). Leave empty if no prefix is needed.")
    paths.add_argument("--vae_output_path", required=True,
                       help="Output dir for precomputed VAE latents (step 5).")
    paths.add_argument("--wan_output_path", required=True,
                       help="Output dir for precomputed DIT features (step 5).")

    # ── step 1: render ────────────────────────────────────────────────────────
    s1 = p.add_argument_group("step 1 – render  (run_multiple_chars.py)")
    s1.add_argument("--max_log_lines",    type=int,   default=30)
    s1.add_argument("--n_cam",            type=int,   default=1)
    s1.add_argument("--follow_bone",      type=str,   default="mixamorig:Hips")
    s1.add_argument("--cam_height",       type=float, default=3.0)
    s1.add_argument("--cam_radius",       type=float, default=4.5)
    s1.add_argument("--img_width",        type=int,   default=1280)
    s1.add_argument("--img_height",       type=int,   default=720)
    s1.add_argument("--use_gpu",          action="store_true")
    s1.add_argument("--run_blender",      action="store_true")
    s1.add_argument("--run_projection",   action="store_true")
    s1.add_argument("--only_body_joints", action="store_true")
    s1.add_argument("--skip_plot_map",    action="store_true")

    # ── step 3: chunk ─────────────────────────────────────────────────────────
    s3 = p.add_argument_group("step 3 – chunk  (chunk_data_multiple_chars.py)")
    s3.add_argument("--n_frames", type=int, default=5,
                    help="Frames per chunk.")
    s3.add_argument("--overlap",  type=int, default=0,
                    help="Overlapping frames between consecutive chunks.")
    s3.add_argument("--metadata_name", type=str, default="metadata_front-view.csv",
                    help="Metadata CSV filename to search for inside each character folder.")

    # ── step 4: combine ───────────────────────────────────────────────────────
    s4 = p.add_argument_group("step 4 – combine  (combined_output.py)")
    s4.add_argument("--metadata_to_combined", type=str, default=None,
                    help="Filename to collect and concatenate across characters into one CSV. "
                         "Auto-derived from --metadata_name + --n_frames when omitted "
                         "(e.g. 'metadata_front_view_5frames.csv').")

    # ── step 5: precompute ────────────────────────────────────────────────────
    s5 = p.add_argument_group("step 5 – precompute  (precompute_features.py)")
    s5.add_argument("--wan_height",      type=int, default=320)
    s5.add_argument("--wan_width",       type=int, default=640)
    s5.add_argument("--num_frames",  type=int, default=5)
    s5.add_argument("--mode",
                    choices=["data_process", "data_process_with_wan", "both"],
                    default="both")
    s5.add_argument("--gpu_id",      type=int, default=0)
    s5.add_argument("--dataset_repeat_vae", type=int, default=1)
    s5.add_argument("--dataset_repeat_wan", type=int, default=1)
    s5.add_argument("--preferred_timestep_id", type=int, default=-20)
    s5.add_argument("--model_id_with_origin_paths",
                    default=("Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,"
                             "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,"
                             "Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"))
    s5.add_argument("--tokenizer_path",
                    default=("/host/ist/ist-share/vision/huggingface_hub/"
                             "Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl/"))
    s5.add_argument("--data_file_keys",  default="video,motion")
    s5.add_argument("--offload_models",
                    default="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors")
    s5.add_argument("--extra_inputs",    default="input_image")
    s5.add_argument("--use_gradient_checkpointing_offload", action="store_true")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Helper: derive the combined-metadata filename
# ══════════════════════════════════════════════════════════════════════════════

def _combined_meta_name(a: argparse.Namespace) -> str:
    if a.metadata_to_combined:
        return a.metadata_to_combined
    stem = os.path.splitext(a.metadata_name)[0]       # "metadata_front_view"
    return f"{stem}_{a.n_frames}frames.csv"            # "metadata_front_view_5frames.csv"


def _host_chunk_dir(a: argparse.Namespace) -> str:
    """chunk_output_dir with optional host prefix, no trailing slash."""
    prefix = a.host_prefix.rstrip("/") if a.host_prefix else ""
    return prefix + a.chunk_output_dir.rstrip("/")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ══════════════════════════════════════════════════════════════════════════════

def step1_render(a: argparse.Namespace) -> None:
    """Step 1 – render FBX files through Blender / 2-D projection."""
    flags = ""
    if a.use_gpu:          flags += " --use_gpu"
    if a.run_blender:      flags += " --run_blender"
    if a.run_projection:   flags += " --run_projection"
    if a.only_body_joints: flags += " --only_body_joints"
    if a.skip_plot_map:    flags += " --skip_plot_map"
    if a.host_prefix == "/host": flags += " --blender_bin /host/ist/users/puntawatp/Dev/SkelAg/Blender/blender-5.0.0-linux-x64/blender"

    cmd = (
        f'python run_multiple_chars.py'
        f' --input_dir "{a.input_dir}"'
        f' --output_dir "{a.render_output_dir}"'
        f' --max_log_lines {a.max_log_lines}'
        f' --n_cam {a.n_cam}'
        f' --follow_bone {a.follow_bone}'
        f' --cam_height {a.cam_height}'
        f' --cam_radius {a.cam_radius}'
        f' --img_width {a.img_width}'
        f' --img_height {a.img_height}'
        f'{flags}'
    )
    _run(cmd, cwd=_HERE, step="1 – render FBX files")
    # exit()


def step2_format(a: argparse.Namespace) -> None:
    """Step 2 – reformat raw render output into dataset structure."""
    cmd = (
        f'python gen_data_format_multiple_chars.py'
        f' --data_path "{a.render_output_dir}"'
        f' --output_path "{a.format_output_dir}"'
    )
    _run(cmd, cwd=_HERE, step="2 – gen dataset format")


def step3_chunk(a: argparse.Namespace) -> None:
    """Step 3 – split long sequences into N-frame chunks."""
    # metadata_example just needs the right basename; point into format_output_dir
    # so chunk_data_multiple_chars.py can locate it via os.walk.
    metadata_example = os.path.join(a.format_output_dir, a.metadata_name)

    cmd = (
        f'python chunk_data_multiple_chars.py'
        f' --input_dir "{a.format_output_dir}"'
        f' --output_dir "{a.chunk_output_dir}"'
        f' --n_frames {a.n_frames}'
        f' --overlap {a.overlap}'
        f' --metadata_example "{metadata_example}"'
    )
    _run(cmd, cwd=_HERE, step="3 – chunk into N-frame clips")


def step4_combine(a: argparse.Namespace) -> None:
    """Step 4 – merge all character subfolders into a single 'all/' directory."""
    # host_chunk = _host_chunk_dir(a) + "/"   # trailing slash expected by combined_output.py

    cmd = (
        f'python combined_output.py'
        f' --input_path "{a.chunk_output_dir}"'
        f' --metadata_to_combined "{_combined_meta_name(a)}"'
    )
    _run(cmd, cwd=_HERE, step="4 – combine character outputs")


def step5_precompute(a: argparse.Namespace) -> None:
    """Step 5 – precompute VAE latents and DIT features."""
    # host_chunk            = _host_chunk_dir(a)
    host_chunk            = a.chunk_output_dir
    combined_meta         = _combined_meta_name(a)
    dataset_base_path     = f"{host_chunk}/all/"
    dataset_metadata_path = f"{host_chunk}/all/all_{combined_meta}"

    gc_flag = " --use_gradient_checkpointing_offload" if a.use_gradient_checkpointing_offload else ""

    cmd = (
        f'python examples/wanvideo/my_scripts/precomputed_dit_features/precompute_features.py'
        f' --dataset_base_path "{dataset_base_path}"'
        f' --dataset_metadata_path "{dataset_metadata_path}"'
        f' --height {a.wan_height}'
        f' --width {a.wan_width}'
        f' --num_frames {a.num_frames}'
        f' --dataset_repeat_vae {a.dataset_repeat_vae}'
        f' --dataset_repeat_wan {a.dataset_repeat_wan}'
        f' --model_id_with_origin_paths "{a.model_id_with_origin_paths}"'
        f' --tokenizer_path "{a.tokenizer_path}"'
        f' --mode "{a.mode}"'
        f' --output_path_vae "{a.vae_output_path}"'
        f' --output_path_wan "{a.wan_output_path}"'
        f' --data_file_keys "{a.data_file_keys}"'
        f' --offload_models "{a.offload_models}"'
        f' --extra_inputs "{a.extra_inputs}"'
        f' --preferred_timestep_id {a.preferred_timestep_id}'
        f' --gpu_id {a.gpu_id}'
        f'{gc_flag}'
    )
    # precompute_features.py resolves internal paths relative to the repo root.
    _run(cmd, cwd=_REPO_ROOT, step="5 – precompute VAE latents + DIT features")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    a = _parse_args()

    completed = []

    if not a.skip_render:
        step1_render(a)
        completed.append("1 – render")

    if not a.skip_format:
        step2_format(a)
        completed.append("2 – format")

    if not a.skip_chunk:
        step3_chunk(a)
        completed.append("3 – chunk")

    if not a.skip_combine:
        step4_combine(a)
        completed.append("4 – combine")

    if not a.skip_precompute:
        step5_precompute(a)
        completed.append("5 – precompute")
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    for s in completed:
        print(f"    [OK] Step {s}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()