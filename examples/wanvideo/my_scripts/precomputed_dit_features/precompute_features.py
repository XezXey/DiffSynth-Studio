"""
precompute_features.py
──────────────────────
Automates the two-stage DIT feature precomputation pipeline.

Stages
  data_process          – precompute latents (stage 1)
  data_process_with_wan – train DIT features (stage 2)
  both                  – run stage 1 then stage 2   [default]

Display modes  (--display)
  rich     – live rolling window, no terminal flood   [default]
  textual  – full-screen TUI with scrollable log pane
  plain    – raw stdout, safe for non-TTY / CI

All display logic lives in utils/display_rich.py and utils/display_textual.py
so the same helpers can be imported by any other automation script.
"""

import argparse
import os
import sys

# ── resolve utils/ regardless of working directory ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.display_common import die, print_config, print_summary, run_command_plain


# ── display backend selector ──────────────────────────────────────────────────

def _get_runner(display: str):
    """Return the run_command callable for the requested display backend."""
    if display == "textual":
        from utils.display_textual import run_command
        return run_command
    elif display == "rich":
        from utils.display_rich import run_command
        return run_command
    else:  # plain
        return run_command_plain


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Automate DIT feature precomputation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── pipeline control ──────────────────────────────────────────────────────
    p.add_argument("--mode", default="both",
                   choices=["data_process", "data_process_with_wan", "both"],
                   help="Which stage(s) to run.")
    p.add_argument("--display", default="rich",
                   choices=["rich", "textual", "plain"],
                   help="Terminal display backend.")
    p.add_argument("--gpu_id", type=int, default=0,
                   help="CUDA device index.")

    # ── dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset_base_path",     required=True)
    p.add_argument("--dataset_metadata_path", required=True)

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model_id_with_origin_paths",
                   default=("Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,"
                            "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,"
                            "Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"))
    p.add_argument("--tokenizer_path",
                   default=("/host/ist/ist-share/vision/huggingface_hub/"
                            "Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl/"))

    # ── video ─────────────────────────────────────────────────────────────────
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--num_frames", type=int, default=5)

    # ── Mock up parameters  ────────────────────────────────────────────────
    p.add_argument("--learning_rate",  type=float, default=1e-5)
    p.add_argument("--num_epochs",     type=int,   default=1)
    p.add_argument("--save_steps",     type=int,   default=100_000)
    p.add_argument("--vis_steps",          type=int, default=100)
    p.add_argument("--log_steps",          type=int, default=50)
    p.add_argument("--n_joints",              type=int, default=25)

    # ── stage-2 inference knobs ────────────────────────────────────────────────
    p.add_argument("--dataset_repeat_vae", type=int,   default=1, help="How many times to repeat the dataset for stage 1 (precompute latents).")
    p.add_argument("--dataset_repeat_wan", type=int, default=25, help="How many times to repeat the dataset for stage 2 (computed DIT features on #N different noise).")
    p.add_argument("--preferred_timestep_id", type=int, required=True)
    p.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    p.add_argument("--fp8_models",
                   default="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors")

    # ── output / logging ──────────────────────────────────────────────────────
    p.add_argument("--output_path_vae",     required=True,
                   help="Output dir for latents (stage 1).")
    p.add_argument("--output_path_wan", required=True,
                   help="Output dir for DIT features (stage 2).")

    # ── shared misc ───────────────────────────────────────────────────────────
    p.add_argument("--data_file_keys", default="video,motion")
    p.add_argument("--offload_models",
                   default="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors")
    p.add_argument("--extra_inputs", default="input_image")

    a = p.parse_args()
    print_config(a)

    # ── stage 1: data_process ─────────────────────────────────────────────────
    if a.mode in ("data_process", "both"):
        cmd = (
            f"CUDA_VISIBLE_DEVICES={a.gpu_id}"
            f" python examples/wanvideo/my_scripts/precomputed_dit_features/precomputed_dit_features.py"
            f" --dataset_base_path {a.dataset_base_path}"
            f" --dataset_metadata_path {a.dataset_metadata_path}"
            f" --height {a.height} --width {a.width}"
            f" --dataset_repeat {a.dataset_repeat_vae}"
            f' --model_id_with_origin_paths "{a.model_id_with_origin_paths}"'
            f' --tokenizer_path "{a.tokenizer_path}"'
            f" --save_steps {a.save_steps}"
            f" --learning_rate {a.learning_rate}"
            f" --num_epochs {a.num_epochs}"
            f' --task "dit_features:data_process"'
            f' --output_path "{a.output_path_vae}"'
            f' --data_file_keys "{a.data_file_keys}"'
            f' --offload_models "{a.offload_models}"'
            f' --extra_inputs "{a.extra_inputs}"'
        )
        # How can i check whether the command succeeded or failed when using os.system? I want to print an error message and exit if it failed.
        status = os.system(cmd)
        exit_code = os.WEXITSTATUS(status)
        if exit_code != 0:
            die("Stage 1 failed.")

        # exit()
        # if not run(cmd,
        #            description="Data Process – Precompute Latents",
        #            stage_num=stage, total_stages=total):
        #     die("Stage 1 failed.")
        # stage += 1

    # ── stage 2: data_process_with_wan ────────────────────────────────────────
    if a.mode in ("data_process_with_wan", "both"):
        gc    = "--use_gradient_checkpointing_offload" if a.use_gradient_checkpointing_offload else ""
        model = a.model_id_with_origin_paths.split(",")[0]
        cmd = (
            f"CUDA_VISIBLE_DEVICES={a.gpu_id}"
            f" python examples/wanvideo/my_scripts/training_dits_features/train_Wan2.2-TI2V-5B_dits_features.py"
            f" --dataset_base_path {a.output_path_vae}"
            f" --height {a.height} --width {a.width}"
            f" --dataset_repeat {a.dataset_repeat_wan}"
            f' --model_id_with_origin_paths "{model}"'
            f' --tokenizer_path "{a.tokenizer_path}"'
            f" --save_steps {a.save_steps}"
            f" --vis_steps {a.vis_steps}"
            f" --log_steps {a.log_steps}"
            f" --learning_rate {a.learning_rate}"
            f" --num_epochs {a.num_epochs}"
            f' --task "dit_features:data_process_with_wan"'
            f' --output_path "{a.output_path_wan}"'
            f' --data_file_keys "{a.data_file_keys}"'
            f" {gc}"
            f' --fp8_models "{a.fp8_models}"'
            f" --preferred_timestep_id={a.preferred_timestep_id}"
            f" --n_joints {a.n_joints}"
            f' --extra_inputs "{a.extra_inputs}"'
        )
        status = os.system(cmd)
        exit_code = os.WEXITSTATUS(status)
        if exit_code != 0:
            die("Stage 2 failed.")

    # ── summary ───────────────────────────────────────────────────────────────
    rows = []
    if a.mode in ("data_process", "both"):
        rows.append(("Latents (stage 1)",      a.output_path))
    if a.mode in ("data_process_with_wan", "both"):
        rows.append(("DIT features (stage 2)", a.output_path_wan))
    print_summary(rows, title="✓  All stages completed")

if __name__ == "__main__":
    main()
