import torch
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lora_path', type=str, required=True, help='Path to the LoRA weights file')
parser.add_argument('--output_video', type=str, default='video.mp4', help='Output video file name')
args = parser.parse_args()

os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/host/ist/ist-share/vision/huggingface_hub/"

from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
path = "/host/ist-nas/ist-share/vision/modelscope/Wan2.1-T2V-1.3B/"
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/", **vram_config),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
    redirect_common_files=False,
)

pipe.load_lora(pipe.dit, '../models/train/Wan2.1-T2V-1.3B_lora/epoch-4.safetensors', alpha=1)

video = pipe(
    prompt="A person wearing a grey crop top, yellow pants with blue stripes, black sneakers, orange visor glasses, and orange headphones performs Jumping Jack, captured from the front view.",
    negative_prompt="change in appearance, blurry, low resolution, low quality, overexposed, underexposed, washed out colors, artifacts, jpeg compression, distorted body, deformed limbs, extra limbs, extra fingers, missing fingers, fused fingers, poorly drawn hands, poorly rendered face, warped anatomy, unnatural pose, static frame, frozen motion, motion blur streaks, duplicated person, cluttered background, unrelated objects, text, subtitles, watermark, logo, three legs, many people in the background, incorrect perspective, unrealistic motion",
    seed=0, tiled=True,
)
if os.path.exists(args.output_video):
    # Add timestamp to avoid overwriting
    save_name = args.output_video.split('.mp4')[0] + f'_{int(torch.time.time())}.mp4'
else:
    save_name = args.output_video

save_video(video, save_name, fps=15, quality=5)
