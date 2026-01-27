import torch
import os
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_image", type=str, default="data/examples/wan/cat_fightning.jpg")
args = parser.parse_args()

os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/host/ist/ist-share/vision/huggingface_hub/"
# # CPU Offload
vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="google/umt5-xxl/"),
    redirect_common_files=False,
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Image-to-video
input_image = Image.open(args.input_image).resize((1248, 704))
video = pipe(
    prompt="A young girl from the input image starts walking forward naturally. Her body shows smooth, realistic walking motion with coordinated arm swings and leg strides. The character identity, clothing, hairstyle, and facial features remain exactly the same as the input image. Static camera perspective, no zoom or pan.",
    negative_prompt="zoom, dolly, pan, tilt, camera movement, camera shake, perspective change, focal length change, background motion, parallax, cinematic camera, dynamic camera, motion blur, jitter, identity change, face distortion, body deformation, clothing change, inconsistent lighting",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
    # num_inference_steps=100
)
save_video(video, f"video_2_Wan2.2-TI2V-5B_{os.path.basename(args.input_image).split('.')[0]}.mp4", fps=15, quality=5)