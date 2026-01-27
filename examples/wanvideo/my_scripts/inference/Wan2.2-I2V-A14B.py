import torch
import os
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/host/ist/ist-share/vision/huggingface_hub/"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_image", type=str, default="data/examples/wan/cat_fightning.jpg")
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=832)
parser.add_argument("--num_frames", type=int, default=121)
parser.add_argument("--prompt", type=str, default="The girl in the input image walks straight forward naturally, with smooth and realistic walking motion. Her identity, clothing, hairstyle, and facial features remain unchanged. The camera is static, with no zoom, pan, or perspective change.")
parser.add_argument("--negative_prompt", type=str, default="zoom, dolly, pan, tilt, camera movement, camera shake, perspective change, focal length change, background motion, parallax, cinematic camera, dynamic camera, motion blur, jitter, identity change, face distortion, body deformation, clothing change, inconsistent lighting")
parser.add_argument("--save_suffix", type=str, default=None)
args = parser.parse_args()

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
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
    redirect_common_files=False
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
)
input_image = Image.open(args.input_image).convert("RGB").resize((args.width, args.height))

video = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    seed=0, tiled=True,
    input_image=input_image,
    switch_DiT_boundary=0.9,
)

if args.save_suffix is None:
    save_name = f"video_2_Wan2.2-I2V-A14B_{os.path.basename(args.input_image).split('.')[0]}.mp4"
else:
    save_name = f"video_2_Wan2.2-I2V-A14B_{os.path.basename(args.input_image).split('.')[0]}_{args.save_suffix}.mp4"
save_video(video, save_name, fps=15, quality=5)