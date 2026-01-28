import torch
from safetensors.torch import load_file
import os
import argparse
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.motion_models.joint_map_vae import JointHeatMapMotionUpsample
from modelscope import dataset_snapshot_download
import re

os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/host/ist/ist-share/vision/huggingface_hub/"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # CPU Offload
vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

def wan_parser():
    #NOTE: Extra parameters for training additional modules
    parser = argparse.ArgumentParser(description="WanVideo Inference with DITS Features")
    parser.add_argument("--extra_modules_ckpt", type=str, required=True, help="Name to use when saving checkpoints.")
    parser.add_argument("--split_gpu", default=False, action="store_true", help="Use another gpu for inference the extra modules.")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-TI2V-5B", help="Model id to use for the WanVideoPipeline.")
    parser.add_argument("--input_image", type=str, default="data/examples/wan/cat_fightning.jpg")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--prompt", type=str, default="The girl in the input image walks straight forward naturally, with smooth and realistic walking motion. Her identity, clothing, hairstyle, and facial features remain unchanged. The camera is static, with no zoom, pan, or perspective change.")
    parser.add_argument("--negative_prompt", type=str, default="zoom, dolly, pan, tilt, camera movement, camera shake, perspective change, focal length change, background motion, parallax, cinematic camera, dynamic camera, motion blur, jitter, identity change, face distortion, body deformation, clothing change, inconsistent lighting")
    parser.add_argument("--save_suffix", type=str, default=None)
    return parser

parser = wan_parser()
args = parser.parse_args()

# extract "Wan2.1" or "Wan2.2" from the model id
pattern = r'Wan2\.\d'\.\d'
match = re.search(pattern, args.model_id)
if match:
    model_version = match.group(0)
    print(f"Extracted model version: {model_version}")
else:
    raise ValueError("Model ID does not contain a valid Wan model version.")


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    # model_configs=[
    #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
    #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
    #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
    # ],
    model_configs=[
        ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id=args.model_id, origin_file_pattern=f"{model_version}_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="google/umt5-xxl/"),
    redirect_common_files=False,
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
    return_features=True
)
# Text-to-video
video, return_dict = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True, num_inference_steps=3,
    return_features=True,
    height=704//2, width=1248//2,
    num_frames=81,
)
save_video(video, "dummy.mp4", fps=15, quality=5)
exit()

class DummyExtraModules(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.extra_modules = JointHeatMapMotionUpsample(
            n_joints=23,
            dit_dim=pipe.dit.dim,
            head_out_dim=pipe.dit.out_dim,
            flatten_dim=256, #TODO: Fix this!!!
            vae_latent_dim=pipe.vae.z_dim,
            patch_size=pipe.dit.patch_size,
            device=device
        )
    def forward(self, dit_features, grid_size):
        return self.extra_modules(None, dit_features, grid_size)

extra_modules = DummyExtraModules(device=pipe.device if not args.split_gpu else "cuda:1")
extra_modules.eval()
for param in extra_modules.parameters():
    param.requires_grad = False
tmp_params = torch.mean(torch.stack([p.float().mean() for p in extra_modules.parameters()]))
#NOTE: Just checking we've save and load the extra modules correctly.
ckpt = load_file(args.extra_modules_ckpt)
assert len(extra_modules.state_dict()) == len(ckpt.keys())
all_m = list(ckpt.keys())
all_m_check = list(ckpt.keys())
for m in all_m:
    if m in list(extra_modules.state_dict()):
        all_m_check.remove(m)
assert len(all_m_check) == 0, "The extra modules checkpoint keys do not match the model architecture."
extra_modules.load_state_dict(ckpt)
new_tmp_params = torch.mean(torch.stack([p.float().mean() for p in extra_modules.parameters()]))
assert tmp_params != new_tmp_params, "The extra modules parameters do not seem to have changed"

# Remove pipe models from GPU to save memory
pipe.load_models_to_device([])
# Forward pass through extra modules to get motion predictions
dit_features = return_dict.get("dit_features", None).to(device=extra_modules.extra_modules.device)
grid_size = return_dict.get("grid_size", None)
assert dit_features is not None, "Dit features not returned from model_fn."
assert grid_size is not None, "Grid size not returned from model_fn."
pixel_coords, depth = extra_modules(dit_features, grid_size)
motion_pred = torch.cat([pixel_coords, depth], dim=-1)
print("motion_pred shape: ", motion_pred.shape)