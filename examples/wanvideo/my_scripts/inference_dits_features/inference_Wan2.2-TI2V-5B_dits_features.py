import torch
from safetensors.torch import load_file
import os
import numpy as np
import argparse
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.motion_models.joint_map_vae import JointHeatMapMotionUpsample
from diffsynth.diffusion.mint_loss import unproject_torch
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
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B", help="Model id to use for the WanVideoPipeline.")
    parser.add_argument("--input_image", type=str, default="data/examples/wan/cat_fightning.jpg")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--prompt", type=str, default="The girl in the input image walks straight forward naturally, with smooth and realistic walking motion. Her identity, clothing, hairstyle, and facial features remain unchanged. The camera is static, with no zoom, pan, or perspective change.")
    parser.add_argument("--negative_prompt", type=str, default="zoom, dolly, pan, tilt, camera movement, camera shake, perspective change, focal length change, background motion, parallax, cinematic camera, dynamic camera, motion blur, jitter, identity change, face distortion, body deformation, clothing change, inconsistent lighting")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="./results/")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--save_suffix", type=str, default=None)
    return parser

parser = wan_parser()
args = parser.parse_args()

# extract "Wan2.1" or "Wan2.2" from the model id
pattern = r'Wan2\.\d'
match = re.search(pattern, args.model_id)
if match:
    model_version = match.group(0)
    print(f"Extracted model version: {model_version}")
else:
    raise ValueError("Model ID does not contain a valid Wan model version.")


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
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

pipe.eval()
for p in pipe.parameters():
    p.requires_grad = False

# Image-to-video
input_image = Image.open(args.input_image).convert('RGB').resize((args.width, args.height))
video, return_dict = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    seed=0, tiled=True,
    height=args.height, width=args.width,
    input_image=input_image,
    num_frames=args.num_frames,
    return_features=True,
    num_inference_steps=args.num_inference_steps,
    cfg_scale=args.cfg_scale
)

os.makedirs(args.save_path, exist_ok=True)
if args.save_suffix is None:
    save_name = os.path.join(args.save_path, f"video_{args.model_id.split('/')[-1]}_{os.path.basename(args.input_image).split('.')[0]}.mp4")
else:
    save_name = os.path.join(args.save_path, f"video_{args.model_id.split('/')[-1]}_{os.path.basename(args.input_image).split('.')[0]}_{args.save_suffix}.mp4")

save_video(video, save_name, fps=15, quality=5)

class DummyExtraModules(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.extra_modules = JointHeatMapMotionUpsample(
            # n_joints=23,
            n_joints=65,
            dit_dim=pipe.dit.dim,
            head_out_dim=pipe.dit.out_dim,
            flatten_dim=256, #TODO: Fix this!!!
            vae_latent_dim=pipe.vae.z_dim,
            patch_size=pipe.dit.patch_size,
            device=device
        )
        self.extra_modules.eval()
        for p in self.extra_modules.parameters():
            p.requires_grad = False

    def forward(self, pipe, dit_features, grid_size):
        return self.extra_modules(pipe, dit_features, grid_size)

extra_modules = DummyExtraModules(device=pipe.device if not args.split_gpu else "cuda:1")
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
pixel_coords, depth = extra_modules(pipe, dit_features, grid_size)  # pixel_coords = (B, J, T, 2); depth = (B, J, T, 1)

motion_data = "/host/data/mint/Motion_Dataset/Mixamo/rdy_mixamo_720p_with_motion_data/Walking_cam_0_motion_data.npz"
motion_data = np.load(motion_data)
# Reconstruct 3D motion from (u, v, depth) using camera intrinsics and extrinsics
fx, fy, cx, cy = motion_data["cams_intr"]
org_h = cy * 2.0 + 1
org_w = cx * 2.0 + 1
E_bl = torch.tensor(motion_data["cams_extr"]).to(device=pipe.device)
E_bl = E_bl[:args.num_frames, ...]

u = pixel_coords[..., 0] * (org_w - 1)    # B, J, T
v = pixel_coords[..., 1] * (org_h - 1)    # B, J, T
d = depth[..., 0]
motion_pred_2d = torch.stack([u / (org_w - 1), v / (org_h - 1)], dim=-1).squeeze(0).permute(1, 0, 2)  # B, J, T -> T, J, 2
motion_pred_3d = unproject_torch(fx, fy, cx, cy, E_bl, torch.stack([u, v, d], dim=-1).squeeze(0).permute(1, 0, 2))
motion_pred_3d = motion_pred_3d.cpu().numpy()

output = {
    "motion_pred_3d": motion_pred_3d,
    "motion_pred_2d": motion_pred_2d.cpu().numpy(),
}

if args.save_suffix is None:
    np.savez(os.path.join(args.save_path, f"motion_pred_3d_{args.model_id.split('/')[-1]}_{os.path.basename(args.input_image).split('.')[0]}.npz"), **output)
else:
    np.savez(os.path.join(args.save_path, f"motion_pred_3d_{args.model_id.split('/')[-1]}_{os.path.basename(args.input_image).split('.')[0]}_{args.save_suffix}.npz"), **output)