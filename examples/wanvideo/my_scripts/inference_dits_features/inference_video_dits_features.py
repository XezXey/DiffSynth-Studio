import torch
from safetensors.torch import load_file
import os
import numpy as np
import argparse
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.diffusion.parsers import add_video_size_config
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.motion_models.joint_map_vae import JointHeatMapMotionUpsample
from diffsynth.core.data.simplified_motion_dataset import SimplifiedMotionDataset
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
    parser.add_argument("--prompt", type=str, default="The girl in the input image walks straight forward naturally, with smooth and realistic walking motion. Her identity, clothing, hairstyle, and facial features remain unchanged. The camera is static, with no zoom, pan, or perspective change.")
    parser.add_argument("--negative_prompt", type=str, default="zoom, dolly, pan, tilt, camera movement, camera shake, perspective change, focal length change, background motion, parallax, cinematic camera, dynamic camera, motion blur, jitter, identity change, face distortion, body deformation, clothing change, inconsistent lighting")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="./results/")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--save_suffix", type=str, default=None)
    # Video size config
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--max_pixels", type=int, default=1024*1024)
    # Loading 
    parser.add_argument("--data_file_keys", type=str, default="video,motion,prompt")
    parser.add_argument("--input_video", type=str, default=None)
    parser.add_argument("--dataset_base_path", type=str, default='')
    parser.add_argument("--dataset_metadata_path", type=str, default=None)
    # Sampling & Scheduler config
    parser.add_argument("--denoising_strength", type=float, default=1.0)
    parser.add_argument("--sigma_shift", type=int, default=5.0)
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

# Load video as an input
dataset = SimplifiedMotionDataset(
    base_path=args.dataset_base_path,
    metadata_path=args.dataset_metadata_path,
    video_path=args.input_video,
    data_file_keys=args.data_file_keys.split(","),
    main_data_operator=SimplifiedMotionDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=args.num_frames, # 81
        time_division_factor=4,
        time_division_remainder=1,
    ),
    motion_data_operator=SimplifiedMotionDataset.default_motion_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=args.num_frames, # 81
        time_division_factor=4,
        time_division_remainder=1,
    ),
    special_operator_map={
        "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
        "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
    }
)

class WanInferenceModule(torch.nn.Module):
    def __init__(self, pipe: WanVideoPipeline, max_timestep_boundary: float = 1.0, min_timestep_boundary: float = 0.0, preferred_timestep_id: list = [-1], preferred_dit_block_id: list = [-1]):
        super().__init__()
        self.pipe = pipe
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.preferred_timestep_id = preferred_timestep_id
        self.preferred_dit_block_id = preferred_dit_block_id
        
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return data
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        elif isinstance(data, tuple):
            data = tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, list):
            data = list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, dict):
            data = {i: self.transfer_data_to_device(data[i], device, torch_float_dtype) for i in data}
            return data
        else:
            return data
    
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": True,
            "tile_size": (30, 52),
            "tile_stride": (15, 26),
            # tiled: Optional[bool] = True,
            # tile_size: Optional[tuple[int, int]] = (30, 52),
            # tile_stride: Optional[tuple[int, int]] = (15, 26),
            "rand_device": self.pipe.device,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "preferred_timestep_id": self.preferred_timestep_id,
            "preferred_dit_block_id": self.preferred_dit_block_id,
            # Motion data
            "joints_3d": data.get("motion", {}).get("joints_3d"),
            "joints_2d": data.get("motion", {}).get("joints_2d"),
            "cams_intr": data.get("motion", {}).get("cams_intr"),
            "cams_extr": data.get("motion", {}).get("cams_extr"),
            "joint_names": data.get("motion", {}).get("joint_names"),
            "bones": data.get("motion", {}).get("bones"),
        }
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            
        inputs_shared, inputs_posi, inputs_nega = inputs
        inputs = {**inputs_shared, **inputs_posi, **inputs_nega}
        preferred_timestep_id = inputs.get("preferred_timestep_id", [-1])
        timestep_id = torch.tensor(preferred_timestep_id, dtype=torch.int)
        timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
        print(f"[#] Getting DIT features at timestep: ", timestep.item())
        noise = torch.randn_like(inputs["vae_latents"])
        inputs["latents"] = pipe.scheduler.add_noise(inputs["vae_latents"], noise, timestep)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        noise_pred, return_dict = pipe.model_fn(**models, **inputs, timestep=timestep)

        dit_features = return_dict.get("dit_features", None)
        grid_size = return_dict.get("grid_size", None)
        assert dit_features is not None, "Dit features not returned from model_fn."
        assert grid_size is not None, "Grid size not returned from model_fn."
        
        return noise_pred, return_dict

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

# Force setting this so that the dataloaders work correctly
pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=args.denoising_strength, shift=args.sigma_shift)

pipe.eval()
for p in pipe.parameters():
    p.requires_grad = False

inference_module = WanInferenceModule(
    pipe=pipe,
    preferred_timestep_id=[-1],
    preferred_dit_block_id=[-1],
)

data = next(iter(dataset))
data['prompt'] = args.prompt
noise_pred, return_dict = inference_module(data)

# video, return_dict = pipe.inference_on_video(
#     prompt=args.prompt,
#     negative_prompt=args.negative_prompt,
#     seed=0, tiled=True,
#     height=args.height, width=args.width,
#     input_image=input_image,
#     num_frames=args.num_frames,
#     return_features=True,
#     num_inference_steps=args.num_inference_steps,
#     cfg_scale=args.cfg_scale
# )

# Copy the video to save_path
os.makedirs(args.save_path, exist_ok=True)
import shutil
shutil.copy(args.input_video, args.save_path)

# if args.save_suffix is None:
#     save_name = os.path.join(args.save_path, f"video_{args.model_id.split('/')[-1]}_{os.path.basename(args.input_video).split('.')[0]}.mp4")
# else:
#     save_name = os.path.join(args.save_path, f"video_{args.model_id.split('/')[-1]}_{os.path.basename(args.input_video).split('.')[0]}_{args.save_suffix}.mp4")
    
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
    np.savez(os.path.join(args.save_path, f"motion_pred_3d_{args.model_id.split('/')[-1]}_{os.path.basename(args.input_video).split('.')[0]}.npz"), **output)
else:
    np.savez(os.path.join(args.save_path, f"motion_pred_3d_{args.model_id.split('/')[-1]}_{os.path.basename(args.input_video).split('.')[0]}_{args.save_suffix}.npz"), **output)