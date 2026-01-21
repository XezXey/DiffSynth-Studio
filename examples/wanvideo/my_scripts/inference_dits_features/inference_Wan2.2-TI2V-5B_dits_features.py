import torch
from safetensors.torch import load_file
import os
import argparse
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.motion_models.joint_map_vae import JointHeatMapMotionUpsample
from modelscope import dataset_snapshot_download
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

def wan_parser():
    #NOTE: Extra parameters for training additional modules
    parser = argparse.ArgumentParser(description="WanVideo Inference with DITS Features")
    parser.add_argument("--extra_modules_ckpt", type=str, required=True, help="Name to use when saving checkpoints.")
    return parser

parser = wan_parser()
args = parser.parse_args()

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    # model_configs=[
    #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
    #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
    #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
    # ],
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    redirect_common_files=False,
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
    return_features=True
)
class DummyExtraModules(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extra_modules = JointHeatMapMotionUpsample(
            n_joints=23,
            dit_dim=pipe.dit.dim,
            head_out_dim=pipe.dit.out_dim,
            flatten_dim=256, #TODO: Fix this!!!
            vae_latent_dim=pipe.vae.z_dim,
            patch_size=pipe.dit.patch_size,
            device=pipe.device
        )
    def forward(self, inp):
        return self.extra_modules(inp)

extra_modules = DummyExtraModules()
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

# Text-to-video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     num_frames=121,
# )
# save_video(video, "video_1_Wan2.2-TI2V-5B.mp4", fps=15, quality=5)

# Text-to-video
video, return_dict = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True, num_inference_steps=3,

)
save_video(video, "video_1_Wan2.1-T2V-1.3B.mp4", fps=15, quality=5)

# Image-to-video
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
)
input_image = Image.open("data/examples/wan/cat_fightning.jpg").resize((1248, 704))
video, return_dict = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
    num_inference_steps=3,
    preferred_timestep_id=[-1],
    preferred_dit_block_id=[-1],
    return_features=True,
)
save_video(video, "video_2_Wan2.2-TI2V-5B.mp4", fps=15, quality=5)