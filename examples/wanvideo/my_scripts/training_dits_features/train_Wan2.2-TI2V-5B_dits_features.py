import torch, os, argparse, accelerate, warnings
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.core import UnifiedMotionDataset
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.motion_models.joint_map_vae import JointHeatMapMotionUpsample
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.diffusion import *
from diffsynth.diffusion.mint_loss import TrainingOnDitFeaturesLoss
def seed_everything(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(47)


import wandb

os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/host/ist/ist-share/vision/huggingface_hub/"
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--preferred_timestep_id", type=int, nargs='+', default=[-1], help="Preferred timestep IDs for training on DIT features. Use -1 to indicate the last timestep.")
    parser.add_argument("--preferred_dit_block_id", type=int, nargs='+', default=[-1], help="Preferred DIT block IDs for training on DIT features. Use -1 to indicate the last DIT block.")
    #NOTE: Extra parameters for training additional modules
    parser.add_argument("--save_name", type=str, default=None, help="Name to use when saving checkpoints.")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Whether to use wandb for logging.")
    return parser

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        # pipe: WanVideoPipeline,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        preferred_timestep_id=[-1],   # Use last timestep by default to train on dit features
        preferred_dit_block_id=[-1],    # Use last block by default to train on dit features
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,

    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, redirect_common_files=False, return_features=True)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        self.preferred_timestep_id = preferred_timestep_id
        self.preferred_dit_block_id = preferred_dit_block_id

        # Use Wan models as frozen models
        self.force_no_grad()

        if ":data_process" in task:
            self.extra_modules = None
        else:
            vae_latent_dim = self.pipe.vae.z_dim if hasattr(self.pipe.vae, 'z_dim') else self.pipe.dit.out_dim
            self.extra_modules = JointHeatMapMotionUpsample(
                n_joints=65,    #TODO: Fix this!!!
                dit_dim=self.pipe.dit.dim,
                head_out_dim=self.pipe.dit.out_dim,
                flatten_dim=256, #TODO: Fix this!!!
                # flatten_dim=384, #TODO: Fix this!!!
                vae_latent_dim=vae_latent_dim,
                patch_size=self.pipe.dit.patch_size,
                device=self.pipe.device
            )

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            task=task,
        )

        self.task = task
        self.task_to_loss = {
            "dit_features": lambda pipe, inputs_shared, inputs_posi, inputs_nega: TrainingOnDitFeaturesLoss(pipe, self.extra_modules, **inputs_shared, **inputs_posi),
            "dit_features:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: TrainingOnDitFeaturesLoss(pipe, self.extra_modules, **inputs_shared, **inputs_posi),
            "dit_features:data_process": lambda pipe, *args: args,
        }

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
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "preferred_timestep_id": self.preferred_timestep_id,
            "preferred_dit_block_id": self.preferred_dit_block_id,
            # Motion data
            "joints_3d": data['motion']["joints_3d"],
            "joints_2d": data['motion']["joints_2d"],
            "cams_intr": data['motion']["cams_intr"],
            "cams_extr": data['motion']["cams_extr"],
            "joint_names": data['motion']["joint_names"],
            "bones": data['motion']["bones"],
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        if ':data_process' in self.task:
            loss = self.task_to_loss[self.task](self.pipe, *inputs)
            pred_dict = {}
        else:
            loss, pred_dict = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss, pred_dict


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    # Image-to-video
    dataset = UnifiedMotionDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedMotionDataset.default_video_operator(
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
        motion_data_operator=UnifiedMotionDataset.default_motion_operator(
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
    # pipe = WanVideoPipeline.from_pretrained(
    #     torch_dtype=torch.bfloat16,
    #     device="cuda",
    #     # model_configs=[
    #     #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
    #     #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
    #     #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
    #     # ],
    #     model_configs=[
    #         ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
    #         ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
    #         ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    #     ],
    #     tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    #     redirect_common_files=False,
    #     vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
    #     return_features=True
    # )
    
    model = WanTrainingModule(
        # pipe=pipe,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device="cuda",
        task=args.task,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        preferred_dit_block_id=args.preferred_dit_block_id,
        preferred_timestep_id=args.preferred_timestep_id,
    )
    os.makedirs(args.output_path + "/wandb", exist_ok=True)
    if args.use_wandb:
        print("Using wandb logger...")
        wandb_logger = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="xezxey",
            # Set the wandb project where this run will be logged.
            project="SkelAg",
            # Name of this run
            name=args.save_name,
            # Track hyperparameters and run metadata.
            config={
                # Model info
                "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "model_id_with_origin_paths": args.model_id_with_origin_paths,
                # Training info
                "learning_rate": args.learning_rate,
                "epochs": args.num_epochs,
                "task": args.task,
                "dataset_repeat": args.dataset_repeat,
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                # Saving info
                "output_path": args.output_path,
                "save_name": args.save_name,
                # Dataset info
                "dataset_base_path": args.dataset_base_path,
                "dataset_metadata_path": args.dataset_metadata_path,
            },
            dir=args.output_path + "/wandb",

        )

        training_logger = TrainingLogger(
            wandb_logger,
            args.output_path + "/wandb",
        )
    else: 
        print("No training logger is used.")
        training_logger = None

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launcher_map = {
        "dit_features": launch_training_task_add_modules,
        "dit_features:train": launch_training_task_add_modules,
        "dit_features:data_process": launch_data_process_task_add_modules,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, training_logger, args=args)