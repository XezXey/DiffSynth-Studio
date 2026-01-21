from .base_pipeline import BasePipeline
import torch

def TrainingOnDitFeaturesLoss(pipe: BasePipeline, extra_modules=None, **inputs):
    # max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    # min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    preferred_timestep_id = inputs.get("preferred_timestep_id", [-1])
    timestep_id = torch.tensor(preferred_timestep_id, dtype=torch.int)

    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    # training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, return_dict = pipe.model_fn(**models, **inputs, timestep=timestep)

    dit_features = return_dict.get("dit_features", None)
    grid_size = return_dict.get("grid_size", None)
    assert dit_features is not None, "Dit features not returned from model_fn."
    assert grid_size is not None, "Grid size not returned from model_fn."

    pixel_coords, depth = extra_modules(pipe, dit_features, grid_size)
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss