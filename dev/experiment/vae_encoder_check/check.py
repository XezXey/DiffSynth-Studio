import torch as th

def check_stats(x):
    print("===> ", x.shape, x.dtype, x.min().item(), x.max().item(), th.mean(x).item(), th.std(x).item())

inference_from_video = th.load("./vae_latents_inference_from_video.pth", weights_only=False)
input_latents_from_inference_video = th.load('./input_latents_from_inference_video.pth', weights_only=False)
input_latents_from_inference_video_bfloat16 = th.load('./input_latents_from_inference_video_bfloat16.pth', weights_only=False)
input_video_from_inference_video = th.load('./input_video_from_inference_video.pth', weights_only=False)

# Data processed saved corrected vae-encoded latents (save right after the encoding without adding noise)
data_process_new = th.load("/host/data/mint/SkelAg/training/test_single_sample_encode_vae/0/0.pth", weights_only=False)
data_process_old = th.load("/host/data/mint/SkelAg/training/heatmap_TI2V-5B_320x640_single_sample/0/0.pth", weights_only=False)
input_latents_from_data_processed = th.load('./input_latents_from_data_processed.pth', weights_only=False)
input_latents_from_data_processed_nofixseed = th.load('./input_latents_from_data_processed_nofixseed.pth', weights_only=False)
input_video_from_data_processed = th.load('./input_video_from_data_processed.pth', weights_only=False)

print("=== Stats's Latents ===")
print("From inference video:")
check_stats(inference_from_video)
check_stats(input_latents_from_inference_video)
check_stats(input_latents_from_inference_video_bfloat16)
print("From data processed:")
check_stats(data_process_new[0]['input_latents'])
check_stats(data_process_old[0]['input_latents'])
check_stats(input_latents_from_data_processed)
check_stats(input_latents_from_data_processed_nofixseed)

print("=== Stats's Videos ===")
print("From inference video:")
check_stats(input_video_from_inference_video)
print("From data processed:")
check_stats(input_video_from_data_processed)


