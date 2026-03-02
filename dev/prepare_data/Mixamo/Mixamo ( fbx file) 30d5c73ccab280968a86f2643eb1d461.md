# Mixamo (.fbx file)

## **Preprocessing for any characters**

## Summarization

This consisted of 5 steps

1. Rendering step
    - Input: .fbx file
    - Output: All motion frames (images) + `motion_data.npz` + `skeleton_cam.json` per camera
2. Reformat step: convert rendered frames into Wan's dataset format
    - Input: PNG frames organised by character / motion / camera
    - Output: `.mp4` videos + `.npz` motion files + `metadata.csv` / `metadata_front_view.csv`
3. Chunking step: chunk full-length sequences into N-frame clips
    - Input: `.mp4` + `.npz` files from step 2 + metadata CSV
    - Output: Chunked `.mp4` + `.npz` files + `metadata_front_view_{N}frames.csv`
4. Combining step: merge all per-character folders into one flat `all/` directory
    - Input: Chunked data organised by character
    - Output: `all/` directory with character-prefixed filenames + `all_metadata_front_view_{N}frames.csv`
5. Precompute features step: precompute VAE latents and DIT features for training
    - Input: `all/` directory + combined metadata CSV
    - Output: `latents/` (VAE latents) and `dit_features/` (DIT features)

### ***Rendering***

- Prerequisites:
    1. Download motion from [Mixamo](https://www.mixamo.com/)
    2. Put into the any folder that contains motion files.
    3. Scripts 
        1. [run.py](https://github.com/XezXey/DiffSynth-Studio/blob/main/dev/prepare_data/Mixamo/run.py): Automatic script that will run both render_fbx.py and project_2d.py
        2. [render_fbx.py](https://github.com/XezXey/DiffSynth-Studio/blob/main/dev/prepare_data/Mixamo/render_fbx.py): Render .fbx given rendering parameters (e.g., cam_height, cam_radius, follow_bone, see the full set of params in the [code](https://github.com/XezXey/DiffSynth-Studio/blob/main/dev/prepare_data/Mixamo/render_fbx.py))
        3. [project_2d.py](https://github.com/XezXey/DiffSynth-Studio/blob/main/dev/prepare_data/Mixamo/project_2d.py): Projection the skeleton into 2D screen using 3D joints and camera parameters (Usually for checking the project/unproject function)
- Execute command:
    - Automatically run to the given folder
        - Expected input folder structure:
            
            ```bash
            # args.fbx = ./testing_motion/prisoner_b_styperek
            - ./testing_motion/prisoner_b_styperek (/<path_to_fbx>/)
            	|- Walking.fbx
            	|- Running.fbx
            	|- ...
            ```
            
        - Expected output folder structure:
            
            ```bash
            # args.output_dir = /data2/mint/Motion_Dataset/Mixamo/testing_motion_720p/prisoner_b_styperek
            - /data2/mint/Motion_Dataset/Mixamo/testing_motion_720p/prisoner_b_styperek
            	|- Walking
            		|- Cam_0
            			|- frame0000.png
            			|- ...
            			|- skeleton_cam0.json (Combined the joints 2d/3d data, camera, etc.)
            		|- Cam_1 (if n_cam > 1)
            		|- ..... (if n_cam > 1)
            	|- Running
            		|- Cam_0
            		|- ..... (if n_cam > 1)
            	
            ```
            
        - Running Command
        
        ```bash
        # Command to run
        python run.py 
        --fbx "./testing_motion/prisoner_b_styperek/" 
        --out_dir "/data2/mint/Motion_Dataset/Mixamo/testing_motion_720p/prisoner_b_styperek"
        --n_cam 4 
        --follow_bone mixamorig:Hips 
        --cam_height 3.0 --cam_radius 4.5 
        --img_width 1280 --img_height 720
        
        ```
        
        - **In case**, we have a multiple character that have the same/similar motion (Mixamo shared the same name of the motion). We can run the script to process throughout all the characters
            - Running Command
                
                ```bash
                python run_multiple_chars.py 
                --input_dir ./testing_motion/ 
                --output_dir /data2/mint/Motion_Dataset/Mixamo/testing_motion_720p 
                --run_blender 
                --max_log_lines 30 
                --n_cam 1 # Other params (similar to run.py) can also be added.
                --use_gpu 
                --run_blender --run_projection --skip_plot_map --only_body_joints
                
                [#] Note that the input and output directories are just 1 step back to the parents.
                [#] The rest folder structure is the same.
                ```
                

### Generate ready-to-train Wan’s format

- Scripts
    - [gen_data_format.py](https://github.com/XezXey/DiffSynth-Studio/blob/main/dev/prepare_data/Mixamo/gen_data_format.py): Convert rendered PNG frames for a **single** character into `.mp4` + metadata.
    - [gen_data_format_multiple_chars.py](https://github.com/XezXey/DiffSynth-Studio/blob/main/dev/prepare_data/Mixamo/gen_data_format_multiple_chars.py): Wrapper that calls `gen_data_format.py` for each character folder.
- Expected input folder structure:
    
    ```bash
    - /data2/mint/Motion_Dataset/Mixamo/testset_raw/
    	|- mannequin/
    		|- Walking/
    			|- Cam_0/
    				|- frame0000.png
    				|- ...
    				|- motion_data.npz
    				|- skeleton_cam0.json
    			|- Cam_1/ (if n_cam > 1)
    		|- Running/
    	|- prisoner_b_styperek/
    	|- ...
    ```
    
- Expected output folder structure:
    
    ```bash
    - /data2/mint/Motion_Dataset/Mixamo/testset_fmt/
    	|- mannequin/
    		|- Walking_cam_0_render.mp4
    		|- Walking_cam_0_proj.mp4
    		|- Walking_cam_0_motion_data.npz
    		|- ...
    		|- metadata.csv              # all cameras
    		|- metadata_front_view.csv   # cam_0 only
    	|- prisoner_b_styperek/
    	|- ...
    ```
    
- Running Command
    
    ```bash
    # Single character
    python gen_data_format.py \
      --data_path /data2/mint/Motion_Dataset/Mixamo/testset_raw/mannequin \
      --output_path /data2/mint/Motion_Dataset/Mixamo/testset_fmt/mannequin
    
    # All characters at once
    python gen_data_format_multiple_chars.py \
      --data_path /data2/mint/Motion_Dataset/Mixamo/testset_raw \
      --output_path /data2/mint/Motion_Dataset/Mixamo/testset_fmt
    ```
    

---

### Chunking

- Scripts
    - [chunk_data.py](https://github.com/XezXey/DiffSynth-Studio/blob/main/dev/prepare_data/Mixamo/chunk_data.py): Chunk a **single** character's sequences into N-frame clips.
    - [chunk_data_multiple_chars.py](https://github.com/XezXey/DiffSynth-Studio/blob/main/dev/prepare_data/Mixamo/chunk_data_multiple_chars.py): Wrapper that calls `chunk_data.py` for each character folder.
- Expected input folder structure:
    
    ```bash
    - /data2/mint/Motion_Dataset/Mixamo/testset_fmt/
    	|- mannequin/
    		|- Walking_cam_0_render.mp4
    		|- Walking_cam_0_motion_data.npz
    		|- metadata_front_view.csv     # ← used as --metadata_example
    	|- prisoner_b_styperek/
    	|- ...
    ```
    
- Expected output folder structure:
    
    ```bash
    - /data2/mint/Motion_Dataset/Mixamo/testset_5f/
    	|- mannequin/
    		|- Walking_cam_0_render_chunk0_video.mp4
    		|- Walking_cam_0_render_chunk0_motion.npz
    		|- ...
    		|- metadata_front_view_5frames.csv   # auto-named: {metadata_name}_{n_frames}frames.csv
    	|- prisoner_b_styperek/
    	|- ...
    ```
    
- Running Command
    
    ```bash
    # Single character
    python chunk_data.py \
      --input_dir /data2/mint/Motion_Dataset/Mixamo/testset_fmt/mannequin \
      --output_dir /data2/mint/Motion_Dataset/Mixamo/testset_5f/mannequin \
      --n_frames 5 \
      --metadata_file /data2/mint/Motion_Dataset/Mixamo/testset_fmt/mannequin/metadata_front_view.csv
    
    # All characters at once
    python chunk_data_multiple_chars.py \
      --input_dir /data2/mint/Motion_Dataset/Mixamo/testset_fmt \
      --output_dir /data2/mint/Motion_Dataset/Mixamo/testset_5f \
      --n_frames 5 \
      --metadata_example /data2/mint/Motion_Dataset/Mixamo/testset_fmt/mannequin/metadata_front_view.csv
    ```
    

---

### Combining

- Scripts
    - [combined_output.py](https://github.com/XezXey/DiffSynth-Studio/blob/main/dev/prepare_data/Mixamo/combined_output.py): Merges all per-character subfolders into a single `all/` directory, prepending the character name to every file and concatenating matching metadata CSVs.
- Expected input folder structure:
    
    ```bash
    - /data2/mint/Motion_Dataset/Mixamo/testset_5f/
    	|- mannequin/
    		|- Walking_cam_0_render_chunk0_video.mp4
    		|- Walking_cam_0_render_chunk0_motion.npz
    		|- metadata_front_view_5frames.csv
    	|- prisoner_b_styperek/
    	|- ...
    ```
    
- Expected output folder structure:
    
    ```bash
    - /data2/mint/Motion_Dataset/Mixamo/testset_5f/
    	|- all/
    		|- mannequin_Walking_cam_0_render_chunk0_video.mp4
    		|- mannequin_Walking_cam_0_render_chunk0_motion.npz
    		|- mannequin_metadata_front_view_5frames.csv   # per-character CSV (prefixed)
    		|- prisoner_b_styperek_Walking_cam_0_render_chunk0_video.mp4
    		|- ...
    		|- all_metadata_front_view_5frames.csv          # ← combined across all characters
    ```
    
- Running Command
    
    ```bash
    python combined_output.py \
      --input_path /host/data2/mint/Motion_Dataset/Mixamo/testset_5f/ \
      --metadata_to_combined metadata_front_view_5frames.csv
    
    # [#] Note: --input_path must have a trailing slash.
    # [#] The combined file is written to all/all_{metadata_to_combined}.
    ```
    

---

### Precompute features

- Scripts
    - [precompute_features.py](https://github.com/XezXey/DiffSynth-Studio/blob/main/examples/wanvideo/my_scripts/precomputed_dit_features/precompute_features.py): Runs two stages:
        - **Stage 1** (`data_process`): Encode videos with the VAE → save latents.
        - **Stage 2** (`data_process_with_wan`): Run the DIT forward pass on fixed noise → save DIT features.
- Expected input folder structure:
    
    ```bash
    - /host/data2/mint/Motion_Dataset/Mixamo/testset_5f/all/
    	|- mannequin_Walking_cam_0_render_chunk0_video.mp4
    	|- mannequin_Walking_cam_0_render_chunk0_motion.npz
    	|- all_metadata_front_view_5frames.csv
    ```
    
- Expected output folder structure:
    
    ```bash
    - /host/data2/mint/Motion_Dataset/SkelAg/testset_5f/
    	|- latents/         # stage 1 output  (--output_path_vae)
    	|- dit_features/    # stage 2 output  (--output_path_wan)
    ```
    
- Running Command
    
    ```bash
    # Must run from the repo root: .../DiffSynth-Studio/
    python examples/wanvideo/my_scripts/precomputed_dit_features/precompute_features.py \
      --dataset_base_path /host/data2/mint/Motion_Dataset/Mixamo/testset_5f/all/ \
      --dataset_metadata_path /host/data2/mint/Motion_Dataset/Mixamo/testset_5f/all/all_metadata_front_view_5frames.csv \
      --height 320 --width 640 \
      --num_frames 5 \
      --dataset_repeat_vae 1 --dataset_repeat_wan 1 \
      --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
      --tokenizer_path "/host/ist/ist-share/vision/huggingface_hub/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl/" \
      --mode "both" \
      --output_path_vae "/host/data2/mint/Motion_Dataset/SkelAg/testset_5f/latents/" \
      --output_path_wan "/host/data2/mint/Motion_Dataset/SkelAg/testset_5f/dit_features/" \
      --data_file_keys "video,motion" \
      --offload_models "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" \
      --extra_inputs "input_image" \
      --preferred_timestep_id -20 \
      --gpu_id 0
    ```
    

---

### Full pipeline (`run_pipeline.py`)

- Script: [run_pipeline.py](https://github.com/XezXey/DiffSynth-Studio/blob/main/dev/prepare_data/Mixamo/run_pipeline.py)
- Runs all 5 steps in sequence. Each step can be individually skipped with `--skip_*` flags.
- **Working directory must be** `.../DiffSynth-Studio/dev/prepare_data/Mixamo`
- Key arguments:

    | Argument | Description |
    | --- | --- |
    | `--input_dir` | Root dir with character subdirs of FBX files (step 1 input) |
    | `--render_output_dir` | Output of step 1 / input to step 2 |
    | `--format_output_dir` | Output of step 2 / input to step 3 |
    | `--chunk_output_dir` | Output of step 3 / input to steps 4–5 |
    | `--host_prefix` | Prefix prepended to `chunk_output_dir` for steps 4–5 (e.g. `/host` inside Docker) |
    | `--vae_output_path` | Output dir for VAE latents (step 5) |
    | `--wan_output_path` | Output dir for DIT features (step 5) |
    | `--n_frames` | Frames per chunk (default: `5`) |
    | `--metadata_name` | Metadata CSV basename to search for (default: `metadata_front_view.csv`) |
    | `--wan_height` / `--wan_width` | Resolution for precompute stage (default: `320` / `640`) |
    | `--gpu_id` | CUDA device to use in step 5 (default: `0`) |
    | `--skip_render` / `--skip_format` / … | Skip individual steps to resume a partial run |

- Running Command

    ```bash
    # Must be run from: .../DiffSynth-Studio/dev/prepare_data/Mixamo
    python run_pipeline.py \
      --input_dir ./testset_motion/ \
      --render_output_dir /data2/mint/Motion_Dataset/Mixamo/testset_mixamo_720p_only_body_with_motion_data \
      --format_output_dir /data2/mint/Motion_Dataset/Mixamo/rdy_testset_mixamo_720p_only_body_with_motion_data \
      --chunk_output_dir  /data2/mint/Motion_Dataset/Mixamo/rdy_testset_mixamo_720p_only_body_with_motion_data_5frames \
      --host_prefix /host \
      --vae_output_path /host/data2/mint/Motion_Dataset/SkelAg/testset_frontview_bodyjoints_320p_5frames/latents \
      --wan_output_path /host/data2/mint/Motion_Dataset/SkelAg/testset_frontview_bodyjoints_320p_5frames/dit_features \
      --use_gpu --run_projection --skip_plot_map --only_body_joints \
      --wan_height 320 --wan_width 640 --n_frames 5 \
      --preferred_timestep_id -20 --gpu_id 3
    
    # [#] To resume from step 3 onward (skipping render + format):
    python run_pipeline.py ... --skip_render --skip_format
    ```
