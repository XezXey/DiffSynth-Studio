import torch as th
from typing import Tuple, Optional
import torchvision
import math
from einops import rearrange, repeat
from diffsynth.models.wan_video_dit import SelfAttention, rearrange, precompute_freqs_cis_3d, precompute_freqs_cis, sinusoidal_embedding_1d
from diffsynth.models.wan_video_vae import Decoder3d, WanVideoVAE, count_conv3d, CausalConv3d, Upsample
from tqdm import tqdm
import torch.nn.functional as F



class Head(th.nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int]):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.head = th.nn.Linear(dim, out_dim * math.prod(patch_size))

    def forward(self, x):
        x = (self.head(x))
        return x

class JointHeatMapMotionVAEDecoder(th.nn.Module):
    def __init__(self, n_joints, dit_dim, head_out_dim, flatten_dim, vae_latent_dim, patch_size, device, out_channels=2, num_heads=12):
        super().__init__()
        self.J = n_joints
        self.dit_dim = dit_dim
        self.vae_latent_dim = vae_latent_dim
        self.out_joint_map_channels = out_channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        #TODO: Figure out the flatten dim calculation from flattened dit features
        self.flatten_dim = flatten_dim
        self.upsampling_factor = 8  #NOTE: Default from WanVideoVAE
        
        self.self_attn = SelfAttention(dim=dit_dim, num_heads=num_heads).to(device)
        #NOTE: Taken from the default decoder of WanVideoVAE -> VideoVAE_
        # https://vscode.dev/github/XezXey/DiffSynth-Studio/blob/main/diffsynth/models/wan_video_vae.py#L951
        self.conv2 = CausalConv3d(vae_latent_dim, vae_latent_dim, 1)
        self.decoder = Decoder3d(dim=96, z_dim=vae_latent_dim, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                            attn_scales=[], temperal_upsample=[False, True, True][::-1], dropout=0.0).to(device)
        self.dit_head = Head(dit_dim, head_out_dim, patch_size=patch_size).to(device)
        self.joint_head = th.nn.Linear(flatten_dim, flatten_dim * self.J)

        #NOTE: Default parameters for tiled decoding
        # https://vscode.dev/github/XezXey/DiffSynth-Studio/blob/main/diffsynth/pipelines/wan_video.py#L242 - #VAE tiling
        self.tiled = True,
        self.tile_size = (30, 52)
        self.tile_stride = (15, 26)
        self.device = device

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

    def decode_fn(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        hidden_states = [hidden_state.to("cpu") for hidden_state in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.single_decode(hidden_state, device)
            video = video.squeeze(0)
            videos.append(video)
        videos = th.stack(videos)
        return videos

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = th.ones((length,))
        if not left_bound:
            x[:border_width] = (th.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = th.flip((th.arange(border_width) + 1) / border_width, dims=(0,))
        return x
    
    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = th.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask
    
    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = T * 4 - 3
        weight = th.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)
        values = th.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)

        for h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            # Call VideoVAE_.decode(...)
            hidden_states_batch = self.videovae_decode(hidden_states_batch).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) * self.upsampling_factor, (size_w - stride_w) * self.upsampling_factor)
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values

    def videovae_decode(self, z):
        self.clear_cache()
        # z: [b,c,t,h,w]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i:i + 1, :, :],
                                   feat_cache=self._feat_map,
                                   feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i:i + 1, :, :],
                                    feat_cache=self._feat_map,
                                    feat_idx=self._conv_idx)
                out = th.cat([out, out_], 2) # may add tensor offload
        return out

    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        video = self.videovae_decode(hidden_state)
        return video.clamp_(-1, 1)
    
    def forward(self, pipe, dit_features, grid_size):
        """
        dit_features: (n, b, f, d); 
            - n: num of features from DIT blocks (#Preferred dit block id * #Preferred timesteps),
            - b: batch size,
            - f: flattened spatial-temporal dimension (num_frames * height * width),
            - d: dit feature dimension (usually 1536)
        """
        head_dim = self.dit_dim // self.num_heads
        x = dit_features
        x_flat = rearrange(x, 'n b f d -> b (n f) d')


        # Self-attention on dit features + 1D RoPE
        freqs = precompute_freqs_cis(head_dim, end=x_flat.shape[1], theta=1e4).to(self.device)
        freqs = freqs[None, :, None, :]
        attn_flat = self.self_attn(x_flat.float(), freqs=freqs)
        out_attn = rearrange(attn_flat, 'b (n f) d -> n b f d', n=x.shape[0], f=x.shape[2])
        # DIT head
        out_ffn = self.dit_head(out_attn)
        out_ffn = rearrange(out_ffn, 'n b f d -> n b d f')
        joint_map_pred = self.joint_head(out_ffn)
        joint_map_pred = rearrange(joint_map_pred, 'n b d (j f) -> n b j f d', j=self.J)
        joint_map_pred = joint_map_pred.mean(dim=0) #NOTE: Use average instead of squeeze in case we have multiple n-dit features
        joint_map_unpatch = rearrange(
            joint_map_pred, 'b j (f h w) (x y z c) -> b j c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2], j = self.J
        )
        final_joint_map = []
        for i in tqdm(range(self.J), desc="Decoding joint heat maps"):
            joint_heat_map = self.decode_fn(joint_map_unpatch[0:1, i, ...], device=self.device, tiled=self.tiled, tile_size=self.tile_size, tile_stride=self.tile_stride)
            final_joint_map.append(joint_heat_map)

        return final_joint_map
    
class JointHeatMapMotionUpsample(th.nn.Module):
    def __init__(self, n_joints, dit_dim, head_out_dim, flatten_dim, vae_latent_dim, patch_size, device, upsample_dim=96, out_channels=2, num_heads=12):
        super().__init__()
        self.J = n_joints
        self.dit_dim = dit_dim
        self.vae_latent_dim = vae_latent_dim
        self.out_joint_map_channels = out_channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        #TODO: Figure out the flatten dim calculation from flattened dit features
        self.flatten_dim = flatten_dim
        self.upsampling_factor = 8  #NOTE: Default from WanVideoVAE
        
        self.self_attn = SelfAttention(dim=dit_dim, num_heads=num_heads).to(device)
        self.dit_head = Head(dit_dim, head_out_dim, patch_size=patch_size).to(device)

        # Upsample block
        # Initial conv
        dim_mult = [1, 2, 4, 4]
        upsample_dims = [upsample_dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv1 = CausalConv3d(vae_latent_dim, upsample_dims[0], 3, padding=1).to(device)
        # Upsample layers
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(upsample_dims[:-1], upsample_dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
                upsamples.append(CausalConv3d(in_dim, out_dim, 3, padding=1))
            if i != len(dim_mult) - 1:
                    upsamples.append(th.nn.Sequential(
                            Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                            th.nn.Conv2d(out_dim, out_dim // 2, 3, padding=1)))

        self.upsamples = th.nn.Sequential(*upsamples).to(device)
        # Joint heatmap final conv
        self.joint_map_conv = th.nn.Conv2d(upsample_dims[-1], self.out_joint_map_channels * self.J, 3, padding=1).to(device)
        self.device = device


    def forward(self, pipe, dit_features, grid_size):
        """
        dit_features: (n, b, f, d); 
            - n: num of features from DIT blocks (#Preferred dit block id * #Preferred timesteps),
            - b: batch size,
            - f: flattened spatial-temporal dimension (num_frames * height * width),
            - d: dit feature dimension (usually 1536)
        """
        head_dim = self.dit_dim // self.num_heads
        x = dit_features
        x_flat = rearrange(x, 'n b f d -> b (n f) d')

        # Self-attention on dit features + 1D RoPE
        freqs = precompute_freqs_cis(head_dim, end=x_flat.shape[1], theta=1e4).to(self.device)
        freqs = freqs[None, :, None, :]
        attn_flat = self.self_attn(x_flat.float(), freqs=freqs)
        out_attn = rearrange(attn_flat, 'b (n f) d -> n b f d', n=x.shape[0], f=x.shape[2])
        # DIT head
        out_head = self.dit_head(out_attn)
        out_head = out_head.mean(dim=0) #NOTE: Use average instead of squeeze in case we have multiple n-dit features
        # Unpatching
        out_unpatch = rearrange(
            out_head, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
        )

        # Time-upsampling
        t = out_unpatch.shape[2]
        out_upsample = F.interpolate(
            out_unpatch, size=(t * 4 - 3, out_unpatch.shape[3], out_unpatch.shape[4]),
            mode="trilinear", align_corners=False
        )
        out_upsample = self.conv1(out_upsample)
        # Upsample block
        t = out_upsample.shape[2]
        for layer in self.upsamples:
            if not isinstance(layer, CausalConv3d):
                out_upsample = rearrange(out_upsample, 'b c t h w -> (b t) c h w')
            out_upsample = layer(out_upsample)
            if not isinstance(layer, CausalConv3d):
                out_upsample = rearrange(out_upsample, '(b t) c h w -> b c t h w', t=t)

        out_upsample = rearrange(out_upsample, 'b c t h w -> (b t) c h w')
        joint_map_pred = self.joint_map_conv(out_upsample)
        joint_map_pred = rearrange(joint_map_pred, '(b t) c h w -> b c t h w', t=t)
        pixel_coords, depth = self.map_to_joint(joint_map_pred)
        return pixel_coords, depth


    
    def map_to_joint(self, joint_map):
        """
        Inputs: 
            joint_map: (b, c, t, h, w)
                - c = 2 * J (heatmap and depth channels)
        Returns:
            pixel_coords: (b, J, t, 2) - x,y pixel coordinates
            depth: (b, J, t) - depth values
        """
        b, c, t, h, w = joint_map.shape
        joint_map_list = th.chunk(joint_map, self.J, dim=1)  # List of (b, 2, t, h, w), length J
        all_pixel_coords = []
        all_depths = []
        for map in joint_map_list:
            heatmap = map[:, 0, :, :, :]  # (b, t, h, w)
            depth_map = map[:, 1, :, :, :]  # (b, t, h, w)

            # Softmax over spatial dimensions to get probabilities
            heatmap_flat = heatmap.view(b, t, -1)  # (b, t, h*w)
            prob_map = th.softmax(heatmap_flat, dim=-1).view(b, t, h, w)  # (b, t, h, w)

            # Create coordinate grids
            # y_coords, x_coords = th.meshgrid(th.arange(h, device=joint_map_list[0].device), th.arange(w, device=joint_map_list[0].device), indexing='ij')
            y_coords, x_coords = th.meshgrid(th.linspace(0, 1, h, device=joint_map_list[0].device), th.linspace(0, 1, w, device=joint_map_list[0].device), indexing='ij')

            y_coords = y_coords.view(1, 1, h, w).expand(b, t, h, w)
            x_coords = x_coords.view(1, 1, h, w).expand(b, t, h, w)

            x_pixel = th.sum(prob_map * x_coords, dim=(2, 3))
            y_pixel = th.sum(prob_map * y_coords, dim=(2, 3)) 

            pixel_coords = th.stack([x_pixel, y_pixel], dim=-1)  # (b, t, 2)

            # Compute expected depth
            depth = th.sum(prob_map * depth_map, dim=(2, 3))  # (b, t)
            depth = depth.unsqueeze(-1)  # (b, t, 1)

            all_pixel_coords.append(pixel_coords[:, None, :, :])  # (b, 1, t, 2)
            all_depths.append(depth[:, None, :, :])  # (b, 1, t, 1)
        
        pixel_coords = th.cat(all_pixel_coords, dim=1)  # (b, J, t, 2)
        depth = th.cat(all_depths, dim=1)  # (b, J, t, 1)

        return pixel_coords, depth