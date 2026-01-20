import torch as th
from diffsynth.models.wan_video_dit import SelfAttention, rearrange, precompute_freqs_cis_3d, precompute_freqs_cis, sinusoidal_embedding_1d
from diffsynth.models.wan_video_vae import Decoder3d, WanVideoVAE
import tqdm

class JointHeatMapMotionVAEDecoder(th.nn.Module):
    def __init__(self, n_joints, dit_dim, head_out_dim, flatten_dim, vae_latent_dim, patch_size, grid_size, device, out_channels=2, num_heads=12):
        super().__init__()
        self.J = n_joints
        self.dit_dim = dit_dim
        self.vae_latent_dim = vae_latent_dim
        self.out_joint_map_channels = out_channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        #TODO: Figure out the flatten dim calculation from flattened dit features
        self.flatten_dim = flatten_dim
        
        self.self_attn = SelfAttention(dim=dit_dim, num_heads=num_heads).to(device)
        self.decoder = Decoder3d().to(device)
        self.ffn_out = th.nn.Linear(dit_dim, head_out_dim).to(device)
        self.joint_head = th.nn.Linear(flatten_dim, flatten_dim * self.out_joint_map_channels)
        self.device = device


    
    def forward(self, dit_features, grid_size):
        """
        dit_features: (n, b, f, d); 
            - n: num of features from DIT blocks (#Preferred dit block id * #Preferred timesteps),
            - b: batch size,
            - f: flattened spatial-temporal dimension (num_frames * height * width),
            - d: dit feature dimension (usually 1536)
        """
        head_dim = self.dit_dim // self.num_heads
        x = dit_features
        x_flat = rearrange(x, 'n b f d -> b (n f) d', n=self.J)

        # Self-attention on dit features + 1D RoPE
        freqs = precompute_freqs_cis(head_dim, x_flat.shape[1], theta=1e4).to(self.device)
        freqs = freqs[None, :, None, :]
        attn_flat = self.self_attn(x_flat, freqs_cis=freqs)
        out_attn = rearrange(attn_flat, 'b (n f) d -> n b f d', n=x.shape[0], f=x.shape[2])
        # DIT head
        out_ffn = self.ffn_out(out_attn)
        out_ffn = rearrange(out_ffn, 'n b f d -> n b d f')
        joint_map_pred = self.joint_head(out_ffn)
        joint_map_pred = rearrange(joint_map_pred, 'n b d (j f) -> n b j f d', j=self.n_joints)
        joint_map_unpatch = rearrange(
            joint_map_pred, 'b j (f h w) (x y z c) -> b j c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2], j = self.J
        )
        final_joint_map = []
        for i in tqdm.tqdm(range(self.J), desc="Decoding joint heat maps"):
            joint_heat_map = self.video_vae.decode(joint_map_unpatch[0:1, i, ...], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            final_joint_map.append(joint_heat_map)

        return final_joint_map
        








