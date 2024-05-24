# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch, sys
import torch.nn as nn
import math, os
import numpy as np
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
current_file_directory = os.path.dirname(os.path.abspath(__file__))
from flash_attn import (
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func
    )
from einops import rearrange, reduce, repeat
#import utils.checkpoint as cp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        attn_output = flash_attn_qkvpacked_func(
                qkv,
                self.drop,
                softmax_scale=self.scale
            )
        x = attn_output.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.drop = attn_drop
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wkv = nn.Linear(context_dim if context_dim else dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, spatial=True):
        P = 700
        if spatial: #spatial attention between SMPL and video patches
            B, T, C = x.shape
            x = rearrange(x, 'b t c -> (b t) c',b=B,t=T)
            x = x.unsqueeze(1)
            N, S, _ = x.shape
            q = self.wq(x).view(N, S, self.num_heads, C // self.num_heads)
            kv = self.wkv(context).view(N, P, 2, self.num_heads, C // self.num_heads)
            x = flash_attn_kvpacked_func(
                    q,
                    kv,
                    self.drop,
                    softmax_scale=self.scale
                )
            x = x.reshape(N, C)
            x = rearrange(x, '(b t) c -> b t c',b=B,t=T)
        else: #temporal attention between SMPL and video frames
            B, T, C = x.shape
            x = x.unsqueeze(2).repeat(1, 1, P, 1)
            x = rearrange(x, 'b t n m -> (b n) t m',b=B,t=T,n=P,m=C)
            N, _, _ = x.shape
            context = rearrange(context, '(b t) n m -> (b n) t m',b=B,t=T,n=P,m=C)
            q = self.wq(x).view(N, T, self.num_heads, C // self.num_heads)
            kv = self.wkv(context).view(N, T, 2, self.num_heads, C // self.num_heads)
            x = flash_attn_kvpacked_func(
                    q,
                    kv,
                    self.drop,
                    softmax_scale=self.scale
                )
            x = x.reshape(N, T, C)
            x = rearrange(x, '(b n) t m -> b t n m',b=B,t=T,n=P,m=C)
            x = x.mean(dim=2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, context_dim=hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, context, spatial=True):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) #self attention with SMPL params
        x = x + self.cross_attn(self.norm2(x), context, spatial)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model (modified) with a Transformer backbone.
    """
    def __init__(
        self,
        hidden_size=1152,
        depth=28,
        seqLen=600,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_heads = num_heads
        in_channels = 161
        self.seqLen = seqLen
        self.hidden_size = hidden_size
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        # self.pos_y_embed = nn.Parameter(torch.zeros(1, 700, hidden_size), requires_grad=False)
        # Initialize positional encoding:
        self.x_embedder = nn.Linear(in_channels, hidden_size) #4 + 3 + 144 + 10 = 161; camera + bbox + pose + shape + betas (aka SMPL vector)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.time_embed = nn.Parameter(torch.zeros(1, seqLen, hidden_size), requires_grad=False)
        self.time_drop = nn.Dropout(p=0)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # grid_patches = np.arange(700)
        #pos_y_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, grid_patches)
        #self.pos_y_embed.data.copy_(torch.from_numpy(pos_y_embed).float().unsqueeze(0)) #unsure if you need this as DINOv2 may already have pos embedding for patches
        
        grid_num_frames = np.arange(self.seqLen)
        time_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, grid_num_frames)
        self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, P, F) tensor of motion latents where F is num frames, P is vector representation of motion_i
        t: (N,) tensor of diffusion timesteps
        y: (N, D, F) tensor of video frames with context prepended, where D is DINOv2 (1024 for DINOV2-l) dimension
        """
        x = x.permute(0, 2, 1) #N, P, F --> N, F, P
        x = self.x_embedder(x)  
        x = x + self.time_embed #add pos embedding to vectorized SMPL latent vectors (N, F, P)
        
        #spatial embedding of the given patches
        B, T, P, D = y.shape #F is T
        y = y.contiguous().view(-1, P, D)
        #y = y + self.pos_y_embed #no need to positionally embed the patches as DINOv2 already has patch embeddings
        
        #temporal embedding of the frames
        y = rearrange(y, '(b t) n m -> (b n) t m',b=B,t=T)
        y = y + self.time_embed
        y = self.time_drop(y)
        y = rearrange(y, '(b n) t m -> (b t) n m',b=B,t=T)
        t = self.t_embedder(t)                   # (N, D) 
        #condition on timestep, cross attn on video frames
        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(spatial_block), x, t, y.to(x.device), True) 
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(temp_block), x, t, y.to(x.device), False)
        x = self.final_layer(x, t)
        x = x.permute(0, 2, 1) #N, F, P --> N, P, F
        return x


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_action_model(**kwargs):
    return DiT(depth=24, hidden_size=1024, seqLen=1100, num_heads=16, **kwargs) #large model feature vectors (via dinov2 large embeddings)

DiT_action = DiT_action_model
