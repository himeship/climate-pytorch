#!/usr/bin/env python
# coding: utf-8

# In [0]: Imports


import torch
import einops
import operator
import functools

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import spectral_normalization as sn

from config import configuration
from utils import time2vec as t2v

leaky_alpha = 0.01
config = configuration()
sn_eps = config.sn_eps  # spectral_normalized_eps=0.0001
spec_dtype = config.dtype  # set the specific dtype for weights (the default is None)


# In [1]: D Blocks


class D_Block(nn.Module):
    """D Block for 2D"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', scale_factor=2, device=None, dtype=None):
        
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.dtype = dtype if dtype else torch.float32
        self.device = device if device else torch.device('cpu')        
        
        # plain connection
        self.mid_channels = (in_channels + out_channels) // 2  # the channels in the middle of output & input: (input -> middle -> out)
        self.mainpath = nn.Sequential(
            # 1st conv2d block
            #nn.LeakyReLU(),
            sn.conv2d(in_channels, self.mid_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            nn.LeakyReLU(inplace=False),
            # 2nd conv2d block
            sn.conv2d(self.mid_channels, out_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            sn.avgpool2d(kernel_size=scale_factor, dtype=self.dtype),  # Downsampling
            nn.LeakyReLU(inplace=False),
        )
        
        # skip connection
        self.shortcut = nn.Sequential(
            sn.conv2d(in_channels, out_channels, kernel_size=1, padding=padding, device=self.device, dtype=self.dtype),
            sn.avgpool2d(kernel_size=scale_factor, dtype=self.dtype),  # Downsampling
            nn.LeakyReLU(inplace=False),
        )         
        
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return D_Block(self.in_channels, self.out_channels, self.kernel_size, self.padding, self.scale_factor, device if device else self.device, dtype if dtype else self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mainpath(x) + self.shortcut(x)
        return out
 
        
# In [2]: 3D Blocks


class D3_Block(nn.Module):
    """D Block for 2D"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', scale_factor=2, device=None, dtype=None):
        
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.dtype = dtype if dtype else torch.float32
        self.device = device if device else torch.device('cpu')        
        
        # plain connection
        self.mid_channels = (in_channels + out_channels) // 2  # the channels in the middle of output & input: (input -> middle -> out)
        self.mainpath = nn.Sequential(
            # 1st conv3d block
            #nn.LeakyReLU(),
            sn.conv3d(in_channels, self.mid_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            nn.LeakyReLU(inplace=False),
            # 2nd conv3d block
            sn.conv3d(self.mid_channels, out_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            sn.avgpool3d(kernel_size=scale_factor, dtype=self.dtype),  # Downsampling
            nn.LeakyReLU(inplace=False),
        )
        
        # skip connection
        self.shortcut = nn.Sequential(
            sn.conv3d(in_channels, out_channels, kernel_size=1, padding=padding, device=self.device, dtype=self.dtype),
            sn.avgpool3d(kernel_size=scale_factor, dtype=self.dtype),  # Downsampling
            nn.LeakyReLU(inplace=False),
        )         
        
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return D3_Block(self.in_channels, self.out_channels, self.kernel_size, self.padding, self.scale_factor, device if device else self.device, dtype if dtype else self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mainpath(x) + self.shortcut(x)
        return out
        

# In [3]: G Block


class G_Block(nn.Module):
    """G Block without upsampling"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', device=None, dtype=None):
    
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype if dtype else torch.float32
        self.device = device if device else torch.device('cpu')  
                
        # plain connection
        self.mid_channels = (in_channels + out_channels) // 2  # the channels in the middle of output & input: (input -> middle -> out) 
        self.mainpath = nn.Sequential(
            # 1st conv2d block
            nn.BatchNorm2d(in_channels, device=self.device, dtype=self.dtype),
            nn.LeakyReLU(inplace=False),
            sn.conv2d(in_channels, self.mid_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            # 2nd conv2d block
            nn.BatchNorm2d(self.mid_channels, device=self.device, dtype=self.dtype),
            nn.LeakyReLU(inplace=False),
            sn.conv2d(self.mid_channels, out_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
        )
        
        # skip connection
        self.shortcut = nn.Sequential(
            sn.conv2d(in_channels, out_channels, kernel_size=1, padding=padding, device=self.device, dtype=self.dtype),
        )    
        
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return G_Block(self.in_channels, self.out_channels, self.kernel_size, self.padding, device if device else self.device, dtype if dtype else self.dtype)       
             
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mainpath(x) + self.shortcut(x)
        return out
        

class UpG_Block(nn.Module):
    """G Block with upsampling"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', scale_factor=2, upsample_mode='nearest', device=None, dtype=None):
    
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_mode = upsample_mode
        self.dtype = dtype if dtype else torch.float32
        self.device = device if device else torch.device('cpu')  
                
        # plain connection
        self.mid_channels = (in_channels + out_channels) // 2  # the channels in the middle of output & input: (input -> middle -> out) 
        self.mainpath = nn.Sequential(
            # 1st conv2d block
            nn.BatchNorm2d(in_channels, device=self.device, dtype=self.dtype),
            nn.LeakyReLU(inplace=False),
            sn.upsample(scale_factor=scale_factor, mode=upsample_mode, dtype=self.dtype),  # Upsampling
            sn.conv2d(in_channels, self.mid_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            # 2nd conv2d block
            nn.BatchNorm2d(self.mid_channels, device=self.device, dtype=self.dtype),
            nn.LeakyReLU(inplace=False),
            sn.conv2d(self.mid_channels, out_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
        )
        
        # skip connection
        self.shortcut = nn.Sequential(
            sn.upsample(scale_factor=scale_factor, mode=upsample_mode, dtype=self.dtype),  # Upsampling
            sn.conv2d(in_channels, out_channels, kernel_size=1, padding=padding, device=self.device, dtype=self.dtype),
        )    
    
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return UpG_Block(self.in_channels, self.out_channels, self.kernel_size, self.padding, self.scale_factor, self.upsample_mode, 
            device if device else self.device, dtype if dtype else self.dtype)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mainpath(x) + self.shortcut(x)
        return out


# In [4]: L Block


class L_Block(nn.Module):
    """L Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', device=None, dtype=None):
    
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype if dtype else torch.float32
        self.device = device if device else torch.device('cpu')  
                
        # plain connection
        self.mid_channels = (in_channels + out_channels) // 2  # the channels in the middle of output & input: (input -> middle -> out)
        self.mainpath = nn.Sequential(
            # 1st conv2d block
            sn.conv2d(in_channels, self.mid_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            # 2nd conv2d block
            nn.LeakyReLU(inplace=False),
            sn.conv2d(self.mid_channels, out_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
        )
        
        # skip connection (output channel = c_o - c_i)
        self.shortcut = nn.Sequential(
            sn.conv2d(in_channels, out_channels-in_channels, kernel_size=1, padding=padding, device=self.device, dtype=self.dtype),
        )
        
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return L_Block(self.in_channels, self.out_channels, self.kernel_size, self.padding, device if device else self.device, dtype if dtype else self.dtype)       
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.out_channels > self.in_channels:
            out = self.mainpath(x) + torch.concat([x, self.shortcut(x)], dim=1)
        else:
            out = self.mainpath(x) + x
    
        return out


# In [5]: Attention Block


class Att_Block(nn.Module):
    """Attention Block"""
    def __init__(self, shape, num_heads=8, dropout=0.0, bias=False, device=None, dtype=None):
    
        super().__init__()
        self.bias = bias
        self.shape = shape
        self.dropout = dropout
        self.num_heads = num_heads
        self.dtype = dtype if dtype else torch.float32
        self.device = device if device else torch.device('cpu')
        self.embed_dim = functools.reduce(operator.mul, shape)
        # Query, Key, Value
        self.Q = sn.linear(self.embed_dim, self.embed_dim, device=self.device, dtype=self.dtype)
        self.K = sn.linear(self.embed_dim, self.embed_dim, device=self.device, dtype=self.dtype)
        self.V = sn.linear(self.embed_dim, self.embed_dim, device=self.device, dtype=self.dtype)
        # multi-head attention
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=dropout, bias=bias, device=self.device, dtype=self.dtype)
        
    def forward(self, x):
    
        # x shape: (N, C, H, W) -> (seq_len, N, embed_dim)
        x = einops.rearrange(x, 'n c h w -> c n (h w)')
        
        # query, key, value
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        
        # multi-head attention
        x, _ = self.multihead_attn(q, k, v)  # _ is the attn weights
        x = einops.rearrange(x, 'c n z -> n c z')  # (seq_len, N, embed_dim) -> (N, seq_len, embed_dim)
        out = x.reshape(x.shape[:2] + self.shape)  # (N, seq_len, embed_dim) -> (N, C, H, W)
        
        return out


# In [6|1]: Convolutional GRU (Gated Recurrent Unit)


class ConvGRU(nn.Module):
    """Convolutional GRU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', device=None, dtype=None):
    
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype if dtype else torch.float32
        self.device = device if device else torch.device('cpu')  
        # gate conv layers
        self.reset_gate_conv = sn.conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, device=self.device, dtype=self.dtype)
        self.update_gate_conv = sn.conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, device=self.device, dtype=self.dtype)
        self.output_conv = sn.conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, device=self.device, dtype=self.dtype)
        
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return ConvGRU(self.in_channels, self.out_channels, self.kernel_size, self.padding, device if device else self.device, dtype if dtype else self.dtype)
      
    def forward(self, x, prev_state):
    
        xh = torch.cat([x, prev_state], dim=1)  # shape: (N, Cx+Ch, H, W)
        # reset gate of GRU
        r_t = F.sigmoid(self.reset_gate_conv(xh))  # shape: (N, Cx+Ch, H, W)
        # update gate of GRU
        z_t = F.sigmoid(self.update_gate_conv(xh))  # shape: (N, Cx+Ch, H, W)
        # candidate activation vector
        cand_vec = F.leaky_relu(self.output_conv(torch.cat([x, r_t * prev_state], dim=1)), inplace=False)  # shape: (N, Cx+Ch, H, W)
        # cell state / output
        out = z_t * prev_state + (1.0 - z_t) * cand_vec
        new_state = out
            
        return out, new_state


# In [6|2]: Trajectory GRU (Gated Recurrent Unit)
   
        
class TrajGRU(nn.Module):
    """Trajectory GRU with learned flow-based hidden state warping"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', device=None, dtype=None):
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype if dtype else torch.float32
        self.device = device if device else torch.device('cpu')

        # Gate convolution layers
        self.reset_gate_conv = sn.conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, device=self.device, dtype=self.dtype)
        self.update_gate_conv = sn.conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, device=self.device, dtype=self.dtype)
        self.output_conv = sn.conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, device=self.device, dtype=self.dtype)

        # Flow estimation layer: outputs 2D displacement (dx, dy)
        self.flow_conv = sn.conv2d(in_channels, 2, kernel_size=kernel_size, padding=padding, device=self.device, dtype=self.dtype)

    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return TrajGRU(self.in_channels, self.out_channels, self.kernel_size, self.padding, device if device else self.device, dtype if dtype else self.dtype)

    def _spatial_transform(self, feat, flow):
        feat_f32 = feat.float()
        B, C, H, W = feat_f32.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H, dtype=feat_f32.dtype, device=feat_f32.device), torch.arange(W, dtype=feat_f32.dtype, device=feat_f32.device), indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=0)  # (2, H, W)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)
        grid = grid + flow  # apply learned flow offset

        # Normalize grid to [-1, 1] for grid_sample
        grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0
        grid = grid.permute(0, 2, 3, 1)  # (B, H, W, 2)
        grid = torch.clamp(grid, -1, 1)
        
        warped = F.grid_sample(feat_f32, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return warped.to(dtype=feat.dtype)
        
    def forward(self, x, prev_state):
        xh = torch.cat([x, prev_state], dim=1)  # (N, Cx+Ch, H, W)
        flow = self.flow_conv(xh)  # (N, 2, H, W)
        warped_prev_state = self._spatial_transform(prev_state, flow)

        xh_warped = torch.cat([x, warped_prev_state], dim=1)
        r_t = F.sigmoid(self.reset_gate_conv(xh_warped))
        z_t = F.sigmoid(self.update_gate_conv(xh_warped))
        cand_vec = F.leaky_relu(self.output_conv(torch.cat([x, r_t * warped_prev_state], dim=1)), inplace=False)

        out = z_t * warped_prev_state + (1.0 - z_t) * cand_vec
        new_state = out
        return out, new_state
        

# In [7]: Time Embedding Block


class TE_Block(nn.Module):
    """Time Regime & Hour of Day Embedding"""
    def __init__(self, embed_shape, device=None, dtype=None):
    
        super().__init__()
        self.embed_shape = embed_shape
        self.dtype = dtype if dtype else torch.float32
        self.device = device if device else torch.device('cpu')  
        self.out_features = int(torch.prod(torch.Tensor(embed_shape)))  
        self.mapping = sn.linear(4, self.out_features, bias=False, device=self.device, dtype=self.dtype)     
        
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return TE_Block(self.embed_shape, device if device else self.device, dtype if dtype else self.dtype)   
          
    def forward(self, dates: torch.Tensor) -> torch.Tensor:
        
        # convert dates -> list if dates is torch.Tensor
        #dates = [dt.datetime.fromtimestamp(date) for date in dates.tolist()] if isinstance(dates, torch.Tensor) else dates                    
        #x = t2v(dates, embedding_type='dH', discrete_clocktime=False, normalization=True, device=self.device, dtype=self.dtype)
        d = (dates[:, 0] * 30 + dates[:, 1]) / 365 
        H = (dates[:, 2] * 3600 + dates[:, 3] * 60 + dates[:, 4]) / 86400
        x = torch.stack([d, H], dim=1)
        # T = [sin(2p * d), sin(2p * H), cos(2p * d), cos(2p * H)]
        x = torch.cat([torch.sin(2*torch.pi*x), torch.cos(2*torch.pi*x)], dim=1)  
        x = self.mapping(x)
        out = x.reshape((x.size(0),) + self.embed_shape)
        
        return out


# In [8]: Gradient Calculation Block


class Grad_Block(nn.Module):
    """Gradient of Wind Field"""
    def __init__(self, shape, dx=1000, dy=1000, dz=500, device=None, dtype=None):
    
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.shape = shape
        self.dtype = dtype if dtype else torch.float32
        self.device = device if device else torch.device('cpu')
        
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return Grad_Block(self.shape, self.dx, self.dy, self.dz, device if device else self.device, dtype if dtype else self.dtype)  
        
    def forward(self, var) -> torch.Tensor:
        dvar_dx = torch.gradient(var, spacing=self.dx, dim=-1)[0].to(dtype=self.dtype, device=self.device)  # Derivative with respect to x (longitude)
        dvar_dy = torch.gradient(var, spacing=self.dy, dim=-2)[0].to(dtype=self.dtype, device=self.device)  # Derivative with respect to y (latitude)
        dvar_dz = torch.gradient(var, spacing=self.dz, dim=-3)[0].to(dtype=self.dtype, device=self.device)  # Derivative with respect to z (vertical)
        return torch.stack([dvar_dx, dvar_dy, dvar_dz], dim=1)
   
        
        
        