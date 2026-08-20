#!/usr/bin/env python
# coding: utf-8

# In[0]: Imports


import torch
import einops

import datetime as dt
import torch.nn as nn
import torch.nn.functional as F
import spectral_normalization as sn

from config import configuration
from layers import D_Block, D3_Block, L_Block, G_Block, UpG_Block, Att_Block, TE_Block, ConvGRU


# In[1]: Spatial Discriminator


class Spatial_Discriminator(nn.Module):

    def __init__(
        self,
        p: float = 0.1,  # dropout rate
        scale_factor: int = 2,  # for average pooling
        t_channels: int = 1,
        in_channels: int = 2,
        out_channels: int = 768,
        time_delta: int | float = 5,  # unit is [min]
        conditioning_steps: int = 6,
        forecast_steps: int = 18,
        num_timesteps: int = 18,  # the number of time steps for representation
        shape: (int, int) = (256, 256),
        kernel_size: int or tuple[int, int] = (3, 3),
        padding: int | str = 'same',
        s2d_factor: int = 2, 
        num_blocks: int = 5,  # total number of downsampling D Blocks 
        amp_0: int = 10,  # the ratio of (out_channels & in_channels) of the 1st D Block [amp_0 = 12 // in_channels for recommandation]
        device: str | int | torch.device = None,
        dtype: type | torch.dtype = None,
    ):

        super().__init__()        
        amp_1 = round((out_channels / (in_channels*amp_0*s2d_factor**2)) ** (1/(num_blocks-1)))   # the ratio of (out_channels & in_channels) of the other D Blocks
        # device & dtype
        self.dtype = dtype if dtype else torch.float32
        self.device = torch.device(device) if device else torch.device(type='cpu')
        # layer config
        self.p = p
        self.amp_0 = amp_0
        self.amp_1 = amp_1
        self.padding = padding
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.t_channels = t_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        # architecture def
        self.shape = shape
        self.num_blocks = num_blocks
        self.s2d_factor = s2d_factor
        self.time_delta = time_delta
        self.num_timesteps = num_timesteps
        self.forecast_steps = forecast_steps
        self.conditioning_steps = conditioning_steps
        self.in_channels_conv = in_channels * s2d_factor**2
        self.channels = [self.in_channels_conv] + [self.in_channels_conv*amp_0*(amp_1**i) for i in range(num_blocks-1)] + [out_channels]
        # conv, fc layers & time info embedding
        self.LN = nn.LayerNorm(self.shape, device=self.device, dtype=torch.float32)
        self.time_embed_2d = TE_Block((t_channels,) + shape, device=self.device, dtype=self.dtype)
        self.space2depth = nn.PixelUnshuffle(downscale_factor=s2d_factor)
        self.conv_blocks = nn.ModuleList(
            [D_Block(self.channels[i], self.channels[i+1], kernel_size, padding=padding, scale_factor=scale_factor, device=self.device, dtype=self.dtype) for i in range(num_blocks)] + \
            [D_Block(self.out_channels, self.out_channels, kernel_size, padding=padding, scale_factor=1, device=self.device, dtype=self.dtype)]  # non-downsampling block
        )
        self.fc_stack = nn.Sequential(
            nn.BatchNorm1d(self.out_channels, device=self.device, dtype=torch.float32, track_running_stats=False),
            sn.linear(self.out_channels, self.out_channels*2, device=self.device, dtype=self.dtype),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(self.p),
            sn.linear(self.out_channels*2, self.out_channels//96, device=self.device, dtype=self.dtype),
            nn.Tanh(),
        )
        self.time_embed_1d = TE_Block((self.out_channels//96,), device=self.device, dtype=self.dtype)
        self.final_layer = sn.linear((self.out_channels//96)*2, 1, device=self.device, dtype=self.dtype)
        
    #================================================================================== Fluent Interface ==================================================================================#
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return  Spatial_Discriminator(self.p, self.scale_factor, self.t_channels, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.forecast_steps, 
            self.num_timesteps, self.shape, self.kernel_size, self.padding, self.s2d_factor, self.num_blocks, self.amp_0, device if device else self.device, dtype if dtype else self.dtype)
    # device conversion using method chaining
    def cpu(self):
        return  Spatial_Discriminator(self.p, self.scale_factor, self.t_channels, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.forecast_steps, 
            self.num_timesteps, self.shape, self.kernel_size, self.padding, self.s2d_factor, self.num_blocks, self.amp_0, torch.device(type='cpu'), self.dtype)
    def cuda(self, index: int = None):
        return  Spatial_Discriminator(self.p, self.scale_factor, self.t_channels, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.forecast_steps, 
            self.num_timesteps, self.shape, self.kernel_size, self.padding, self.s2d_factor, self.num_blocks, self.amp_0, torch.device(type='cuda', index=index), self.dtype)
    # dtype conversion using method chaining
    @classmethod
    def create_dtype_convert(cls, name, dtype):
        def func(self):
            return  Spatial_Discriminator(self.p, self.scale_factor, self.t_channels, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.forecast_steps, 
                self.num_timesteps, self.shape, self.kernel_size, self.padding, self.s2d_factor, self.num_blocks, self.amp_0, self.device, dtype)
        setattr(cls, name, func)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#  
            
    def forward(self, x: torch.Tensor, dates=None) -> torch.Tensor:
        # shape of x -> (N, T, C, H, W)
        idxs = torch.randint(low=self.conditioning_steps, high=self.conditioning_steps + self.forecast_steps, size=(self.num_timesteps,)).tolist()
        N, _, C, H, W = x.shape
        T = self.num_timesteps
    
        if dates is None:
            # === Vectorized version for no-date branch ===
            x_selected = x[:, idxs, :, :, :]  # (N, T, C, H, W)
            x_flat = x_selected.view(N * T, C, H, W)
    
            x_flat = self.space2depth(x_flat)
            x_flat = sn.avgpool2d(kernel_size=self.scale_factor, dtype=self.dtype)(x_flat)
    
            for conv_block in self.conv_blocks:
                x_flat = conv_block(x_flat)
    
            x_flat = x_flat.sum(dim=(2, 3))  # (N*T, C)
            #x_flat = x_flat.float()
            mean = x_flat.mean(dim=0, keepdim=True)
            var = x_flat.var(dim=0, keepdim=True, unbiased=False)
            x_flat = (x_flat - mean) / torch.sqrt(var + 1e-5)
            x_out = self.fc_stack[1:](x_flat.to(self.dtype))  # (N*T, fc_dim)
            x_out = x_out.view(N, T, -1)  # (N, T, fc_dim)
            out = x_out.mean(dim=1)  # (N, fc_dim)
            return out
    
        else:
            # === Vectorized version for with-date branch ===
            x_selected = x[:, idxs, :, :, :]  # (N, T, C, H, W)
    
            # Compute all idx_dates at once
            increments = torch.zeros(N, T, 5, device=self.device, dtype=self.dtype)
            increments[:, :, -2] = torch.tensor([self.time_delta * idx for idx in idxs], device=self.device, dtype=self.dtype)
            idx_dates = dates.unsqueeze(1) + increments  # (N, T, 5)
    
            # Time embeddings
            time_embed_2d = self.time_embed_2d(idx_dates.view(-1, 5))  # (N*T, C=1, H, W)
    
            x_flat = x_selected.view(N * T, C, H, W)
            x_flat = torch.cat([x_flat, time_embed_2d], dim=1)  # (N*T, 2, H, W)
    
            # LayerNorm
            #x_flat = x_flat.float()
            x_flat = self.LN(x_flat)
            x_flat = x_flat.to(self.dtype)
    
            x_flat = self.space2depth(x_flat)
            x_flat = sn.avgpool2d(kernel_size=self.scale_factor, dtype=self.dtype)(x_flat)
    
            for conv_block in self.conv_blocks:
                x_flat = conv_block(x_flat)
    
            x_flat = x_flat.sum(dim=(2, 3))  # (N*T, C)
            #x_flat = x_flat.float()
            mean = x_flat.mean(dim=0, keepdim=True)
            var = x_flat.var(dim=0, keepdim=True, unbiased=False)
            x_flat = (x_flat - mean) / torch.sqrt(var + 1e-5)
    
            x_flat = self.fc_stack[1:](x_flat.to(self.dtype))  # (N*T, fc_dim)
    
            # Time embedding (1D)
            time_embed_1d = self.time_embed_1d(idx_dates.view(-1, 5))  # (N*T, C)
            rep = torch.cat([x_flat, time_embed_1d], dim=1)  # (N*T, total_dim)
    
            rep = F.sigmoid(self.final_layer(rep))  # (N*T, 1)
            rep = rep.view(N, T, 1)
            out = rep.mean(dim=1)  # (N, 1)
    
            return out

        
                  
# In[2]: Temporal Discriminator            
            
            
class Temporal_Discriminator(nn.Module):     
            
    def __init__(
        self,
        p: float = 0.1,  # dropout rate
        scale_factor: int = 2,  # for average pooling
        t_channels: int = 1,
        in_channels: int = 2,
        out_channels: int = 768,
        time_delta: int | float = 5,  # unit is [min]
        conditioning_steps: int = 6,
        forecast_steps: int = 18,
        num_timesteps: int = 24,  # the number of time steps for representation
        shape: (int, int) = (256, 256),
        kernel_size: int or tuple[int, int, int] = (1, 3, 3),
        padding: int | str = 'same',
        s2d_factor: int = 2, 
        num_blocks_d: int = 3,  # total number of downsampling D Blocks 
        num_blocks_3d: int = 2,  # total number of downsampling 3D Blocks 
        amp_0: int = 12,  # the ratio of (out_channels & in_channels) of the 1st D Block [amp_0 = 12 // in_channels for recommandation]
        device: str | int | torch.device = None,
        dtype: type | torch.dtype = None,
    ):

        super().__init__()        
        self.num_blocks = num_blocks_d + num_blocks_3d
        amp_1 = round((out_channels / (in_channels*amp_0*s2d_factor**2)) ** (1/(self.num_blocks-1)))   # the ratio of (out_channels & in_channels) of the other D Blocks
        # device & dtype
        self.dtype = dtype if dtype else torch.float32
        self.device = torch.device(device) if device else torch.device(type='cpu')
        # layer config
        self.p = p
        self.amp_0 = amp_0
        self.amp_1 = amp_1
        self.padding = padding
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.t_channels = t_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        # architecture def
        self.shape = shape
        self.s2d_factor = s2d_factor
        self.time_delta = time_delta
        self.num_blocks_d = num_blocks_d
        self.num_blocks_3d = num_blocks_3d
        self.num_timesteps = num_timesteps
        self.forecast_steps = forecast_steps
        self.conditioning_steps = conditioning_steps
        self.in_channels_conv = in_channels * s2d_factor**2
        self.channels = [self.in_channels_conv] + [self.in_channels_conv*amp_0*(amp_1**i) for i in range(self.num_blocks-1)] + [out_channels]
        # conv, fc layers & time info embedding
        self.LN = nn.LayerNorm((num_timesteps,)+shape, device=self.device, dtype=self.dtype)
        self.time_embed_2d = TE_Block((t_channels,) + shape, device=self.device, dtype=self.dtype)
        self.space2depth = nn.PixelUnshuffle(downscale_factor=s2d_factor)
        self.conv_blocks_3d = nn.ModuleList(
            [D3_Block(self.channels[i], self.channels[i+1], kernel_size, padding=padding, scale_factor=scale_factor, device=self.device, dtype=self.dtype) for i in range(num_blocks_3d)]
        )
        self.conv_blocks_d = nn.ModuleList(           
            [D_Block(self.channels[i], self.channels[i+1], kernel_size[1:], padding=padding, scale_factor=scale_factor, device=self.device, dtype=self.dtype) 
                for i in range(num_blocks_3d, self.num_blocks)] + \
            [D_Block(self.out_channels, self.out_channels, kernel_size[1:], padding=padding, scale_factor=1, device=self.device, dtype=self.dtype)]  # non-downsampling block
        )
        self.fc_stack = nn.Sequential(
            nn.BatchNorm1d(self.out_channels, device=self.device, dtype=torch.float32, track_running_stats=False),
            sn.linear(self.out_channels, self.out_channels*2, device=self.device, dtype=self.dtype),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(self.p),
            sn.linear(self.out_channels*2, self.out_channels//96, device=self.device, dtype=self.dtype),
            nn.Tanh(),
        )
        self.time_embed_1d = TE_Block((self.out_channels//96,), device=self.device, dtype=self.dtype)
        self.final_layer = sn.linear((self.out_channels//96), 1, device=self.device, dtype=self.dtype)

    #========================================================================================= Fluent Interface =========================================================================================#
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return  Temporal_Discriminator(self.p, self.scale_factor, self.t_channels, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.forecast_steps, self.num_timesteps,
            self.shape, self.kernel_size, self.padding, self.s2d_factor, self.num_blocks_d, self.num_blocks_3d, self.amp_0, device if device else self.device, dtype if dtype else self.dtype)
    # device conversion using method chaining
    def cpu(self):
        return  Temporal_Discriminator(self.p, self.scale_factor, self.t_channels, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.forecast_steps, self.num_timesteps,
            self.shape, self.kernel_size, self.padding, self.s2d_factor, self.num_blocks_d, self.num_blocks_3d, self.amp_0, torch.device(type='cpu'), self.dtype)
    def cuda(self, index: int = None):
        return  Temporal_Discriminator(self.p, self.scale_factor, self.t_channels, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.forecast_steps, self.num_timesteps,
            self.shape, self.kernel_size, self.padding, self.s2d_factor, self.num_blocks_d, self.num_blocks_3d, self.amp_0, torch.device(type='cuda', index=index), self.dtype)
    # dtype conversion using method chaining
    @classmethod
    def create_dtype_convert(cls, name, dtype):
        def func(self):
            return  Temporal_Discriminator(self.p, self.scale_factor, self.t_channels, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.forecast_steps, 
                self.num_timesteps, self.shape, self.kernel_size, self.padding, self.s2d_factor, self.num_blocks_d, self.num_blocks_3d, self.amp_0, self.device, dtype)
        setattr(cls, name, func)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# 
            
    def forward(self, x: torch.Tensor, dates=None) -> torch.Tensor:
        # shape of x -> (N, T, C, H, W)
        N, _, C, H, W = x.shape
        T = self.num_timesteps
        idx_1st = torch.randint(low=0, high=x.size(1)-self.num_timesteps+1, size=(1,))[0]
        idx_end = idx_1st + self.num_timesteps
        idxs = torch.arange(idx_1st, idx_end).tolist()
        x_slice = x[:, idx_1st:idx_end, :, :, :]
        
        if dates == None:
            x_slice = self.space2depth(x_slice)
            x_slice = sn.avgpool3d(kernel_size=(1, self.scale_factor, self.scale_factor), dtype=self.dtype)(x_slice)  # downsampling
            x_slice = einops.rearrange(x_slice, 'n t c h w -> n c t h w')
            for conv_block in self.conv_blocks_3d:
                x_slice = conv_block(x_slice)   
            x_slice = einops.rearrange(x_slice, 'n c t h w -> n t c h w')
            # seperete conv for each time steps
            N_, T_, C_, H_, W_ = x_slice.size()
            rep = x_slice.contiguous().view(-1, C_, H_, W_)
            for conv_block in self.conv_blocks_d:
                rep = conv_block(rep)
            rep = torch.sum(rep, dim=(2, 3))  # shape of rep -> (N, C) ==default==> (N, 768)
            # Convert to float32 for batch norm
            #rep = rep.float()
            # Apply batch norm manually to avoid inplace operations
            mean = rep.mean(dim=0, keepdim=True)
            var = rep.var(dim=0, keepdim=True, unbiased=False)
            rep = (rep - mean) / torch.sqrt(var + 1e-5)
            # Continue with the rest of fc_stack
            rep = self.fc_stack[1:](rep)  # Skip the BatchNorm1d layer
            #rep = rep.to(self.dtype)  # Convert back to original dtype
                
        else:
            # Compute all idx_dates at once
            increments = torch.zeros(N, T, 5, device=self.device, dtype=self.dtype)
            increments[:, :, -2] = torch.tensor([self.time_delta * idx for idx in idxs], device=self.device, dtype=self.dtype)
            idx_dates = dates.unsqueeze(1) + increments  # (N, T, 5)
            # Time embeddings
            time_embed_2d = self.time_embed_2d(idx_dates.view(-1, 5))  # (N*T, C=1, H, W)
            time_embed_2d = time_embed_2d.view(N, T, 1, H, W) 
            #embeds = torch.stack(embeds, dim=1) # embeds shape : (N, t_steps, t_channels, h, w) ==default==> (N, 22, 1, 256, 256)
            x_cat = torch.cat([x_slice, time_embed_2d], dim=2)  # Create a new tensor instead of modifying x
            x_cat = einops.rearrange(x_cat, 'n t c h w -> n c t h w')
            x_cat = self.LN(x_cat)  # maintain numerical stability
            x_cat = einops.rearrange(x_cat, 'n c t h w -> n t c h w')   
            x_slice = x_cat[:, idx_1st:idx_end, :, :, :]  # Create a new tensor
            x_slice = self.space2depth(x_slice)
            x_slice = sn.avgpool3d(kernel_size=(1, self.scale_factor, self.scale_factor), dtype=self.dtype)(x_slice)  # downsampling
            x_slice = einops.rearrange(x_slice, 'n t c h w -> n c t h w')
            for conv_block in self.conv_blocks_3d:
                x_slice = conv_block(x_slice)  
            x_slice = einops.rearrange(x_slice, 'n c t h w -> n t c h w')
            # seperete conv for each time steps
            N_, T_, C_, H_, W_ = x_slice.size()
            rep = x_slice.contiguous().view(-1, C_, H_, W_)
            for conv_block in self.conv_blocks_d:
                rep = conv_block(rep)
            rep = torch.sum(rep, dim=(2, 3))  # shape of rep -> (N, C) ==default==> (N, 768)
            # Convert to float32 for batch norm
            #rep = rep.float()
            # Apply batch norm manually to avoid inplace operations
            mean = rep.mean(dim=0, keepdim=True)
            var = rep.var(dim=0, keepdim=True, unbiased=False)
            rep = (rep - mean) / torch.sqrt(var + 1e-5)
            # Continue with the rest of fc_stack
            rep = self.fc_stack[1:](rep)  # Skip the BatchNorm1d layer
            #rep = rep.to(self.dtype)  # Convert back to original dtype
            rep = F.sigmoid(self.final_layer(rep))
              
        x = rep.view(N, -1)  # shape of x -> (N, T, 1)
        out = torch.mean(x, dim=1, keepdim=True)  # shape of out -> (N, 1)
        
        return out   

        
        
# In[3]: Discriminator   


class Discriminator(nn.Module): 

    def __init__(
        self,
        spatial_discriminator: nn.Module,
        temporal_discriminator: nn.Module,
        device: str | int | torch.device = None,
        dtype: type | torch.dtype = None,
        ):
        
        super().__init__()
        # device & dtype (default device & dtype is consistent with spatial_discriminator)
        self.dtype = dtype if dtype else spatial_discriminator.dtype
        self.device = torch.device(device) if device else spatial_discriminator.device
        # disc
        self.spatial_discriminator = spatial_discriminator.to(self.device, self.dtype)
        self.temporal_discriminator = temporal_discriminator.to(self.device, self.dtype)
        
    #===================================================================================== Fluent Interface =====================================================================================#
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return  Discriminator(self.spatial_discriminator, self.temporal_discriminator, device if device else self.device, dtype if dtype else self.dtype)
    # device conversion using method chaining
    def cpu(self):
        return  Discriminator(self.spatial_discriminator, self.temporal_discriminator, torch.device(type='cpu'), self.dtype)
    def cuda(self, index: int = None):
        return  Discriminator(self.spatial_discriminator, self.temporal_discriminator, torch.device(type='cuda', index=index), self.dtype)
    # dtype conversion using method chaining
    @classmethod
    def create_dtype_convert(cls, name, dtype):
        def func(self):
            return  Discriminator(self.spatial_discriminator, self.temporal_discriminator, self.device, dtype)
        setattr(cls, name, func)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# 
        
    def forward(self, x: torch.Tensor, dates=None) -> torch.Tensor:
        #dates = [dt.datetime.fromtimestamp(date) for date in dates.tolist()] if isinstance(dates, torch.Tensor) else dates  # convert dates -> list if dates is torch.Tensor
        return torch.cat([self.spatial_discriminator(x, dates), self.temporal_discriminator(x, dates)], dim=1)
        
            
# In [*]: Fluent Interface for Dtype Conversion
        
        
# Define the dtype name convention
dtype_name_convention = [
    ('half', torch.float16),          #      
    ('float', torch.float32),         #
    ('double', torch.float64),        #
    ('chalf', torch.complex32),       #      
    ('cfloat', torch.complex64),      #
    ('cdouble', torch.complex128),    #
    ('bfloat16', torch.bfloat16),     #
]

# Create the dtype conversion methods for the class
for name, dtype in dtype_name_convention:
    Discriminator.create_dtype_convert(name, dtype)
    Spatial_Discriminator.create_dtype_convert(name, dtype)
    Temporal_Discriminator.create_dtype_convert(name, dtype)
    