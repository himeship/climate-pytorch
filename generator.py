#!/usr/bin/env python
# coding: utf-8

# In [0]: Imports


import torch
import einops
import warnings

import datetime as dt
import torch.nn as nn
import torch.nn.functional as F
import spectral_normalization as sn
import torch.distributions.normal as normal

from layers import D_Block, D3_Block, L_Block, G_Block, UpG_Block, Att_Block, TE_Block, ConvGRU, Grad_Block


# In [1]: Conditioning Stack


class Conditioning_stack(nn.Module):

    def __init__(
        self,
        scale_factor: int = 2,  # for average pooling
        in_channels: int = 12,
        out_channels: int = 384,
        time_delta: int | float = 5,  # unit is [min]
        conditioning_steps: int = 6,
        shape: (int, int) = (256, 256),
        kernel_size: int or tuple[int, int] = (3, 3),
        padding: int | str = 'same',
        s2d_factor: int = 2, 
        num_blocks: int = 4,  # total number of D Blocks 
        amp_0: int = 12,  # the ratio of (out_channels & in_channels) of the 1st D Block [amp_0 = 12 // in_channels for recommandation]
        num_attn_heads: int = 4,
        device: str | int | torch.device = None,
        dtype: type | torch.dtype = None,
    ):
        
        super().__init__()        
        amp_1 = round((out_channels / (in_channels*amp_0*s2d_factor**2)) ** (1/(num_blocks-1)))   # the ratio of (out_channels & in_channels) of the other D Blocks
        # device & dtype
        self.dtype = dtype if dtype else torch.float32
        self.device = torch.device(device) if device else torch.device(type='cpu')
        # layer config
        self.amp_0 = amp_0
        self.amp_1 = amp_1
        self.padding = padding
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        # architecture def
        self.shape = shape
        self.num_blocks = num_blocks
        self.s2d_factor = s2d_factor
        self.time_delta = time_delta
        self.num_attn_heads = num_attn_heads
        self.conditioning_steps = conditioning_steps
        # cell def
        self.space2depth = nn.PixelUnshuffle(downscale_factor=s2d_factor)
        self.in_channels_conv = in_channels * s2d_factor**2
        self.channels = [self.in_channels_conv] + [min(self.out_channels//2**(num_blocks-1-i), self.in_channels_conv*amp_0*(amp_1**i)) for i in range(num_blocks-1)] + [self.out_channels]
        # all the convolutional layers
        self.conv2d_layers = nn.ModuleList(
            [
                sn.conv2d(
                    in_channels=self.channels[i+1]*self.conditioning_steps, 
                    out_channels=self.channels[i+1], 
                    kernel_size = self.kernel_size, 
                    padding=self.padding,
                    device=self.device,
                    dtype=self.dtype,
                ) for i in range(num_blocks)
            ]
        )
        self.attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=self.channels[i+1], num_heads=self.num_attn_heads, batch_first=True, device=self.device, dtype=self.dtype) for i in range(num_blocks)]
        )   
        self._blocks = nn.ModuleList(
            [D_Block(self.channels[i], self.channels[i+1], kernel_size, padding=padding, scale_factor=scale_factor, device=self.device, dtype=self.dtype) for i in range(num_blocks)]
        )
        
    #============================================================================= Fluent Interface =============================================================================#
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return  Conditioning_stack(self.scale_factor, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.shape, self.kernel_size, self.padding, 
            self.s2d_factor, self.num_blocks, self.amp_0, self.num_attn_heads, device if device else self.device, dtype if dtype else self.dtype)
    # device conversion using method chaining
    def cpu(self):
        return  Conditioning_stack(self.scale_factor, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.shape, self.kernel_size, self.padding, 
            self.s2d_factor, self.num_blocks, self.amp_0, self.num_attn_heads, torch.device(type='cpu'), self.dtype)
    def cuda(self, index: int = None):
        return  Conditioning_stack(self.scale_factor, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.shape, self.kernel_size, self.padding, 
            self.s2d_factor, self.num_blocks, self.amp_0, self.num_attn_heads, torch.device(type='cuda', index=index), self.dtype)
    # dtype conversion using method chaining
    @classmethod
    def create_dtype_convert(cls, name, dtype):
        def func(self):
            return Conditioning_stack(self.scale_factor, self.in_channels, self.out_channels, self.time_delta, self.conditioning_steps, self.shape, self.kernel_size, self.padding, 
                self.s2d_factor, self.num_blocks, self.amp_0, self.num_attn_heads, self.device, dtype)
        setattr(cls, name, func)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#   
                                
    def forward(self, x: torch.Tensor) -> list:
        
        # raise values or warnings to inconsistant input and pre-defined conditioning time steps
        if x.size(1) < self.conditioning_steps:
            raise ValueError(f"The time steps of the input conditioning tensor ({x.size(1)}) shall not be less than the pre-defined conditioning steps ({self.conditioning_steps})")
        elif x.size(1) > self.conditioning_steps:
            warnings.warn(
                f"The time steps of the input conditioning tensor ({x.size(1)}) is greater than conditioning steps ({self.conditioning_steps}). "
                f"Only the last {self.conditioning_steps} will be taken into calcultaion",
                UserWarning
            )
            
        hidden_states_1_step_list = []  # list containing multiple "hidden_states_1_step"
        
        for i in range(self.conditioning_steps):  # loop for every time step
            hidden_states_1_step = []  # hidden conditioning states to input into sampler (for 1 time step)
            out = self.space2depth(x[:, i+x.size(1)-self.conditioning_steps, :, :, :])
            # loop for all layers
            for j, _block in enumerate(self._blocks):
                out = _block(out)
                hidden_states_1_step.append(out.unsqueeze(1))
            # store the out from same time step together
            hidden_states_1_step_list.append(hidden_states_1_step) # --> [ [(t1b1),(t1b2),...,(t1b4)],  [(t2b1),(t2b1),...,(t2b4)],  ...,  [(t4b1),(t4b2),...,(t4b4)] ] 
            
        hidden_states = [torch.cat([state[i] for state in hidden_states_1_step_list], dim=1) for i in range(self.num_blocks)]
        # Convert from [batch_size, time, c, h, w] -> [batch_size, time * c, h, w]
        # then perform "convolution & ReLU" on the output while preserving number of c.
        for i, states in enumerate(hidden_states):
            N, T, C, H, W = states.shape
            states = states.permute(0, 3, 4, 2, 1).reshape(-1, T, C)  # (N*H*W, T, C)
            states, _ = self.attn_layers[i](states, states, states)
            states = states.reshape(N, H, W, T, C).permute(0, 3, 4, 1, 2)  # (N, T, C, H, W)
            stack_states = einops.rearrange(states, 'n t c h w -> n (t c) h w')
            hidden_states[i] = F.leaky_relu(self.conv2d_layers[i](stack_states), inplace=False)
        
        # channels in [hidden_states] ascending with larger i (e.g., 48 -> 96 -> 192 -> 384)
        return hidden_states[::-1]  # shape: [(N, C, H, W), ... , (N, C, H, W)] ==default==> [(N, 384, 8, 8), (N, 192, 16, 16), (N, 96, 32, 32), (N, 48, 64, 64)]


# In [2]: Latent Conditioning Stack


class Latent_stack(nn.Module):

    def __init__(
        self,
        use_date: bool=True,
        t_channels: int = 2,
        in_channels: int = 8,
        out_channels: int = 768,
        num_attn_heads: int = 8,
        shape: (int, int) = (8, 8),
        kernel_size: int or (int, int) = (3, 3),
        padding: int | str = 'same',
        sampling_loc: int | float = 0.0,
        sampling_scale: int | float = 1.0,
        device: str | torch.device = None,
        dtype: torch.dtype = None,
    ):
        
        super().__init__()
        # device & dtype
        self.dtype = dtype if dtype else torch.float32
        self.device = torch.device(device) if device else torch.device(type='cpu')
        # time regime embedding
        self.use_date = use_date
        # layer config
        self.shape = shape
        self.padding = padding
        self.kernel_size = kernel_size
        self.t_channels = t_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_attn_heads = num_attn_heads
        # sampling policy -> N(0,1) normal distritution
        self.sampling_loc = sampling_loc
        self.sampling_scale = sampling_scale
        self.distribution = normal.Normal(loc=sampling_loc, scale=sampling_scale)
        # time regime embedding def
        self.time_embed = TE_Block((t_channels,) + shape, device=self.device, dtype=self.dtype) 
        # block def
        self.LN = nn.LayerNorm(self.shape, device=self.device, dtype=self.dtype)
        self._stack = nn.Sequential(
            sn.conv2d(in_channels, in_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            L_Block(in_channels, out_channels//32, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            L_Block(out_channels//32, out_channels//16, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            L_Block(out_channels//16, out_channels//4, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
            Att_Block(shape, num_heads=num_attn_heads, device=self.device, dtype=self.dtype),
            L_Block(out_channels//4, out_channels, kernel_size, padding=padding, device=self.device, dtype=self.dtype),
        )

    #================================================================================= Fluent Interface =================================================================================#
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return Latent_stack(self.use_date, self.t_channels, self.in_channels, self.out_channels, self.num_attn_heads, self.shape, self.kernel_size, self.padding, self.sampling_loc,
            self.sampling_scale, device if device else self.device, dtype if dtype else self.dtype)
    # device conversion using method chaining
    def cpu(self):
        return Latent_stack(self.use_date, self.t_channels, self.in_channels, self.out_channels, self.num_attn_heads, self.shape, self.kernel_size, self.padding, self.sampling_loc,
            self.sampling_scale, torch.device(type='cpu'), self.dtype)
    def cuda(self, index: int = None):
        return Latent_stack(self.use_date, self.t_channels, self.in_channels, self.out_channels, self.num_attn_heads, self.shape, self.kernel_size, self.padding, self.sampling_loc,
            self.sampling_scale, torch.device(type='cuda', index=index), self.dtype)
    # dtype conversion using method chaining
    @classmethod
    def create_dtype_convert(cls, name, dtype):
        def func(self):
            return Latent_stack(self.use_date, self.t_channels, self.in_channels, self.out_channels, self.num_attn_heads, self.shape, self.kernel_size, self.padding, self.sampleing_loc,
                self.sampling_scale, self.device, dtype)
        setattr(cls, name, func)
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#      
        
    def forward(self, x: torch.Tensor, dates=None) -> torch.Tensor:
    
        if dates == None:
            z = self.distribution.sample((x.size(0), self.in_channels) + self.shape).to(self.device, dtype=self.dtype)  # z shape: (N, C, H, W) ==default==> (N, 8, 8, 8)
            out = self._stack(z)
        
        else:
            z = self.distribution.sample((x.size(0), self.in_channels-self.t_channels) + self.shape).to(self.device, dtype=self.dtype)
            embed = self.time_embed(dates)
            z = torch.cat([z, embed], dim=1)
            #z = nn.LayerNorm(z.shape[-2:], device=self.device, dtype=self.dtype)(z)
            z = self.LN(z)
            out = self._stack(z)
        
        return out  # shape: (N, C, H, W) ==default==> (N, 768, 8, 8)
                           
                    
# In [4]: Sampler


class Sampler(nn.Module):

    def __init__(
        self,
        conditioning_stack: nn.Module,
        latent_stack: nn.Module,
        forecast_steps: int = 18,
        in_channels: int = 768,
        out_channels: int = 1,
        kernel_size: int or (int, int) = (3, 3),
        padding: int | str = 'same',
        device: str | torch.device = None,
        dtype: torch.dtype = None,
    ):
        
        super().__init__()
        # device & dtype
        self.dtype = dtype if dtype else torch.float32
        self.device = torch.device(device) if device else torch.device(type='cpu')
        # layer config
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d2s_factor = conditioning_stack.s2d_factor
        self.out_channels_conv = out_channels * self.d2s_factor**2
        # info from other stacks
        self.latent_stack = latent_stack
        self.in_shape = latent_stack.shape
        self.out_shape = conditioning_stack.shape
        self.conditioning_stack = conditioning_stack
        self.num_blocks = conditioning_stack.num_blocks
        self.scale_factor = conditioning_stack.scale_factor
        self.hidden_channels = conditioning_stack.channels[::-1][:-1]
        self.channels = [in_channels] + conditioning_stack.channels[::-1][:-1] + [out_channels * self.d2s_factor**2] + [out_channels]
        self.conditioning_steps = conditioning_stack.conditioning_steps
        self.time_delta = conditioning_stack.time_delta
        # cell def
        self.depth2space = nn.PixelShuffle(upscale_factor=self.d2s_factor)
        # model architecture
        self.forecast_steps = forecast_steps
        
        # block def (each block contains "[ConvGRU + conv1x1 + G + G']" + "BN + out_conv")
        # e.g. for (c, h, w) evolution: (768, 8, 8) ==ConvGRU==> (384, 8, 8) ==conv_1x1==> (768, 8, 8) ==G==> (384, 8, 8) ==G'==> (384, 16, 16)    <--    {1st stack in sampler}
        
        # [ConvGRU] -> (torch.Tensor, torch.Tensor)
        self._ConvGRU_blocks = nn.ModuleList(
            [
                ConvGRU(self.channels[i]+self.hidden_channels[i], self.hidden_channels[i],  kernel_size=kernel_size, padding=padding, device=self.device, dtype=self.dtype)
                for i in range(self.num_blocks)
            ]
        )
        # [TrajGRU] <-> [ConvGRU]
#        self._TrajGRU_blocks = nn.ModuleList(
#            [
#                TrajGRU(self.channels[i]+self.hidden_channels[i], self.hidden_channels[i],  kernel_size=kernel_size, padding=padding, device=self.device, dtype=self.dtype)
#                for i in range(self.num_blocks)
#            ]
#        ) 
        # [conv_1x1 + G + G'] -> torch.Tensor
        self._conv1x1_G_upG_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    sn.conv2d(self.hidden_channels[i], self.channels[i], kernel_size=(1, 1), padding='same', device=self.device, dtype=self.dtype),
                    G_Block(self.channels[i], self.channels[i+1], kernel_size=kernel_size, padding=padding, device=self.device, dtype=self.dtype),
                    UpG_Block(self.channels[i+1], self.channels[i+1], kernel_size=kernel_size, padding=padding, scale_factor=self.scale_factor, device=self.device, dtype=self.dtype),
                ) for i in range(self.num_blocks)
            ]
        )
        # BN + out_conv
        self._out_block = nn.Sequential(
            nn.BatchNorm2d(self.channels[-3], device=self.device, dtype=self.dtype),  # result in no bias
            sn.conv2d(self.channels[-3], self.channels[-2], kernel_size=kernel_size, padding=padding, bias=True, device=self.device, dtype=self.dtype),
            nn.ReLU(inplace=False),
        )
        
    #=============================================================================== Fluent Interface ==============================================================================#
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return Sampler(self.conditioning_stack, self.latent_stack, self.forecast_steps, self.in_channels, self.out_channels, self.kernel_size, self.padding,
            device if device else self.device, dtype if dtype else self.dtype)
    # device conversion using method chaining
    def cpu(self):
        return Sampler(self.conditioning_stack, self.latent_stack, self.forecast_steps, self.in_channels, self.out_channels, self.kernel_size, self.padding,
            torch.device(type='cpu'), self.dtype)
    def cuda(self, index: int = None):
        return Sampler(self.conditioning_stack, self.latent_stack, self.forecast_steps, self.in_channels, self.out_channels, self.kernel_size, self.padding,
            torch.device(type='cuda', index=index), self.dtype)
    # dtype conversion using method chaining
    @classmethod
    def create_dtype_convert(cls, name, dtype):
        def func(self):
            return Sampler(self.conditioning_stack, self.latent_stack, self.forecast_steps, self.in_channels, self.out_channels, self.kernel_size, self.padding, self.device, dtype)
        setattr(cls, name, func)
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#      
          
    # the channels of the input [hidden_states] shall be descending with larger i (e.g., 384 -> 192 -> 96 -> 48), which is the reverse of the output from the Conditioning Stack
    def forward(self, x: torch.Tensor, init_states) -> (torch.Tensor, list):
      
        new_states = []  # to store state outputs from ConvGRU
        
        for i in range(self.num_blocks):
            x, state = self._ConvGRU_blocks[i](x, init_states[i])
            #x, state = tuple(a + b for a, b in zip(self._ConvGRU_blocks[i](x, init_states[i]), self._TrajGRU_blocks[i](x, init_states[i])))
            #x, state = tuple(torch.cat((a, b), dim=1) for a, b in zip(self._ConvGRU_blocks[i](x, init_states[i]), self._TrajGRU_blocks[i](x, init_states[i])))
            #N, C, H, W = x.shape
            #x, state = x.view(N, C//2, 2, H, W).mean(dim=2), state.view(N, C//2, 2, H, W).mean(dim=2)
            x = self._conv1x1_G_upG_blocks[i](x)
            new_states.append(state)
            
        x = self._out_block(x)
        out = self.depth2space(x).unsqueeze(1)  # shape: (N, C, H, W) -> (N, 1[T], C, H, W)
        
        # [new_states] channels is descending (e.g., 384 -> 192 -> 96 -> 48)
        return out, new_states
        
        
# In [5]: Wind Conditioning Stack


class Wind_stack(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        in_shape: (int, int) = (128, 128),
        out_shape: (int, int) = (256, 256),
        num_blocks_3d: int = 3,
        mid_channel_wind: int = 3,
        mid_channel_grad: int = 3, 
        conv_kernel_size: int or (int, int, int) = (3, 3, 3),
        pool_kernel_size: int or (int, int, int) = (2, 1, 1),
        conv_kernel_size_final: int or (int, int) = (3, 3),
        pool_kernel_size_final: int or (int, int) = (1, 1),
        padding: int | str = 'same',
        device: str | torch.device = None,
        dtype: torch.dtype = None,
    ):
    
        super().__init__()
        # device & dtype
        self.dtype = dtype if dtype else torch.float32
        self.device = torch.device(device) if device else torch.device(type='cpu')
        # architecture
        self.padding = padding
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks_3d = num_blocks_3d
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.mid_channel_wind = mid_channel_wind
        self.mid_channel_grad = mid_channel_grad
        self.conv_kernel_size_final = conv_kernel_size_final
        self.pool_kernel_size_final = pool_kernel_size_final
        self.channels_wind = [3,] + [self.mid_channel_wind,]*(self.num_blocks_3d-1) + [1,]
        self.channels_grad = [9,] + [self.mid_channel_grad,]*(self.num_blocks_3d-1) + [3,]
        self.grad_blocks = Grad_Block(self.in_shape, device=self.device, dtype=self.dtype)
        self.conv_blocks_wind = nn.ModuleList(
            [
                D3_Block(
                    self.channels_wind[i], self.channels_wind[i+1], self.conv_kernel_size,
                    padding=self.padding, scale_factor=self.pool_kernel_size, device=self.device, dtype=self.dtype
                )
                for i in range(self.num_blocks_3d)
            ]
        )
        self.conv_blocks_grad = nn.ModuleList(
            [
                D3_Block(
                    self.channels_grad[i], self.channels_grad[i+1], self.conv_kernel_size, 
                    padding=self.padding, scale_factor=self.pool_kernel_size, device=self.device, dtype=self.dtype
                ) 
                for i in range(self.num_blocks_3d)
            ]
        )
        self.final_conv_block = nn.Sequential(
            D_Block(
                self.channels_wind[-1]+self.channels_grad[-1], self.out_channels, self.conv_kernel_size_final, 
                padding=self.padding, scale_factor=self.pool_kernel_size_final, device=self.device, dtype=self.dtype
            ),
            sn.upsample(scale_factor=round(self.out_shape[0]/self.in_shape[0]), mode='bilinear', dtype=self.dtype)
        )     
        
    #================================================================================= Fluent Interface ================================================================================#
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)
        return Wind_stack(self.in_channels, self.out_channels, self.in_shape, self.out_shape, self.num_blocks_3d, self.mid_channel_wind, self.mid_channel_grad, self.conv_kernel_size, 
            self.pool_kernel_size, self.conv_kernel_size_final, self.pool_kernel_size_final, self.padding, device if device else self.device, dtype if dtype else self.dtype)
    # device conversion using method chaining
    def cpu(self):
        return Wind_stack(self.in_channels, self.out_channels, self.in_shape, self.out_shape, self.num_blocks_3d, self.mid_channel_wind, self.mid_channel_grad, self.conv_kernel_size,
            self.pool_kernel_size, self.conv_kernel_size_final, self.pool_kernel_size_final, self.padding, torch.device(type='cpu'), self.dtype)
    def cuda(self, index: int = None):
        return Wind_stack(self.in_channels, self.out_channels, self.in_shape, self.out_shape, self.num_blocks_3d, self.mid_channel_wind, self.mid_channel_grad, self.conv_kernel_size,
            self.pool_kernel_size, self.conv_kernel_size_final, self.pool_kernel_size_final, self.padding, torch.device(type='cuda', index=index), self.dtype)
    # dtype conversion using method chaining
    @classmethod
    def create_dtype_convert(cls, name, dtype):
        def func(self):
            return Wind_stack(self.in_channels, self.out_channels, self.in_shape, self.out_shape, self.num_blocks_3d, self.mid_channel_wind, self.mid_channel_grad, self.conv_kernel_size,
                self.pool_kernel_size, self.conv_kernel_size_final, self.pool_kernel_size_final, self.padding, self.device, dtype)
        setattr(cls, name, func)
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# 
        
    def forward(self, v: torch.Tensor):
        
        # calculate wind gradient
        vx_grad = self.grad_blocks(v[:, 0])
        vy_grad = self.grad_blocks(v[:, 1])
        vz_grad = self.grad_blocks(v[:, 2])
        v_grad = torch.cat([vx_grad, vy_grad, vz_grad], dim=1)
        # conv layers
        for i in range(self.num_blocks_3d):
            v = self.conv_blocks_wind[i](v)
            v_grad = self.conv_blocks_grad[i](v_grad)
        # combination of wind and its gradient
        out = self.final_conv_block(torch.cat([v, v_grad], dim=1).squeeze(-3))
        
        return out
        
        
# In [6]: Generator        


class Generator(nn.Module):

    def __init__(
        self,
        conditioning_stack: nn.Module,
        latent_stack: nn.Module,
        wind_stack: nn.Module,
        sampler: nn.Module,
        device: str | int | torch.device = None,
        dtype: type | torch.dtype = None,
    ):
        
        super().__init__()
        # device & dtype
        self.dtype = dtype if dtype else torch.float32
        self.device = torch.device(device) if device else torch.device(type='cpu')
        # stacks
        self.conditioning_stack = conditioning_stack.to(self.device, self.dtype)
        self.latent_stack = latent_stack.to(self.device, self.dtype)
        self.wind_stack = wind_stack.to(self.device, self.dtype)
        self.sampler = sampler.to(self.device, self.dtype)
        # model architectures
        self.cond_steps = sampler.conditioning_steps
        self.fore_steps = sampler.forecast_steps
        self.time_delta = sampler.time_delta
        
    #========================================================== Fluent Interface ==========================================================#
    def to(self, device: str | int | torch.device | type | torch.dtype = None, dtype: type | torch.dtype = None):
        # make it robust if only the arguement "dtype" is fed <-- {e.g., XXX.to(torch.float16)}
        device, dtype = (None, device) if isinstance(device, (type | torch.dtype)) else (device, dtype)  
        return Generator(self.conditioning_stack.to(device, dtype), self.latent_stack.to(device, dtype), self.wind_stack.to(device, dtype),
            self.sampler.to(device, dtype), device if device else self.device, dtype if dtype else self.dtype)
    # device conversion using method chaining
    def cpu(self):
        return Generator(self.conditioning_stack.to(device, dtype), self.latent_stack.to(device, dtype), self.wind_stack.to(device, dtype),
            self.sampler.to(device, dtype), torch.device(type='cpu'), self.dtype)
    def cuda(self, index: int = None):
        return Generator(self.conditioning_stack.to(device, dtype), self.latent_stack.to(device, dtype), self.wind_stack.to(device, dtype),
            self.sampler.to(device, dtype), torch.device(type='cuda', index=index), self.dtype)
    # dtype conversion using method chaining
    @classmethod
    def create_dtype_convert(cls, name, dtype):
        def func(self):
            return Generator(self.conditioning_stack.to(dtype=dtype), self.latent_stack.to(dtype=dtype), self.wind_stack.to(dtype=dtype),
                self.sampler.to(dtype=dtype), self.device, dtype)
        setattr(cls, name, func)
    #--------------------------------------------------------------------------------------------------------------------------------------# 
        
    def forward(self, x: torch.Tensor, dates=None, wind=None) -> torch.Tensor:
        
        forecasts = []
        if wind == None:
            hidden_states = self.conditioning_stack(x)  # retreive sampler initial states from conditioning stack
        else:      
            hidden_wind_seq = torch.stack([self.wind_stack(wind[:, t]) for t in range(self.cond_steps)], dim=1)
            hidden_states = self.conditioning_stack(torch.cat([x, hidden_wind_seq], dim=-3))
        
        if dates == None:
        
            for i in range(self.cond_steps, self.cond_steps+self.fore_steps):
                z = self.latent_stack(x)
                out, new_states = self.sampler(z, hidden_states)
                hidden_states = new_states  # update hidden states for sampler
                forecasts.append(out)
        
        else:
            #dates = [dt.datetime.fromtimestamp(date) for date in dates.tolist()] if isinstance(dates, torch.Tensor) else dates  # convert dates -> list if dates is torch.Tensor
            for i in range(self.cond_steps, self.cond_steps+self.fore_steps):
                # convert dates to current time step
                increment = torch.zeros(dates.size(0), 5).to(self.device, self.dtype)
                increment[:, -2] = self.time_delta * i
                #z = self.latent_stack(x, [date + dt.timedelta(minutes=self.time_delta*i) for date in dates])
                z = self.latent_stack(x, dates+increment)
                out, new_states = self.sampler(z, hidden_states)
                hidden_states = new_states  # update hidden states for sampler
                forecasts.append(out)
            
        return torch.cat(forecasts, dim=1)   
        
        
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
    Conditioning_stack.create_dtype_convert(name, dtype)        
    Latent_stack.create_dtype_convert(name, dtype) 
    Sampler.create_dtype_convert(name, dtype)
    Wind_stack.create_dtype_convert(name, dtype)
    Generator.create_dtype_convert(name, dtype) 
        
        
        