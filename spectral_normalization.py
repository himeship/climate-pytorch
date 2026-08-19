#!/usr/bin/env python
# coding: utf-8

# In[0]: Imports


import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

sn_eps = 0  # spectral_normalized_eps=0.0001 (default)


# In[1]: sn.linear 


class linear(nn.Module):

    def __init__(self, in_features, out_features, eps=sn_eps, bias=True, device=None, dtype=None):
        super().__init__()
        # Create a dictionary of the arguments
        args = locals()
        args.pop('self')
        # Use setattr to set each attribute
        for key, value in args.items():
            setattr(self, key, value)
        # Create layer
        self.layer = spectral_norm(
            nn.Linear(
                in_features=self.in_features,
                out_features=self.out_features,
                bias=self.bias,
                device=self.device,
                dtype=self.dtype,
            ),
            eps=self.eps,
        )
        
    def forward(self, x) -> torch.Tensor:
        return self.layer(x)
        
 
 # In[2]: sn.conv2d
 
        
class conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, eps=sn_eps, stride=1, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True, device=None, dtype=None):
        super().__init__()
        # Create a dictionary of the arguments
        args = locals()
        args.pop('self')
        # Use setattr to set each attribute
        for key, value in args.items():
            setattr(self, key, value)
        # Create layer
        self.layer = spectral_norm(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                padding_mode=self.padding_mode,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
                device=self.device,
                dtype=self.dtype,
            ),
            eps=self.eps,
        )
        
    def forward(self, x) -> torch.Tensor:        
        return self.layer(x)
        
        
 # In [3]: sn.conv3d
 
 
class conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, eps=sn_eps, stride=1, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True, device=None, dtype=None):
        super().__init__()
        # Create a dictionary of the arguments
        args = locals()
        args.pop('self')
        # Use setattr to set each attribute
        for key, value in args.items():
            setattr(self, key, value)
        # Create layer      
        self.layer = spectral_norm(
            nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                padding_mode=self.padding_mode,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
                device=self.device,
                dtype=self.dtype,                
            ),
            eps=self.eps,
        )
        
    def forward(self, x) -> torch.Tensor:
        return self.layer(x)

        
 # In [4]: sn.upsample (here is for using nn.UpSample for different dtypes, 'sn' stands for Special Networks)
 
 
class upsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, dtype=None):
        super().__init__()
        # Create a dictionary of the arguments
        args = locals()
        args.pop('self')
        # Use setattr to set each attribute
        for key, value in args.items():
            setattr(self, key, value)
        # default dtype is torch.float32
        self.dtype = self.dtype if self.dtype else torch.float32
        self.layer = nn.Upsample(
            size=self.size, 
            scale_factor=self.scale_factor,
            mode=self.mode, 
            align_corners=self.align_corners, 
            recompute_scale_factor=self.recompute_scale_factor,
        )
        
    def forward(self, x) -> torch.Tensor:
        return self.layer(x.to(torch.float32)).to(self.dtype)
        
        
 # In [5]: sn.avgpool2d (here is for using nn.AvgPool2d for different dtypes, 'sn' stands for Special Networks)
 
 
class avgpool2d(nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, dtype=None):
        super().__init__()
        # Create a dictionary of the arguments
        args = locals()
        args.pop('self')
        # Use setattr to set each attribute
        for key, value in args.items():
            setattr(self, key, value)
        # default dtype is torch.float32
        self.dtype = self.dtype if self.dtype else torch.float32
        self.layer = nn.AvgPool2d(
            kernel_size=self.kernel_size, 
            stride=self.stride,
            padding=self.padding, 
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )
        
    def forward(self, x) -> torch.Tensor:
        return self.layer(x.to(torch.float32)).to(self.dtype)    
        
        
 # In [6]: sn.avgpool3d (here is for using nn.AvgPool3d for different dtypes, 'sn' stands for Special Networks)
 
 
class avgpool3d(nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, dtype=None):
        super().__init__()
        # Create a dictionary of the arguments
        args = locals()
        args.pop('self')
        # Use setattr to set each attribute
        for key, value in args.items():
            setattr(self, key, value)
        # default dtype is torch.float32
        self.dtype = self.dtype if self.dtype else torch.float32
        self.layer = nn.AvgPool3d(
            kernel_size=self.kernel_size, 
            stride=self.stride,
            padding=self.padding, 
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )
                
    def forward(self, x) -> torch.Tensor:
        return self.layer(x.to(torch.float32)).to(self.dtype)     
        
        
        
        
        
        
        
        
        