#!/usr/bin/env python
# coding: utf-8

# In[0]: Imports


import torch
import calendar

import torch.nn as nn
import datetime as dt
import torch.nn.functional as F


# In [1] : Time 2 Vector


def time2vec(
        dates: list,
        embedding_type: str = 'mdH', 
        discrete_clocktime: bool = False, 
        normalization: bool = True,
        device: str | int | torch.device = None,
        dtype: type | torch.Tensor = None,
    ) -> torch.Tensor:
       
    X = torch.arange(0).view(0, len(embedding_type)).to(device, dtype)
        
    for date in dates:
        
        Y = date.year
        m = date.month
        d = date.day
        H = date.hour
        M = date.minute
        S = date.second
        
        num_days_in_Y = 365 + calendar.isleap(Y)*1
        _, num_days_in_m = calendar.monthrange(Y, m)
        time_dict = {'Y': None, 'm': None, 'd': None, 'H': None, 'M': None, 'S': None}

        # normalized date embedding with the range of [0, 1]
        if 'Y' in embedding_type:
            time_dict['Y'] = (Y-2016) / (2022-2016) 
            if ~normalization:
                time_dict['Y'] = Y

        if 'm' in embedding_type:
            time_dict['m'] = m / (12*normalization + (not normalization))

        if 'd' in embedding_type:
            if 'm' in embedding_type:
                time_dict['d'] = d / (num_days_in_m*normalization + (not normalization))
            else:
                time_dict['d'] = date.timetuple().tm_yday / (num_days_in_Y*normalization + (not normalization))

        if 'H' in embedding_type:
            time_dict['H'] = (H + M/60 + S/3600) / (24*normalization + (not normalization))
            if (discrete_clocktime) or ('M' in embedding_type) or ('S' in embedding_type):
                time_dict['H'] = H / (24*normalization + (not normalization))

        if 'M' in embedding_type:
            time_dict['M'] = (M + S/60) / (60*normalization + (not normalization))
            if (discrete_clocktime) or ('S' in embedding_type):
                time_dict['M'] = M / (60*normalization + (not normalization))

        if 'S' in embedding_type:
            time_dict['S'] = S / (60*normalization + (not normalization)) 

        x = torch.tensor([time_dict[label] for label in embedding_type]).view(1, -1).to(device, dtype)
        X = torch.cat([X, x], dim=0).to(device, dtype)
    
    return X   


# In [2] : Gradient Penalty    
    

def gradient_penalty(disc, real, fake, device='cpu'):

		BATCH_SIZE, T, C, H, W = real.shape
		epsilon = torch.rand((BATCH_SIZE, 1, 1, 1, 1)).repeat(1, T, C, H, W).to(device)
		interpolated_videos = real * epsilon + fake * (1 - epsilon)
		
		# calculate disc scores
		mixed_scores = disc(interpolated_videos)
		
		gradient = torch.autograd.grad(
				inputs=interpolated_videos,
				outputs=mixed_scores,
				grad_outputs=torch.ones_like(mixed_scores),
				create_graph=True,
				retain_graph=True,
		)[0]
		
		gradient = gradient.view(gradient.shape[0], -1)
		gradient_norm = gradient.norm(2, dim=1)
		gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
		
		return gradient_penalty




