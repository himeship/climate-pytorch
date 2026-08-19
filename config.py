#!/usr/bin/env python
# coding: utf-8

# In[0]: Imports


import torch
import calendar


# In [1] : DGMR Configuration


class configuration(object):

    def __init__(
            self,
            # layer hyperparameters
            sn_eps: float = 1e-4,
            spec_dtype: torch.dtype = None,
            # block hyperparameters
            d_block_kernel_size = 3,
            # model architecture hyperparameters
            time_embedding: bool = True,
            conditioning_steps: int = 4,
            forecast_steps: int = 18,
            in_channels: int = 1,
            out_channels: int = 1,
        ):
        
        self.sn_eps = 1e-4
        self.dtype = None


#    class layer(object):
#        def __init__(
#                self,
#                sn_eps: float = 1e-4,
#                spec_dtype: torch.dtype = None,
#            ):
#            self.sn_eps = sn_eps
#            self.spec_dtype = spec_dtype


