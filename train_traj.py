#!/usr/bin/env python
# coding: utf-8

# In [0]: Imports


import os
import time
import torch
import random
import argparse

import numpy as np
import datetime as dt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributed as dist

from utils import time2vec as t2v
from torchvision import transforms
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from RadarDataset import RadarRainrateDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from generator import Conditioning_stack, Latent_stack, Wind_stack, Sampler, Generator
from discriminator import Spatial_Discriminator, Temporal_Discriminator, Discriminator


job_start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
local_rank = int(os.environ["LOCAL_RANK"])

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
is_main_process = (dist.get_rank()==0)

# for H100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#torch.backends.cudnn.benchmark = True
#torch.backends.cuda.matmul.allow_bf16 = True   # supported in PyTorch 2.1+
#torch.backends.cudnn.allow_bf16 = True         # for convolution layers (cuDNN)

print(f"TF32 Matmul: {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")


# In [1]: Hyperparameters Setting


# Adjust batch size based on number of GPUs
BATCH_SIZE = 5
TRAJECTORY = 4
DISC_ITER = 2
NUM_EPOCHS = 1000
LAMBDA1 = 10
LAMBDA2 = 0.5

LEARNING_RATE_D = 5e-4        # 1e-5
LEARNING_RATE_G = 2e-4        # 2e-5

# Keep using BFloat16
dtype = torch.float32
TRANSFORM = transforms.Lambda(lambda x: torch.clamp(x, min=0, max=400))
TRANSFORM = None


# In [2]: Dataset & Model Loading


cond_steps = 6
fore_steps = 18

# Convert models to device with BFloat16
conditioning_stack = Conditioning_stack(in_channels=12, conditioning_steps=cond_steps).to(device, dtype)
latent_stack = Latent_stack(sampling_scale=2.0).to(device, dtype)
wind_stack = Wind_stack().to(device, dtype)
sampler = Sampler(conditioning_stack, latent_stack).to(device, dtype)
gen = Generator(conditioning_stack=conditioning_stack, latent_stack=latent_stack, wind_stack=wind_stack, sampler=sampler).to(device, dtype)
print('Gen: {}'.format(gen.dtype), flush=True)

spatial_disc = Spatial_Discriminator(in_channels=2, conditioning_steps=cond_steps, num_timesteps=12)
temporal_disc = Temporal_Discriminator(in_channels=2, conditioning_steps=cond_steps, num_timesteps=16)
disc = Discriminator(spatial_disc, temporal_disc).to(device, dtype)
print('Disc: {}'.format(disc.dtype), flush=True)

Load_path = '/home/users/astar/ares/deshp/scratch/climate-pytorch/traj_model_state_dict'
Save_path = '/home/users/astar/ares/deshp/scratch/climate-pytorch/traj_model_state_dict'


# In [3]: Parallellism


# define dataset for training
dataset = RadarRainrateDataset(root_dir='/home/users/astar/ares/deshp/scratch/climate-pytorch/data', transform=TRANSFORM)

# ensure differemt
#shared_seed = 42
#torch.manual_seed(shared_seed)
#np.random.seed(shared_seed)
#random.seed(shared_seed)
#g = torch.Generator().manual_seed(shared_seed)

SAMPLER = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True,
)

# Initialize DDP with gradient synchronization and static graph
#gen  = torch.compile(gen,  backend="inductor", mode="default")
#disc = torch.compile(disc, backend="inductor", mode="default")

disc = DDP(disc, 
           device_ids=[local_rank], 
           find_unused_parameters=True, 
           broadcast_buffers=False,
           #static_graph=True,
           gradient_as_bucket_view=True)  # Enable gradient as bucket view for better memory efficiency
gen = DDP(gen, 
          device_ids=[local_rank], 
          find_unused_parameters=True,
          broadcast_buffers=False,
          #static_graph=True,
          gradient_as_bucket_view=True)  # Enable gradient as bucket view for better memory efficiency

# set up optimization with gradient clipping
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.0, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(0.0, 0.999))

# enable autograd anomaly detection
torch.autograd.set_detect_anomaly(True)

# list to contain losses
loss_d_list = []
loss_g_list = []

# load trained model and the original loss to continue training
#loss_d_list = np.load('{}/lossD_lrD{:.0e}_lrG{:.0e}_is.npy'.format(Load_path, LEARNING_RATE_D, LEARNING_RATE_G)).tolist()
#loss_g_list = np.load('{}/lossG_lrD{:.0e}_lrG{:.0e}_is.npy'.format(Load_path, LEARNING_RATE_D, LEARNING_RATE_G)).tolist()
#disc.load_state_dict(torch.load('{}/disc_state_dict_lrD{:.0e}_lrG{:.0e}_is.pth'.format(Load_path, LEARNING_RATE_D, LEARNING_RATE_G)))
#gen.load_state_dict(torch.load('{}/gen_state_dict_lrD{:.0e}_lrG{:.0e}_is.pth'.format(Load_path, LEARNING_RATE_D, LEARNING_RATE_G))) 
#print('Trained weights and history have been loaded', flush=True)


# In [p]: Custom Loader 


def custom_collate_fn(batch):
    valid_samples = []
    valid_dates = []

    for sample, dates in batch:
        try:
            # Ensure the samples meet size requirements
            if sample['wind'].shape[0] != 6 or sample['prcp'].shape[0] != 24:
                raise ValueError("Inconsistent sequence length")
            valid_samples.append(sample)
            valid_dates.append(dates)
        except ValueError as e:
            print(f"Skipping invalid sample due to: {e}", flush=True)
            continue

    # If no valid samples, return empty to avoid issues
    if not valid_samples:
        return None, None
        
    samples_wind = torch.stack([s['wind'] for s in valid_samples])
    samples_prcp = torch.stack([s['prcp'] for s in valid_samples])
    samples_refl = torch.stack([s['refl'] for s in valid_samples])
    
    samples_wind = samples_wind * (samples_wind<=999)
    samples_prcp = samples_prcp * (samples_prcp<=999)
    samples_refl = samples_refl * (samples_refl>=-30)
    
    samples_refl = torch.cat([samples_refl, samples_refl.max(dim=2, keepdim=True).values], dim=2)
    samples_video = torch.cat([samples_prcp, samples_refl], dim=2)
    
    # convert dates tensor
    samples_dates = torch.Tensor([list(map(int, [t[4:6], t[6:8], t[8:10], t[10:12], t[12:14]])) for t in valid_dates])
    
    # Return valid batches
    return  (
        samples_wind.repeat_interleave(TRAJECTORY, dim=0), 
        samples_prcp.repeat_interleave(TRAJECTORY, dim=0), 
        samples_video.repeat_interleave(TRAJECTORY, dim=0), 
        samples_dates.repeat_interleave(TRAJECTORY, dim=0),
    )


# In [4]: Traininig


amp = 1
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    
    # Reset shuffle seed per epoch
    #g.manual_seed(shared_seed + epoch)
    
    SAMPLER.set_epoch(epoch)
      
    # Update the DataLoader
    loader = DataLoader(
        dataset,
        sampler=SAMPLER,
        batch_size=BATCH_SIZE,
        #shuffle=True,
        #generator=g,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )
    
    for batch_idx, (samples_wind, samples_prcp, samples_video, samples_dates) in enumerate(loader):

        # Optional: make per-GPU noise different
        # torch.manual_seed(local_rank + epoch * 1000 + batch_idx)
        
        # load physical information
        wind, dates_tensor, prcp, video = (
            samples_wind.to(dtype=dtype, device=device, non_blocking=True), 
            samples_dates.to(dtype=dtype, device=device, non_blocking=True),
            samples_prcp[:, :(cond_steps+fore_steps)].to(dtype=dtype, device=device, non_blocking=True), 
            samples_video[:, :(cond_steps+fore_steps)].to(dtype=dtype, device=device, non_blocking=True),
        )
        context_wind, context, real, context_real = wind[:, :cond_steps], video[:, :cond_steps], prcp[:, cond_steps:], prcp
        cur_batch_size = video.shape[0]
        
        # multiple-trajectory
        #real = real.repeat_interleave(TRAJECTORY, dim=0)
        #context = context.repeat_interleave(TRAJECTORY, dim=0)
        #dates_tensor = dates_tensor.repeat_interleave(TRAJECTORY, dim=0)
        #context_wind = context_wind.repeat_interleave(TRAJECTORY, dim=0)
        
        # ensemble nowcasting
        fake = gen(context, dates=dates_tensor, wind=context_wind)
        context_fake = torch.cat([context[:, :, 0:1], fake], dim=1)
        
        # train Discriminator:
        for _ in range(DISC_ITER):
            with autocast(dtype=dtype):  # Use BFloat16 for A100
                # Create detached copies of the tensors
                context_real_detached = context_real.detach()
                context_fake_detached = context_fake.detach()
                # Stack inputs, run disc once
                combined_input = torch.cat([context_real_detached, context_fake_detached], dim=0)
                combined_dates = dates_tensor.repeat(2, 1) 
                # Forward the stack inputs
                combined_output = disc(combined_input, combined_dates)
                real_output, fake_output = combined_output.chunk(2)
                # Disc loss calculation
                loss_disc = amp * torch.mean(1 - real_output + fake_output)
            
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            # Add synchronization after discriminator update
            try:
                dist.barrier()  # Use NCCL's built-in synchronization
            except Exception as e:
                print(f"Error during discriminator synchronization on rank {local_rank}: {e}", flush=True)
                raise

        # train Generator
        with autocast(dtype=dtype):  # Use BFloat16 for A100; Float32 for H100
            #regularizer1 = torch.mean(20*torch.log1p(torch.abs(fake - real))*torch.clip(torch.log(real+2), 0, 4)) + torch.mean(torch.abs(fake - real)*torch.clip(real+1, 0, 128))
            #regularizer2 = torch.mean(20*torch.log1p(torch.abs(fake - real))*torch.clip(torch.log(fake+5), 0, 4)) + torch.mean(torch.abs(fake - real)*torch.clip(fake+1, 0, 128))
            regularizer1 = torch.mean(torch.abs(fake - real)*torch.clip(real+1, 0, 128))
            regularizer2 = torch.mean(torch.abs(fake - real)*torch.clip(fake+1, 0, 128))
            loss_gen = amp * (4*torch.mean(1-disc(context_fake, dates_tensor.to(dtype))) + regularizer1*LAMBDA1 + regularizer2*LAMBDA2)
        
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        # Add synchronization after generator update
        try:
            dist.barrier()  # Use NCCL's built-in synchronization
        except Exception as e:
            print(f"Error during generator synchronization on rank {local_rank}: {e}", flush=True)
            raise
        
        if is_main_process:
            # Print losses occasionally and print to tensorboard
            if ((batch_idx+1) % 1 == 0) & (batch_idx >= 0):
                print(f"Epoch [{(epoch+1)}/{NUM_EPOCHS}] Batch {batch_idx+1}/{len(loader)}      loss D: {loss_disc/amp:.5f}, loss G: {loss_gen/amp:.4f}      Elapsed Time: {(time.time() - start_time):.2f}", flush=True)
                start_time = time.time()
            
            if (batch_idx) % 1000 == 0:
                # save model state
                torch.save(disc.state_dict(), '{}/disc_state_dict_lrD{:.0e}_lrG{:.0e}_is_epoch{}_iter{}.pth'.format(Save_path, LEARNING_RATE_D, LEARNING_RATE_G, epoch, batch_idx))
                torch.save(gen.state_dict(), '{}/gen_state_dict_lrD{:.0e}_lrG{:.0e}_is_epoch{}_iter{}.pth'.format(Save_path, LEARNING_RATE_D, LEARNING_RATE_G, epoch, batch_idx))
            
            # saving files
            if ((batch_idx+1) % 1 == 0) & (batch_idx >= 0):
            
                # save training loss history
                loss_d_cpu = loss_disc.cpu().float().detach().numpy().reshape(1)[0]
                loss_g_cpu = loss_gen.cpu().float().detach().numpy().reshape(1)[0]
                loss_d_list.append(loss_d_cpu)
                loss_g_list.append(loss_g_cpu)
                LOSS_D = np.array(loss_d_list)
                LOSS_G = np.array(loss_g_list)
                
            # saving files
            if ((batch_idx+1) % 25 == 0) & (batch_idx >= 0):
            
                # save losses
                np.save('{}/lossD_lrD{:.0e}_lrG{:.0e}_is.npy'.format(Save_path, LEARNING_RATE_D, LEARNING_RATE_G), LOSS_D/amp)
                np.save('{}/lossG_lrD{:.0e}_lrG{:.0e}_is.npy'.format(Save_path, LEARNING_RATE_D, LEARNING_RATE_G), LOSS_G/amp)
        
                # save model state
                torch.save(disc.state_dict(), '{}/disc_state_dict_lrD{:.0e}_lrG{:.0e}_is.pth'.format(Save_path, LEARNING_RATE_D, LEARNING_RATE_G))
                torch.save(gen.state_dict(), '{}/gen_state_dict_lrD{:.0e}_lrG{:.0e}_is.pth'.format(Save_path, LEARNING_RATE_D, LEARNING_RATE_G))

torch.cuda.empty_cache()


# In [5]: Time Elapsed


if is_main_process:
    elapsed_time = time.time() - job_start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training Finished! Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}", flush=True)

dist.destroy_process_group()


