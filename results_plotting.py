#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cmaps
import torch
import socket
import operator
import glob
import os

import numpy as np
import xarray as xr
import netCDF4 as nc
import datetime as dt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.feature as cfeature

from scores import categorical
from scores.categorical import BasicContingencyManager

hostname = socket.gethostname()

plt.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams['font.size'] = 15


# In[2]:


# LEARNING_RATE_D = 5e-4
# LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
LEARNING_RATE_G = 1e-4
loss_D_0 = np.load('WIND_model_state_dict_new/lossD_lrD{:.0e}_lrG{:.0e}_is.npy'.format(LEARNING_RATE_D, LEARNING_RATE_G))
loss_G_0 = np.load('WIND_model_state_dict_new/lossG_lrD{:.0e}_lrG{:.0e}_is.npy'.format(LEARNING_RATE_D, LEARNING_RATE_G))

loss_D = np.concatenate([loss_D_0])
loss_G = np.concatenate([loss_G_0])

# model_D = LinearRegression()
# model_D.fit(np.arange(loss_D.shape[0]).reshape(-1, 1), loss_D.reshape(-1, 1))
# y_pred_D = model_D.predict(loss_D.reshape(-1, 1))
# model_G = LinearRegression()
# model_G.fit(np.arange(loss_G.shape[0]).reshape(-1, 1), loss_G.reshape(-1, 1))
# y_pred_G = model_G.predict(loss_G.reshape(-1, 1))

fig, axes = plt.subplots(2, 1, figsize=(20, 12))

axes[0].plot(loss_D[:], alpha=0.6, linewidth=0.5)
#axes[0].plot(y_pred_D, linestyle='--', linewidth=1)
axes[0].set_title('Discriminator Loss', fontsize=25)
axes[1].plot(loss_G[:], alpha=0.6, linewidth=0.5)
#axes[1].plot(y_pred_G, linestyle='--')
axes[1].set_title('Generator Loss', fontsize=25)

#plt.errorbar(range(1, len(means) + 1), means, yerr=stds, fmt='o', capsize=5, label='Mean ± 1 Std Dev')
#plt.plot(range(5000, len(means)*10000+5000, 10000), means, color='red', linestyle='-', marker='o', label='Mean Value')
plt.title('Generator Loss', fontsize=25)
#plt.ylim([230, 340])
#plt.xlabel('chunk of iterations')

# for i, mean in enumerate(means, start=1):
#     plt.text(i, mean + 0.0025 * max(means), f"{mean:.1f}", ha='center', va='bottom', fontsize=12, color='black')


# In[3]:


print(loss_D.min(), loss_D.mean(), loss_D.max())
print(loss_G.min(), loss_G.mean(), loss_G.max())
print('iterations: {}'.format(len(loss_D)))
min(loss_G), min(loss_D)


# In[4]:


# Split the array into chunks of 10,000 elements
chunk_size = 10000
chunks = [loss_G[:][i:i+chunk_size] for i in range(0, len(loss_G), chunk_size)]

# Calculate mean values for each chunk
stds = [chunk.std() for chunk in chunks]
means = [chunk.mean() for chunk in chunks]

# Plot the line connecting the mean values
plt.figure(figsize=(12, 3))
#plt.errorbar(range(1, len(means) + 1), means, yerr=stds, fmt='o', capsize=5, label='Mean ± 1 Std Dev')
plt.plot(range(1, len(means) + 1), means, color='red', linestyle='-', marker='o', label='Mean Value')
plt.title('Average Generator Loss for every 20,000 iterations', fontsize=16)
#plt.ylim([280, 420])
plt.xlabel('chunk of iterations')

for i, mean in enumerate(means, start=1):
    plt.text(i, mean + 0.0025 * max(means), f"{mean:.2f}", ha='center', va='bottom', fontsize=12, color='black')

plt.figure(figsize=(15, 6))
plt.boxplot(chunks, showfliers=False);


# In[5]:


import time
import torch

import numpy as np
import netCDF4 as nc
import datetime as dt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import time2vec as t2v
from torchvision import transforms
from torch.utils.data import DataLoader
from RadarDataset import RadarRainrateDataset
from generator import Conditioning_stack, Latent_stack, Wind_stack, Sampler, Generator
from discriminator import Spatial_Discriminator, Temporal_Discriminator, Discriminator


# In[6]:


# define the default font as 'Myriad Pro'
#plt.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams['font.size'] = 18

# loading presenting dataset
prcp_nc = nc.Dataset('../SG_data/SG_prcp_202101.nc')
# reading variables
lon = prcp_nc.variables['lon'][:]
lat = prcp_nc.variables['lat'][:]
prcp = prcp_nc.variables['prcp'][:, :]
time_var = prcp_nc.variables['time']
date = nc.num2date(time_var, units=time_var.units, calendar=time_var.calendar)


# In[9]:


BATCH_SIZE = 24
DISC_ITERATIONS = 2
ENSEMBLE_MEMBERS = 2
NUM_EPOCHS = 20
LAMBDA = 10
LAMBDA_GP = 20

gen_dtype = torch.bfloat16
disc_dtype = torch.bfloat16
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
TRANSFORM = transforms.Lambda(lambda x: torch.clamp(x, min=0, max=400))
# TRANSFORM = None

conditioning_stack = Conditioning_stack(in_channels=5)
latent_stack = Latent_stack()
wind_stack = Wind_stack()
sampler = Sampler(conditioning_stack, latent_stack)
gen = Generator(conditioning_stack=conditioning_stack, latent_stack=latent_stack, wind_stack=wind_stack, sampler=sampler).to(device, gen_dtype)

spatial_disc = Spatial_Discriminator(in_channels=2)
temporal_disc = Temporal_Discriminator(in_channels=2)
disc = Discriminator(spatial_disc, temporal_disc).to(device, disc_dtype)

# define dataset for training
dataset = RadarRainrateDataset(root_dir='../SG_data/ccrs_project/train_new_wind/', transform=TRANSFORM)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# set up GPU parallelism
# disc = nn.DataParallel(disc, device_ids=device_ids).to(device)
# gen = nn.DataParallel(gen, device_ids=device_ids).to(device)
# set up optimization
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.0, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(0.0, 0.999))


# load trained model
with torch.no_grad():
#     disc.load_state_dict(torch.load('WIND_model_state_dict/disc_state_dict_lrD{:.0e}_lrG{:.0e}_is.pth'
#                            .format(LEARNING_RATE_D, LEARNING_RATE_G, 9, 1950)))
    gen.load_state_dict(torch.load('WIND_model_state_dict_new/gen_state_dict_lrD{:.0e}_lrG{:.0e}_is.pth'
                           .format(LEARNING_RATE_D, LEARNING_RATE_G, 9, 1950))) 


# In[ ]:

dtype = gen_dtype

input_path = '~/scratch/climate-pytorch'
input_file_pattern = os.path.join(input_path, "*.pt")
pt_files = glob.glob(input_file_pattern)
real = [torch.load(f) for f in pt_files]
# real = torch.load('~/scratch/climate-pytorch/20210102013202_20210102013000.pt')
# real = torch.load('~/scratch/climate-pytorch/20221030143224_20221030143000.pt')
# real = torch.load('~/scratch/climate-pytorch/20210101172203_20210101172000.pt')
# real = torch.load('~/scratch/climate-pytorch/20220223163223_20220223163000.pt')
# real = torch.load('~/scratch/climate-pytorch/20210901203203_20210901203000.pt')
# real = torch.load('~/scratch/climate-pytorch/20210901174202_20210901174000.pt')
# real = torch.load('~/scratch/climate-pytorch/20210901215204_20210901215000.pt')
# real = torch.load('~/scratch/climate-pytorch/20210102035703_20210102035500.pt')

# 20210710024204_20210710024000.pt
# 20221009205724_20221009205500.pt
# 20220216005223_20220216005000.pt

real_prcp = real['prcp'].unsqueeze(0).to(device, dtype)
real_refl = real['refl'].unsqueeze(0).to(device, dtype)
real_wind = real['wind'].flip(-2).unsqueeze(0).to(device, dtype)
real_video = torch.cat([real_prcp, real_refl], dim=2)
refl_video = torch.cat([real_prcp*0, real_refl], dim=2)

fake_refl = torch.rand([1, 24, 1, 256, 256]).to(device, dtype) * real_refl.mean() / 2
fake_wind = torch.rand([1, 6, 3, 13, 128, 128]).to(device, dtype)
fake_video = torch.cat([real_prcp, fake_refl], dim=2)
dates_right = torch.Tensor([1, 2, 1, 32, 2]).unsqueeze(0).to(device).bfloat16()
dates_right = torch.Tensor([10, 30, 14, 32, 24]).unsqueeze(0).to(device).bfloat16()
# dates_right = torch.Tensor([9, 1, 17, 42, 2]).unsqueeze(0).cuda().bfloat16()
# dates_right = torch.Tensor([2, 23, 16, 32, 23]).unsqueeze(0).cuda().bfloat16()

dates_tensor = torch.cat([dates_right, dates_right], dim=0)

real_prcp.mean()


# 

# In[ ]:


real_wind.shape, fake_wind.shape


# # Case study

# In[ ]:


ENSEMBLE_MEMBERS = 8
forecast = torch.zeros_like(real_prcp[:, 4:22]).float().cpu().detach().numpy()
forecast_wo = torch.zeros_like(real_prcp[:, 4:22]).float().cpu().detach().numpy()
forecast_wo_dates = torch.zeros_like(real_prcp[:, 4:22]).float().cpu().detach().numpy()
forecast_wo_wind = torch.zeros_like(real_prcp[:, 4:22]).float().cpu().detach().numpy()


# test_date = dates_tensor[:1].float().repeat(ENSEMBLE_MEMBERS)
# test_prcp = real_prcp[:1, :4].float().repeat(ENSEMBLE_MEMBERS, 1, 1, 1, 1)
# test_wind = real_wind[:1, :4].float().repeat(ENSEMBLE_MEMBERS, 1, 1, 1, 1, 1)

# forecast = gen(test_prcp, dates=test_date, wind=test_wind).mean(dim=0)

for i in range(ENSEMBLE_MEMBERS):
    forecast += \
        gen(real_video[:, :4].to(gen_dtype), dates=dates_tensor[:1].to(gen_dtype), wind=real_wind[:, :4]).float().cpu().detach().numpy()
forecast /= ENSEMBLE_MEMBERS

# for i in range(ENSEMBLE_MEMBERS):
#     fake_wind = torch.normal(mean=real_wind.mean(), std=real_wind.std(), size=real_wind.shape)
#     forecast_wo += \
#         gen(refl_video[:, :4].to(gen_dtype), dates=dates_tensor[:1].to(gen_dtype), wind=real_wind[:, :4]).float().cpu().detach().numpy()
# forecast_wo /= ENSEMBLE_MEMBERS

for i in range(ENSEMBLE_MEMBERS):
    fake_wind = torch.normal(mean=real_wind.mean(), std=real_wind.std(), size=real_wind.shape)
    forecast_wo += \
        gen(fake_video[:, :4].to(gen_dtype), wind=fake_wind[:, :4].to(gen_dtype)).float().cpu().detach().numpy()
forecast_wo /= ENSEMBLE_MEMBERS

for i in range(ENSEMBLE_MEMBERS):
    fake_wind = torch.normal(mean=real_wind.mean(), std=real_wind.std(), size=real_wind.shape)
    forecast_wo_dates += \
        gen(real_video[:, :4].to(gen_dtype), wind=real_wind[:, :4].to(gen_dtype)).float().cpu().detach().numpy()
forecast_wo_dates /= ENSEMBLE_MEMBERS

for i in range(ENSEMBLE_MEMBERS):
    fake_wind = torch.normal(mean=real_wind.mean(), std=real_wind.std(), size=real_wind.shape)
    forecast_wo_wind += \
        gen(real_video[:, :4].to(gen_dtype), dates=dates_tensor[:1].to(gen_dtype), wind=fake_wind[:, :4].to(gen_dtype)).float().cpu().detach().numpy()
forecast_wo_wind /= ENSEMBLE_MEMBERS

real_cpu = real_prcp.float().cpu().detach().numpy()
refl_cpu = real_refl.float().cpu().detach().numpy()


# In[ ]:


plt.figure(0)
plt.pcolor(fake_wind[0, 0, 2, 10].float())
plt.colorbar()
plt.figure(1)
plt.pcolor(real_wind[0, 1, 0, 10].float().cpu())
plt.colorbar()


# In[ ]:


real_wind.min()


# In[ ]:


fake_wind.min()


# In[ ]:


real_cpu.shape


# In[ ]:


forecast.max(), real_cpu.max()


# In[ ]:


forecast.mean()


# In[ ]:


levels_ccrs = [-99, 0.1, 0.2, 0.5, 1, 2, 3,
          4, 5, 6, 7, 8, 9, 10,
          15, 20, 25, 30, 35, 40, 45,
          50, 60, 70, 80, 90, 100, 125,
          150, 200, 250, 300]

#D6DBDF
cmap_ccrs = colors.ListedColormap(
    ['#FFFFFF', '#E1EFFF', '#AFECFF', '#37CFFF', '#0168D9', '#014289', '#1066A0',
     '#0D7A89', '#0E8C65', '#109C45', '#50A913', '#16D816', '#93D816', '#D7DC16',
     '#F2F71D', '#D7CD07', '#E9B909', '#D68E0C', '#D6670C', '#FF8001', '#F41902',
     '#C21402', '#901002', '#740D02', '#560A02', '#420042', '#5C005C', '#760076',
     '#960096', '#B800B8', '#EE00EE', '#FF57FF'])
norm_ccrs = colors.BoundaryNorm(levels_ccrs, ncolors=cmap_ccrs.N, clip=True)
counter = 0


# gen

# In[ ]:


n = 0
cb_max_v = 75

# Extracting the components
year = 2021  # Fixed year
month = int(dates_tensor[0, 0].item())
day = int(dates_tensor[0, 1].item())
hour = int(dates_tensor[0, 2].item())
minute = int(dates_tensor[0, 3].item())
second = int(dates_tensor[0, 4].item())

# Create datetime object
date = dt.datetime(year, month, day, hour, minute, second)

for t in range(4):

    fig = plt.figure(t+1, figsize = (24, 6))
    
    
    ax = plt.subplot(141, projection = ccrs.PlateCarree())
    mm = ax.pcolormesh(lon, lat, real_cpu[n, t, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)

    _ = ax.set_title('Context (past {:02d} min)'.format(15-5*t))
    _ = ax.coastlines(alpha = 0.5, linestyle = '-')
    _ = ax.gridlines(draw_labels = ['left', 'bottom'], linestyle = ':')

    _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
    _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
    _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)
    
    

    ax = plt.subplot(142, projection = ccrs.PlateCarree())
    mm = ax.pcolormesh(lon, lat, 0*refl_cpu[n, t, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)

    _ = ax.set_title('Ground Truth (next 90 min)')
    _ = ax.coastlines(alpha = 0.5, linestyle = '-')
    _ = ax.gridlines(draw_labels = ['bottom'], linestyle = ':')

    _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
    _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
    _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)

    
    
    
    ax = plt.subplot(143, projection = ccrs.PlateCarree())
    mm = ax.pcolormesh(lon, lat, 0*forecast[n, t, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)

    _ = ax.set_title('Model With Embedding (next 00 min)')
    _ = ax.coastlines(alpha = 0.5, linestyle = '-')
    _ = ax.gridlines(draw_labels = ['bottom'], linestyle = ':')

    _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
    _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
    _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)
    
    
    
    
    ax = plt.subplot(144, projection = ccrs.PlateCarree())
    mm = ax.pcolormesh(lon, lat, 0*forecast_wo_dates[n, t, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)

    _ = ax.set_title('Model W/O Embedding (next 00 min)')
    _ = ax.coastlines(alpha = 0.5, linestyle = '-')
    _ = ax.gridlines(draw_labels = ['bottom'], linestyle = ':')

    _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
    _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
    _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)

    
    fig.suptitle('{}'.format(date+dt.timedelta(minutes=5*(t))), y = 0.96, fontsize=25)
    
    # Calculate (height_of_image / width_of_image)
    im_ratio = forecast[n, 4, 0, :, :].shape[0] / forecast[n, 4, 0, :, :].shape[1]
    
    cb_ax = fig.add_axes([0.925, 0.154, 0.01, 0.684])
    cb = fig.colorbar(mm, 
                      cax = cb_ax, 
                      label = 'Precip (mm/h)'
                     )
    #cb.set_ticks([0, cb_max_v//3, cb_max_v//3*2, cb_max_v])

    fig.savefig('figures/F_with_2_embed_SG_{:0>4d}'.format(t+1), bbox_inches='tight', dpi=300)


# n = 0
# cb_max_v = 75
# 
# # Extracting the components
# year = 2022  # Fixed year
# month = int(dates_tensor[0, 0].item())
# day = int(dates_tensor[0, 1].item())
# hour = int(dates_tensor[0, 2].item())
# minute = int(dates_tensor[0, 3].item())
# second = int(dates_tensor[0, 4].item())
# 
# # Create datetime object
# date = dt.datetime(year, month, day, hour, minute, second)
# 
# for t in range(18):
# 
#     fig = plt.figure(t+1, figsize = (24, 6))
#     
#     
#     ax = plt.subplot(141, projection = ccrs.PlateCarree())
#     mm = ax.pcolormesh(lon, lat, real_cpu[n, 3, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)
# 
#     _ = ax.set_title('Context (past 20 min)')
#     _ = ax.coastlines(alpha = 0.5, linestyle = '-')
#     _ = ax.gridlines(draw_labels = ['left', 'bottom'], linestyle = ':')
# 
#     _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
#     _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
#     _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
#     _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
#     _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)
#     
#     
# 
#     ax = plt.subplot(142, projection = ccrs.PlateCarree())
#     mm = ax.pcolormesh(lon, lat, real_cpu[n, t+4, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)
# 
#     _ = ax.set_title('Ground Truth (next {:02d} min)'.format(t*5+5))
#     _ = ax.coastlines(alpha = 0.5, linestyle = '-')
#     _ = ax.gridlines(draw_labels = ['bottom'], linestyle = ':')
# 
#     _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
#     _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
#     _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
#     _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
#     _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)
# 
#     
#     
#     
#     ax = plt.subplot(143, projection = ccrs.PlateCarree())
#     mm = ax.pcolormesh(lon, lat, forecast_wo_wind[n, t, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)
# 
#     _ = ax.set_title('Model With Time Regimes (next {:02d} min)'.format(t*5+5))
#     _ = ax.coastlines(alpha = 0.5, linestyle = '-')
#     _ = ax.gridlines(draw_labels = ['bottom'], linestyle = ':')
# 
#     _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
#     _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
#     _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
#     _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
#     _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)
#     
#     
#     
#     
#     ax = plt.subplot(144, projection = ccrs.PlateCarree())
#     mm = ax.pcolormesh(lon, lat, forecast_wo_dates[n, t, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)
# 
#     _ = ax.set_title('Model with Wind (next {:02d} min)'.format(t*5+5))
#     _ = ax.coastlines(alpha = 0.5, linestyle = '-')
#     _ = ax.gridlines(draw_labels = ['bottom'], linestyle = ':')
# 
#     _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
#     _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
#     _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
#     _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
#     _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)
# 
#     
#     fig.suptitle('{}'.format(date+dt.timedelta(minutes=5*(t+4))), y = 0.96, fontsize=25)
#     
#     # Calculate (height_of_image / width_of_image)
#     im_ratio = forecast[n, 4, 0, :, :].shape[0] / forecast[n, 4, 0, :, :].shape[1]
#     
#     cb_ax = fig.add_axes([0.925, 0.154, 0.01, 0.684])
#     cb = fig.colorbar(mm, 
#                       cax = cb_ax, 
#                       label = 'Precip (mm/h)'
#                      )
#     #cb.set_ticks([0, cb_max_v//3, cb_max_v//3*2, cb_max_v])
# 
#     #fig.savefig('figures_ccrs_20250124/F6_with_1_embed_{:0>4d}'.format(t+1+4), bbox_inches = 'tight', dpi = 300)

# In[ ]:


n = 0
cb_max_v = 75

# Extracting the components
year = 2022  # Fixed year
month = int(dates_tensor[0, 0].item())
day = int(dates_tensor[0, 1].item())
hour = int(dates_tensor[0, 2].item())
minute = int(dates_tensor[0, 3].item())
second = int(dates_tensor[0, 4].item())

# Create datetime object
date = dt.datetime(year, month, day, hour, minute, second)

for t in range(18):

    fig = plt.figure(t+1, figsize = (24, 6))
    
    
    ax = plt.subplot(141, projection = ccrs.PlateCarree())
    mm = ax.pcolormesh(lon, lat, real_cpu[n, 3, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)

    _ = ax.set_title('Context (past 20 min)')
    _ = ax.coastlines(alpha = 0.5, linestyle = '-')
    _ = ax.gridlines(draw_labels = ['left', 'bottom'], linestyle = ':')

    _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
    _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
    _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)
    
    

    ax = plt.subplot(142, projection = ccrs.PlateCarree())
    mm = ax.pcolormesh(lon, lat, real_cpu[n, t+4, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)

    _ = ax.set_title('Ground Truth (next {:02d} min)'.format(t*5+5))
    _ = ax.coastlines(alpha = 0.5, linestyle = '-')
    _ = ax.gridlines(draw_labels = ['bottom'], linestyle = ':')

    _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
    _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
    _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)

    
    
    
    ax = plt.subplot(143, projection = ccrs.PlateCarree())
    mm = ax.pcolormesh(lon, lat, forecast[n, t, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)

    _ = ax.set_title('Model With Embedding (next {:02d} min)'.format(t*5+5))
    _ = ax.coastlines(alpha = 0.5, linestyle = '-')
    _ = ax.gridlines(draw_labels = ['bottom'], linestyle = ':')

    _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
    _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
    _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)
    
    
    
    
    ax = plt.subplot(144, projection = ccrs.PlateCarree())
    mm = ax.pcolormesh(lon, lat, forecast_wo[n, t, 0, :, :], transform = ccrs.PlateCarree(), cmap = cmap_ccrs, norm=norm_ccrs)

    _ = ax.set_title('Model W/O Embedding (next {:02d} min)'.format(t*5+5))
    _ = ax.coastlines(alpha = 0.5, linestyle = '-')
    _ = ax.gridlines(draw_labels = ['bottom'], linestyle = ':')

    _ = ax.add_feature(cfeature.STATES, zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.RIVERS, zorder = 5, alpha = 0.75)
    _ = ax.add_feature(cfeature.LAKES, facecolor = 'aqua', zorder = 2, alpha = 0.25)
    _ = ax.add_feature(cfeature.OCEAN, facecolor = 'aqua', zorder = 2, alpha = 0.125)
    _ = ax.add_feature(cfeature.LAND, facecolor = 'silver', zorder = 2, alpha = 0.25)

    
    fig.suptitle('{}'.format(date+dt.timedelta(minutes=5*(t+4))), y = 0.96, fontsize=25)
    
    # Calculate (height_of_image / width_of_image)
    im_ratio = forecast[n, 4, 0, :, :].shape[0] / forecast[n, 4, 0, :, :].shape[1]
    
    cb_ax = fig.add_axes([0.925, 0.154, 0.01, 0.684])
    cb = fig.colorbar(mm, 
                      cax = cb_ax, 
                      label = 'Precip (mm/h)'
                     )
    #cb.set_ticks([0, cb_max_v//3, cb_max_v//3*2, cb_max_v])

    fig.savefig('figures/F_with_2_embed_SG_{:0>4d}'.format(t+1+4), bbox_inches='tight', dpi=300)


# In[ ]:





# In[ ]:


forecast_w = xr.DataArray(forecast[0, :, 0, :, :], 
                               coords=[[i for i in range(5, 95, 5)], [i for i in range(256)], [i for i in range(256)]],
                               dims=["lead_minute", "lat", "lon"])

forecast_wo_both = xr.DataArray(forecast_wo[0, :, 0, :, :], 
                               coords=[[i for i in range(5, 95, 5)], [i for i in range(256)], [i for i in range(256)]],
                               dims=["lead_minute", "lat", "lon"])
forecast_wo_d = xr.DataArray(forecast_wo_dates[0, :, 0, :, :], 
                               coords=[[i for i in range(5, 95, 5)], [i for i in range(256)], [i for i in range(256)]],
                               dims=["lead_minute", "lat", "lon"])
forecast_wo_w = xr.DataArray(forecast_wo_wind[0, :, 0, :, :], 
                               coords=[[i for i in range(5, 95, 5)], [i for i in range(256)], [i for i in range(256)]],
                               dims=["lead_minute", "lat", "lon"])

obs = xr.DataArray(real_cpu[0, 4:22, 0, :, :],
                            coords=[[i for i in range(5, 95, 5)], [i for i in range(256)], [i for i in range(256)]],
                            dims=["lead_minute", "lat", "lon"])


# In[ ]:





# In[ ]:


event_operator = categorical.ThresholdEventOperator(default_event_threshold=25, default_op_fn=operator.ge)
forecast_binary_w, observed_binary = event_operator.make_event_tables(forecast_w, obs)
forecast_binary_wo_both, observed_binary = event_operator.make_event_tables(forecast_wo_both, obs)
forecast_binary_wo_d, observed_binary = event_operator.make_event_tables(forecast_wo_d, obs)
forecast_binary_wo_w, observed_binary = event_operator.make_event_tables(forecast_wo_w, obs)


# In[ ]:


contingency_manager_w = categorical.BinaryContingencyManager(forecast_binary_w, observed_binary).transform(preserve_dims='lead_minute')
contingency_manager_wo_both = categorical.BinaryContingencyManager(forecast_binary_wo_both, observed_binary).transform(preserve_dims='lead_minute')
contingency_manager_wo_d = categorical.BinaryContingencyManager(forecast_binary_wo_d, observed_binary).transform(preserve_dims='lead_minute')
contingency_manager_wo_w = categorical.BinaryContingencyManager(forecast_binary_wo_w, observed_binary).transform(preserve_dims='lead_minute')


# In[ ]:


csi_w = contingency_manager_w.critical_success_index().data
csi_wo_d = contingency_manager_wo_d.critical_success_index().data
csi_wo_w = contingency_manager_wo_w.critical_success_index().data
csi_wo_both = contingency_manager_wo_both.critical_success_index().data

pod_w = contingency_manager_w.probability_of_detection().data
pod_wo_d = contingency_manager_wo_d.probability_of_detection().data
pod_wo_w = contingency_manager_wo_w.probability_of_detection().data
pod_wo_both = contingency_manager_wo_both.probability_of_detection().data


print(csi_w, csi_wo_both)


# In[ ]:


plt.figure(figsize=(8, 6))
plt.plot(np.arange(5, 95, 5), pod_w, label='with wind & dates')
plt.plot(np.arange(5, 95, 5), pod_wo_d, label='with wind')
plt.plot(np.arange(5, 95, 5), pod_wo_w, label='with dates')
plt.plot(np.arange(5, 95, 5), pod_wo_both, label='with nothing')
plt.legend()
plt.ylim(0, 0.9)


# In[ ]:


plt.figure(figsize=(8, 6))
plt.plot(np.arange(5, 95, 5), csi_w, label='with wind & dates')
plt.plot(np.arange(5, 95, 5), csi_wo_d, label='with wind')
plt.plot(np.arange(5, 95, 5), csi_wo_w, label='with dates')
plt.plot(np.arange(5, 95, 5), csi_wo_both, label='with nothing')
plt.legend()
plt.ylim(0, 0.8)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




