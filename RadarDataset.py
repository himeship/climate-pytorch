#!/usr/bin/env python
# coding: utf-8


# In []:


import os
import torch

from torch.utils.data import Dataset


# In [1]: Rainrate


class RadarRainrateDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = os.listdir(self.root_dir)
        self.labels.sort()
        self.annotations = [label[:-3] for label in self.labels]      # '-3' is for '.pt' suffix
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.annotations[index] + '.pt')
        video = torch.load(path)
        timelabel = self.annotations[index]
        # define transform
        if self.transform:
            video = self.transform(video)
        return (video, timelabel)


# In [2]: Rainrate & Wind


class RadarRainWindDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = os.listdir(self.root_dir)
        self.labels.sort()
        self.annotations = [label[:-3] for label in self.labels]      # '-3' is for '.pt' suffix
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.annotations[index] + '.pt')
        video = torch.load(path)
        timelabel = self.annotations[index]
        # define transform
        if self.transform:
            video = self.transform(video)
        return (video, timelabel)
