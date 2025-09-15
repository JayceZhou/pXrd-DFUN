import pandas as pd
import numpy as np  
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt
import torch
from torch.utils.data import Dataset

def normalise(spectra):
    if type(spectra) is np.ndarray:
        max_I = np.max(spectra)
        min_I = np.min(spectra)    
    elif type(spectra) is torch.Tensor:
        max_I = max(spectra)
        min_I = min(spectra)
    spectra_normed = (spectra - min_I) / (max_I - min_I) * 100
    return spectra_normed

class data_augmentation():
    def __init__(self, settings, settings_aug):
        self.settings = settings
        self.settings_aug = settings_aug
    
    def peak_elimination(self, xrd):
        random_window = torch.from_numpy(
            np.random.choice([0,0,1], self.settings_aug[0]),
        )
        dum1 = random_window.repeat(self.settings[2]//self.settings_aug[0]+1)[:self.settings[2]]
        xrd_el = torch.mul(xrd, dum1)
        return xrd_el
    
    def peak_scaling(self, xrd):
        xrd_sc = xrd.clone()
        n_points = len(xrd)
        scale_range = (0.7, 1.3)
        block_size = self.settings_aug[0]
        n_blocks = n_points // block_size
        
        for i in range(n_blocks):
            if torch.rand(1) < 0.1:
                scale_factor = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
                start_idx = i * block_size
                end_idx = start_idx + block_size
                
                xrd_sc[start_idx:end_idx] *= scale_factor.to(xrd.device)

        # scaling_mask = torch.ones(len(xrd))
        # indices_to_change = torch.randperm(len(xrd))[:int(0.2 * len(xrd))] # 随机选30%的点
        # scaling_mask[indices_to_change] = torch.rand(len(indices_to_change))
        # # random_window = torch.rand(self.settings_aug[0])
        # # dum2 = random_window.repeat(self.settings[2]//self.settings_aug[0]+1)[:self.settings[2]]
        # xrd_sc = torch.mul(xrd, scaling_mask)
        return xrd_sc
    
    def peak_shift(self, xrd):
        max_shift = self.settings_aug[1]
        cut_tensor = torch.randint(
            -max_shift,
            max_shift + 1, 
            (1,),
        )
        cut = cut_tensor.item()

        if cut == 0:
            return xrd
        
        elif cut > 0:
            sliced_xrd = xrd[cut:]
            padding = torch.zeros(cut, device=xrd.device) 
            xrd_sh = torch.cat([sliced_xrd, padding], dim=0)
        else: 
            abs_cut = -cut
            sliced_xrd = xrd[:-abs_cut] 
            padding = torch.zeros(abs_cut, device=xrd.device)
            xrd_sh = torch.cat([padding, sliced_xrd], dim=0)
        return xrd_sh
    
    def forward(self, xrd):
        if torch.rand(1) < self.settings_aug[2]:
            xrd = self.peak_elimination(xrd)
        if torch.rand(1) < self.settings_aug[3]:
            xrd = self.peak_scaling(xrd)
        if torch.rand(1) < self.settings_aug[4]:
            xrd = self.peak_shift(xrd)
        return normalise(xrd)
    
class AugmentedDataset(Dataset):
    def __init__(self, tensors, settings, settings_aug):
        self.tensors = tensors
        self.settings = settings
        self.settings_aug = settings_aug
        self.augmentation = data_augmentation(settings, settings_aug)
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.tensors[0][index][0]).float().to(self.settings[4])
        x = self.augmentation.forward(x).unsqueeze(0)
        y = torch.tensor(
            self.tensors[1][index].astype(np.float32)
        ).to(self.settings[4])
        return x, y
    
    def __len__(self):
        return len(self.tensors[0])