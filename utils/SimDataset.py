import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

def get_simxrd_d_grid():
    d_grid = np.linspace(0.889, 17.659, 5000)
    return d_grid

def xrd2di(intensities):
    simxrd_d_grid = get_simxrd_d_grid()
    N = intensities.shape[0]
    batch_dI = np.zeros((N, len(simxrd_d_grid)))
    
    for i in range(N):
        wavelength = (1.54056 * 2 + 1.54439) / 3
        # print(angles_2theta[i])
        angles_2theta = np.arange(5, 120, 0.01)
        theta_rad = np.radians(angles_2theta / 2)
        d_values = wavelength / (2 * np.sin(theta_rad))

        valid_mask = (np.isfinite(d_values) & 
                      (d_values >= simxrd_d_grid.min()) & 
                      (d_values <= simxrd_d_grid.max()))
        d_valid = d_values[valid_mask]
        i_valid = intensities[i][valid_mask]
        if len(d_valid) < 2: 
            continue

        pad = np.min(simxrd_d_grid)
        interpolated = np.interp(simxrd_d_grid, d_valid[::-1], i_valid[::-1], left=pad, right=pad)
        interpolated = np.maximum(interpolated, 0)

        if np.max(interpolated) > 0:
            interpolated = np.sqrt(interpolated)
            interpolated = 100* (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))

        batch_dI[i] = interpolated
    
    return batch_dI

class SimDataset(Dataset):
    def __init__(self, datapath):
        self.data = np.load(datapath)
        self.names = self.data['names']
        # self.intensity = xrd2di(self.data['features'])
        self.intensity = torch.tensor(self.data['features'], dtype=torch.float32)
        self.crysystem = torch.tensor(self.data['labels7']+1, dtype=torch.int64)
        self.labels = torch.tensor(self.data['labels230']+1, dtype=torch.int64)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return {'name': self.names[idx],
                'intensity': self.intensity[idx], 
                'spg': self.labels[idx],
                'crysystem': self.crysystem[idx],}