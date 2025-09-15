from sklearn.calibration import LabelEncoder
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.dataload import *

class CristalDataset(Dataset):
    def __init__(self, intensities, label):
        self.data = torch.tensor(intensities, dtype=torch.float32)
        self.labels = torch.tensor(label+1, dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  
        # dx1 = np.gradient(x)
        # x3 = np.stack([x, dx1], axis=0) 
        # x3 = torch.tensor(x3, dtype=torch.float32)
        return {'intensity': x, 
                'spg': self.labels[idx]}