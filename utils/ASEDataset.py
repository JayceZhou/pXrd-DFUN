import torch
from torch.utils.data import Dataset, DataLoader
import ase.db
import json

class ASEDataset(Dataset):
    def __init__(self, db_paths):
        self.db_paths = db_paths
        print(db_paths)
        self.dbs = [ase.db.connect(db_path) for db_path in db_paths]
        print("Loaded data from:", db_paths)

    def __len__(self):
        total_length = sum(len(db) for db in self.dbs)
        return total_length

    def __getitem__(self, idx):
        
        cumulative_length = 0
        for i, db in enumerate(self.dbs):
            if idx < cumulative_length + len(db):
                # Adjust the index to the range of the current database
                adjusted_idx = idx - cumulative_length
                row = db.get(adjusted_idx + 1)  # ASE db indexing starts from 1

                intensity = eval(getattr(row, 'intensity'))
                spg = eval(getattr(row, 'tager'))[0]
                crysystem = eval(getattr(row, 'tager'))[1]

                # Convert to tensors
                tensor_intensity = torch.tensor(intensity, dtype=torch.float32)
                tensor_spg = torch.tensor(spg, dtype=torch.int64)
                tensor_crysystem = torch.tensor(crysystem, dtype=torch.int64)
                return {
                    'intensity': tensor_intensity,
                    'spg': tensor_spg,
                    'crysystem': tensor_crysystem,
                    'element': torch.tensor([])
                }              
            cumulative_length += len(db)