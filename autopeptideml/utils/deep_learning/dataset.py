"""
Code Adapted from the UniDL4BioPep
implementation of their model for PyTorch
in the GitHub Repository:
https://github.com/David-Dingle/UniDL4BioPep_ASL_PyTorch/
"""

import torch
from torch.utils.data import Dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class UniDL4BioPep_Dataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.data = torch.from_numpy(x).float().to(device)
        self.labels = torch.from_numpy(y).float().to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.data

class UniDL4BioPep_Inference(Dataset):
    def __init__(self, x):
        super().__init__()
        self.data = torch.from_numpy(x).float().to(device)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]