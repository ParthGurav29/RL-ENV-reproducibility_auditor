import torch
from torch.utils.data import Dataset
import random
import numpy as np

class HardDataset(Dataset):
    def __init__(self, size=1000):
        # BUG: The main training file sets the seed, but this separate 
        # dataset file doesn't inherit it properly if workers are spawned
        # leading to a cross-file seed gap.
        self.data = np.random.randn(size, 20)
        self.labels = [random.randint(0, 1) for _ in range(size)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)