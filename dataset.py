import torch
from torch.utils.data import Dataset

class DummyAudioDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=10):
        self.data = torch.randn(num_samples, 1, 128, 128)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
