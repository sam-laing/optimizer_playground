import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# def MnistDataset(root: str, train: bool = True, transform=None):


class MnistDataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None):
        self.data = datasets.MNIST(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
