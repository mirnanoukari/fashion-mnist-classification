import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST

import pytorch_lightning as pl

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
    def prepare_data(self):
        # Download data if needed
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Load the train set
            self.train_dataset = FashionMNIST(self.data_dir, train=True, download=False ,transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(self.train_dataset, [55000, 5000])
            
        if stage == 'test' or stage is None:
            # Load the test set
            self.test_dataset = FashionMNIST(self.data_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=torch.get_num_threads())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=torch.get_num_threads())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=torch.get_num_threads())
