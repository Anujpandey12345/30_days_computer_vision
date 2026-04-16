import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    def predict_dataloader(self):
        # Single GPU
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
    def setup(self, stage=None):
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            # Class --- 8
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]),
            train=True,
            download=False,
        )

        train_size = int(0.8 * len(entire_dataset))  # 48000
        val_size = len(entire_dataset) - train_size  # 12000

        self.train_ds, self.val_ds = random_split(entire_dataset, [train_size, val_size])

        self.test_ds = datasets.MNIST(
            root=self.data_dir,
            transform=transforms.ToTensor(),
            train=False,
            download=False,
        )
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle=False

        )
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle=False

        )
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle=False

        )