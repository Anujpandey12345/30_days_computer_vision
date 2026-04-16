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

from model import NN
from dataset import MnistDataModule
import config



if __name__ == "__main__":
    model = NN(input_size=config.INPUT_SIZE,learning_rate = config.LEARNING_RATE, num_classes=config.NUM_CLASSES)
    # # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    """Step 5 (next)"""
    dm = MnistDataModule(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    # Train Network
    trainer = pl.Trainer(accelerator=config.ACCELERATOR, devices=config.DEVICES, min_epochs=1, max_epochs=3, precision=config.PRECISION)
    trainer.fit(model, dm)
    trainer.validate(model, dm)   # You can use this multiple times
    trainer.test(model, dm)  # but this is only when we complete all things and doing deployment