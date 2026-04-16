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
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger #class -- 8
from pytorch_lightning.profilers import PyTorchProfiler



if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v0")  #class -- 8
    profiler = PyTorchProfiler(
        on_trace_ready = torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule = torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
        
    )
    model = NN(input_size=config.INPUT_SIZE,learning_rate = config.LEARNING_RATE, num_classes=config.NUM_CLASSES)
    # # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    """Step 5 (next)"""
    dm = MnistDataModule(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    # Train Network
    trainer = pl.Trainer(
        profiler=profiler,
        logger=logger,#class -- 8
        accelerator=config.ACCELERATOR, 
        devices=config.DEVICES, 
        min_epochs=1, 
        max_epochs=config.NUM_EPOCHS, 
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)   # You can use this multiple times
    trainer.test(model, dm)  # but this is only when we complete all things and doing deployment