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

"""Normal Pytorch"""
"""# class NN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x """



""" Lightning Module """
"""First way to do this Lightning module"""
# class NN(pl.LightningModule):
#     def __init__(self, input_size, num_classes):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, 50)
#         self.fc2 = nn.Linear(50, num_classes)
#         self.loss_fn = nn.CrossEntropyLoss()
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         x = x.reshape(x.size(0), -1)
#         scores = self.forward(x)
#         loss = self.loss_fn(scores, y)
#         return loss
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         x = x.reshape(x.size(0), -1)
#         scores = self.forward(x)
#         loss = self.loss_fn(scores, y)
#         return loss
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         x = x.reshape(x.size(0), -1)
#         scores = self.forward(x)
#         loss = self.loss_fn(scores, y)
#         return loss
#     def _common_step(self, batch, batch_idx):
#         x, y = batch
#         x = x.reshape(x.size(0), -1)
#         scores = self.forward(x)
#         loss = self.loss_fn(scores, y)
#         return loss, scores, y
#     def configure_optimizers(self):
#       return optim.Adam(self.parameters(), lr=0.001)
"""----------------------------------------------------------------------------------"""

"""This is second way and more cleaner compare to first one (You can make _common_step)"""   
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()
    def compute(self):
        return self.correct.float() / self.total.float()


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.my_accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy, "train_f1_score": f1_score},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


"""DataModule Step 5"""

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
        # Multiple GPU
        entire_dataset = datasets.MNIST(
            root = self.data_dir,
            transform=transforms.ToTensor(),
            train=True,
            download=True,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [5000, 10000])
        self.test_ds = datasets.MNIST(
            root = self.data_dir,
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
    


# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

"""# # Load Data
# entire_dataset = datasets.MNIST(
#     root="dataset/", train=True, transform=transforms.ToTensor(), download=True
# )
# train_ds, val_ds = random_split(entire_dataset, [50000, 10000])
# test_ds = datasets.MNIST(
#     root="dataset/", train=False, transform=transforms.ToTensor(), download=True
# )
# train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False) """

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

"""Step 5 (next)"""
dm = MnistDataModule(data_dir="dataset/", batch_size=batch_size, num_workers=4)

# Train Network
trainer = pl.Trainer(accelerator="gpu", devices=[0], min_epochs=1, max_epochs=3, precision=16)
trainer.fit(model, dm)
trainer.validate(model, dm)   # You can use this multiple times
trainer.test(model, dm)  # but this is only when we complete all things and doing deployment






"""This is Pytorch code"""

"""# # Check accuracy on training & test to see how good our model
# def check_accuracy(loader, model):
#     num_correct = 0
#     num_samples = 0
#     model.eval()

#     # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
#     with torch.no_grad():
#         # Loop through the data
#         for x, y in loader:

#             # Move data to device
#             x = x.to(device=device)
#             y = y.to(device=device)

#             # Get to correct shape
#             x = x.reshape(x.shape[0], -1)

#             # Forward pass
#             scores = model(x)
#             _, predictions = scores.max(1)

#             # Check how many we got correct
#             num_correct += (predictions == y).sum()

#             # Keep track of number of samples
#             num_samples += predictions.size(0)

#     model.train()
#     return num_correct / num_samples


# # Check accuracy on training & test to see how good our model
# model.to(device)
# print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
# print(f"Accuracy on validation set: {check_accuracy(val_loader, model)*100:.2f}")
# print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}") """