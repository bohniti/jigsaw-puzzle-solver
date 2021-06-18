# TODO Change os to pathlib because of windows, linux usbability
import os

import mlflow
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import KMNIST


# TODO Change pytorch version because of non support for autologging
# TODO Refactors such that everything is in sperate Files
# TODO Implement MLFlow Logger for Parameters
# TODO Implement MLFlow credential in config
class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, hyperparameters, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()
        self.data_dir = data_dir or os.getcwd()

        self.layer_1_size = hyperparameters["layer_1_size"]
        self.layer_2_size = hyperparameters["layer_2_size"]
        self.lr = hyperparameters["lr"]
        self.batch_size = hyperparameters["batch_size"]

        mlflow.log_param("layer_1_size", hyperparameters["layer_1_size"])
        mlflow.log_param("layer_2_size", hyperparameters["layer_2_size"])
        mlflow.log_param("lr", hyperparameters["lr"])
        mlflow.log_param("batch_size", hyperparameters["batch_size"])

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        """
        self.logger.log_hyperparams(
            {'Learning Rate': self.lr, 'Layer 2 Size': self.layer_2_size, 'Layer 1 Size': self.layer_1_size,
             'Batch Size': self.batch_size})
        
        """

        # self.logger.log_metrics({'Train Loss': loss, 'Train Accuracy': accuracy})

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    @staticmethod
    def download_data(data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((378,371)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return KMNIST(data_dir, train=True, download=True, transform=transform)

    def prepare_data(self):
        mnist_train = self.download_data(self.data_dir)

        self.mnist_train, self.mnist_val = random_split(
            mnist_train, [55000, 5000])

    def train_dataloader(self):
        # TODO num_workers to external files
        return DataLoader(self.mnist_train, batch_size=int(self.batch_size), num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=int(self.batch_size), num_workers=6)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
