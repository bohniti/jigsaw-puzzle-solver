# TODO Change os to pathlib because of windows, linux usbability
# TODO Change this model with the copy such that is naming correct!

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ContrastiveLoss import ContrastiveLoss
from SiameseNetworkDataset import SiameseNetworkDataset
from torch.utils.data import DataLoader
from torchvision import transforms


# TODO Change pytorch version because of non support for autologging
# TODO Refactors such that everything is in sperate Files
# TODO Implement MLFlow Logger for Parameters
# TODO Implement MLFlow credential in config
class SiameseNetwork(pl.LightningModule):

    def __init__(self, hyperparameters,
                 data_dir='/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/prepared/paris_as_csv/'):
        # Inherit from base class
        super().__init__()

        self.hyperparameters = hyperparameters

        self.data_dir = data_dir

        # init trianing hyperparameters ...
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((378, 371))
        ])

        self.train_data = SiameseNetworkDataset(
            csv_file=self.data_dir + 'train.csv', transform=self.transform)

        self.val_data = SiameseNetworkDataset(
            csv_file=self.data_dir + 'val.csv', transform=self.transform)

        # TODO Parameterize it
        self.margin = 1.0

        self.criterion = ContrastiveLoss(margin=self.margin)

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        output1, output2 = self(x0, x1)
        loss = self.criterion(output1, output2, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        output1, output2 = self(x0, x1)
        loss = self.criterion(output1, output2, y)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=int(self.hyperparameters["lr"]), num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=int(self.hyperparameters["batch_size"]), num_workers=6)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters["lr"])
        return optimizer
