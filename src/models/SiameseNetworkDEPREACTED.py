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
            transforms.CenterCrop(400),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        self.train_data = SiameseNetworkDataset(
            csv_file=self.data_dir + 'dev_train.csv', transform=self.transform)

        self.val_data = SiameseNetworkDataset(
            csv_file=self.data_dir + 'dev_val.csv', transform=self.transform)

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
            nn.Linear(8 * 400 * 400, 500),
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
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        output1, output2 = self(x0, x1)
        loss = self.criterion(output1, output2, y)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=int(self.hyperparameters["batch_size"]), num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=int(self.hyperparameters["batch_size"]), num_workers=6)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters["lr"])
        return optimizer
