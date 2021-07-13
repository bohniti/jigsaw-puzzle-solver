# TODO Change os to pathlib because of windows, linux usbability
# TODO Change this model with the copy such that is naming correct!

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
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
            transforms.CenterCrop(512),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        self.train_data = SiameseNetworkDataset(
            csv_file=self.data_dir + '/train.csv', transform=self.transform)

        self.val_data = SiameseNetworkDataset(
            csv_file=self.data_dir + '/val.csv', transform=self.transform)

        # TODO Parameterize it
        self.margin = 1.0

        self.criterion = nn.BCEWithLogitsLoss()

        self.cnn1 = models.resnet50(pretrained=False)

        self.fc1 = nn.Sequential(
            nn.Linear(256000, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 8)
        )

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1).flatten()
        output2 = self.forward_once(input2).flatten()
        output = torch.abs(output1 - output2)

        output = self.fc1(output)
        print(output.shape)

        return output

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        output = self(x0, x1)
        loss = self.criterion(output, y)

        self.log('train_loss', loss, prog_bar=True)
        acc = self.binary_acc(output, y)

        self.log('train_acc', acc, prog_bar=True)
        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        output = self(x0, x1)

        loss = self.criterion(output, y)
        self.log('val_loss', loss, prog_bar=True)
        acc = self.binary_acc(output, y)
        self.log('val_acc', acc, prog_bar=True)
        return {"val_loss": loss, "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        self.log("avg_val_accuracy", avg_acc)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss)
        self.log("avg_train_accuracy", avg_acc)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=int(self.hyperparameters["batch_size"]), num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=int(self.hyperparameters["batch_size"]), num_workers=6)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters["lr"])
        return optimizer
