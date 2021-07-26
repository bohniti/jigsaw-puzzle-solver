import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from ResNet50 import pdresnet50
import matplotlib.pyplot as plt
import numpy as np

class SiameseNetwork(pl.LightningModule):

    def __init__(self, batch_size, learning_rate, margin, partial_conf, center_crop):
        # Inherit from base class
        super().__init__()

        self.margin = margin
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.partial_conf = partial_conf
        self.reference_image = None
        self.center_crop = center_crop
        print(batch_size)

        self.criterion = nn.BCEWithLogitsLoss()

        if self.partial_conf == 0:
            self.cnn1 = models.resnet50(pretrained=False)
        else:
            self.cnn1 = pdresnet50(pretrained=False)

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 1000, 500),
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

    def forward(self, input1, input2):
        self.reference_image = input1
        # print(self.reference_img.shape)
        output1 = self.cnn1(input1).flatten()
        output2 = self.cnn1(input2).flatten()
        output = torch.abs(output1 - output2)
        output = self.fc1(output)
        return output

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        output = self(x0, x1)
        loss = self.criterion(output, y)
        acc = self.binary_acc(output, y)

        # self.log('train_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log('train_acc', acc, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        output = self(x0, x1)
        loss = self.criterion(output, y)
        acc = self.binary_acc(output, y)

        # self.log('val_acc', acc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log('val_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_accuracy": acc}

    def custom_histogram_adder(self):

        # iterating through all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def makegrid(self, output, numrows):
        outer = (torch.Tensor.cpu(output).detach())
        plt.figure(figsize=(20, 5))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(numrows * outer.shape[2], 0)
        i = 0
        j = 0
        while (i < outer.shape[1]):
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if (j == numrows):
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        return c

    def show_activatioons(self, x):
        # logging reference image
        self.logger.experiment.add_image("input", torch.Tensor.cpu(x[0][0]), self.current_epoch, dataformats="HW")

        # logging cnn resnet activations
        out = self.cnn1(x)
        c = self.makegrid(out, 4)
        self.logger.experiment.add_image("layer 1", c, self.current_epoch, dataformats="HW")

        # logging fc activations
        out = self.fcn1(out)
        c = self.makegrid(out, 8)
        self.logger.experiment.add_image("layer 2", c, self.current_epoch, dataformats="HW")

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Validation", avg_acc, self.current_epoch)

    def training_epoch_end(self, outputs):

        if (self.current_epoch == 1):
            cover_img = torch.rand((self.batch_size, 3, self.center_crop, self.center_crop))
            cover_img2 = torch.rand((self.batch_size, 3, self.center_crop, self.center_crop))
            self.logger.experiment.add_graph(
                SiameseNetwork(self.batch_size, self.learning_rate, self.margin, self.partial_conf),
                [cover_img, cover_img2])

        self.custom_histogram_adder()
        #self.show_activatioons(self.reference_image)

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", avg_acc, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
