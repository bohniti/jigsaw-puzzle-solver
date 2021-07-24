import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models


class SiameseNetwork(pl.LightningModule):

    def __init__(self, batch_size, learning_rate, margin):
        # Inherit from base class
        super().__init__()

        self.margin = margin
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        print(batch_size)

        self.criterion = nn.BCEWithLogitsLoss()

        self.cnn1 = models.resnet50(pretrained=False)

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

    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1).flatten()
        output2 = self.forward_once(input2).flatten()
        output = torch.abs(output1 - output2)
        output = self.fc1(output)
        return output

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        output = self(x0, x1)
        loss = self.criterion(output, y)
        acc = self.binary_acc(output, y)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch, batch_idx):
        x0, x1, y = batch
        output = self(x0, x1)
        loss = self.criterion(output, y)
        acc = self.binary_acc(output, y)

        self.log('val_acc', acc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_accuracy": acc}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss,  logger=True,on_step=False, on_epoch=True)
        self.log("avg_val_accuracy", avg_acc,  logger=True,on_step=False, on_epoch=True)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss, logger=True, on_step=False, on_epoch=True)
        self.log("avg_train_accuracy", avg_acc,  logger=True,on_step=False, on_epoch=True)



    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
