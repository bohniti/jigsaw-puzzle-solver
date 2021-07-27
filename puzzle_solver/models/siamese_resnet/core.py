import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.model_zoo as model_zoo


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
                SiameseNetwork(self.batch_size, self.learning_rate, self.margin, self.partial_conf, self.center_crop),
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

    from puzzle_solver.layers.PartialConv2d import PartialConv2d

    __all__ = ['PDResNet', 'pdresnet18', 'pdresnet34', 'pdresnet50', 'pdresnet101',
               'pdresnet152']

    model_urls = {
        'pdresnet18': '',
        'pdresnet34': '',
        'pdresnet50': '',
        'pdresnet101': '',
        'pdresnet152': '',
    }

    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return PartialConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(Bottleneck, self).__init__()
            self.conv1 = PartialConv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = PartialConv2d(planes, planes, kernel_size=3, stride=stride,
                                       padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = PartialConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    class PDResNet(nn.Module):

        def __init__(self, block, layers, num_classes=1000):
            self.inplanes = 64
            super(PDResNet, self).__init__()
            self.conv1 = PartialConv2d(3, 64, kernel_size=7, stride=1, padding=0,
                                       bias=False, multi_channel=True)

            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, PartialConv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    PartialConv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            mask = x.clone()
            # TODO parameterize the threshold or change it if needed
            mask[mask != 0] = 1

            x = self.conv1(input=x, mask_in=mask)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

    def pdresnet18(pretrained=False, **kwargs):
        """Constructs a PDResNet-18 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = PDResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['pdresnet18']))
        return model

    def pdresnet50(pretrained=False, **kwargs):
        """Constructs a PDResNet-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = PDResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['pdresnet50']))
        return model

    def pdresnet101(pretrained=False, **kwargs):
        """Constructs a PDResNet-101 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = PDResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['pdresnet101']))
        return model

    def pdresnet152(pretrained=False, **kwargs):
        """Constructs a PDResNet-152 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        model = PDResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['pdresnet152']))
        return model

