from src.models.LightningMNISTClassifier import LightningMNISTClassifier
from src.utils.utils import load_config
import pytorch_lightning as pl
import json


def train_mnist(config):
    model = LightningMNISTClassifier(config)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model)


if __name__ == "__main__":
    dev_config = load_config('dev_config')
    train_mnist(dev_config)
