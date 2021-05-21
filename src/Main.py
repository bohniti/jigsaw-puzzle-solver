import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from src.models.LightningMNISTClassifier import LightningMNISTClassifier
from src.utils.utils import load_config

# TODO parameterize it
callback = TuneReportCallback({
    "loss": "avg_val_loss",
    "mean_accuracy": "avg_val_accuracy"
}, on="validation_end")


def train_mnist(config):
    model = LightningMNISTClassifier(config)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model)


def train_mnist_tune(config, data_dir=None, num_epochs=10, num_gpus=0):
    model = LightningMNISTClassifier(config, data_dir)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy"
                },
                on="validation_end")
        ])
    trainer.fit(model)


if __name__ == "__main__":
    tune = True
    if tune:
        # TODO Parameterize it
        #tune_config = load_config('dev_config')
        config = {
            "layer_1_size": tune.choice([32, 64, 128]),
            "layer_2_size": tune.choice([64, 128, 256]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([32, 64, 128]),
        }
        train_mnist_tune(config, data_dir=None, num_epochs=10, num_gpus=0)
    else:
        dev_config = load_config('dev_config')
        train_mnist(dev_config)
