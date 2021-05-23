import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune import CLIReporter
from src.models.LightningMNISTClassifier import LightningMNISTClassifier
from src.utils.utils import load_config
from ray.tune.schedulers import ASHAScheduler
import math
from pytorch_lightning.loggers import TensorBoardLogger


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


def tune_mnist_asha(hyperparameters, config):
    """:cvar
    num_samples = 10
    num_epochs = 10
    gpus_per_trial = 0

    """
    num_samples = config['num_samples']
    num_epochs = config['num_epochs']
    gpus_per_trial = config['gpus_per_trial']

    #data_dir = os.path.expanduser("~/data")
    data_dir = '/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/src/KMNIST/raw'
    LightningMNISTClassifier.download_data(data_dir)

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_mnist_tune,
            data_dir=data_dir,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=hyperparameters,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    use_tune = True
    if use_tune:
        hyperparameters = load_config('hyperparameters', show=True, tune_config=True)
        config = load_config('config', show=True, tune_config=False)
        tune_mnist_asha(hyperparameters=hyperparameters, config=config)
    else:
        dev_config = load_config('dev_config', show=True, tune_config=use_tune)
        train_mnist(dev_config)
