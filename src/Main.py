import math

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from src.models.LightningMNISTClassifier import LightningMNISTClassifier
from src.utils.utils import load_config


def training_function(config, data_dir=None, num_epochs=10, num_gpus=0):
    model = LightningMNISTClassifier(config, data_dir)
    print(tune.get_trial_dir())
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


def main(hyperparameters, config):
    num_samples = config['num_samples']
    num_epochs = config['num_epochs']
    gpus_per_trial = config['gpus_per_trial']
    cpus_per_trial = config['cpus_per_trial']
    raw_data_dir = config['raw_data_dir']
    result_dir = config['result_dir']
    scheduler_config = config['Scheduler']
    scheduler_type = scheduler_config['grace_period']
    scheduler_grace_period = scheduler_config['grace_period']
    scheduler_reduction_factor = scheduler_config['reduction_factor']

    LightningMNISTClassifier.download_data(raw_data_dir)

    scheduler = None

    if scheduler_type == 'ASHAScheduler':
        scheduler = ASHAScheduler(
            max_t=num_epochs,
            grace_period=scheduler_grace_period,
            reduction_factor=scheduler_reduction_factor)
    else:
        NotImplementedError()

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            training_function,
            data_dir=raw_data_dir,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        local_dir=result_dir,
        config=hyperparameters,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    hyperparameter_file = load_config('hyperparameters', show=True, tune_config=True)
    config_file = load_config('config', show=True, tune_config=False)
    main(hyperparameters=hyperparameter_file, config=config_file)
