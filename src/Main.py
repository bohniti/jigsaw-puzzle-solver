import math
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
import matplotlib.pyplot as plt
from src.models.LightningMNISTClassifier import LightningMNISTClassifier
from src.utils.utils import load_config
import torch

# mlflow.set_tracking_uri("databricks")
# mlflow.set_experiment("/Jiggsaw_test")

logger = MLFlowLogger(tracking_uri='databricks', experiment_name="/Users/timo.bohnstedt@fau.de/Jiggsaw_test")


@mlflow_mixin
def training_function(config, data_dir=None, num_epochs=10, num_gpus=0):
    model = LightningMNISTClassifier(config, data_dir)
    mlflow.autolog()
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        #logger=logger,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
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
    num_trials = config['num_trials']
    num_epochs = config['num_epochs']
    gpus_per_trial = config['gpus_per_trial']
    cpus_per_trial = config['cpus_per_trial']
    raw_data_dir = config['raw_data_dir']
    result_dir = config['result_dir']
    figures_dir = config['figures_dir']
    scheduler_config = config['Scheduler']
    scheduler_type = scheduler_config['grace_period']
    scheduler_grace_period = scheduler_config['grace_period']
    scheduler_reduction_factor = scheduler_config['reduction_factor']
    figures_dir = config['figures_dir']

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

    hyperparameters['mlflow'] = {
        "experiment_name": "/Users/timo.bohnstedt@fau.de/Jiggsaw_test",
        "tracking_uri": "databricks"
    }

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
        num_samples=num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha")

    hyperparameter_results_dic = analysis.best_config
    hyperparameter_results_dic.pop('mlflow', None)
    best_config_string = f'Best hyperparameters found were:\n{analysis.best_config}'
    best_config_string = best_config_string.replace('{', '')
    best_config_string = best_config_string.replace('}', '')
    print(best_config_string)

    dfs = analysis.trial_dataframes

    # Plot by epoch
    fig, ax = plt.subplots()
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Accuracy")
    plt.title(best_config_string, fontsize=10, pad=20)
    plt.suptitle(f'Mean Accuracy over {num_epochs} Epochs and {num_trials} Trails', fontsize=16)
    fig.tight_layout()
    plt.savefig(figures_dir + 'mean_accuracy.png')
    plt.show()

    df = analysis.results_df
    logdir = analysis.get_best_logdir("mean_accuracy", mode="max")
    print(logdir)
    torch.save()


if __name__ == "__main__":
    hyperparameter_file = load_config('hyperparameters', show=True, tune_config=True)
    config_file = load_config('config', show=True, tune_config=False)
    main(hyperparameters=hyperparameter_file, config=config_file)
