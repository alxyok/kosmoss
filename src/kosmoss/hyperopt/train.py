import json
import os
import os.path as osp
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks.base import Callback
# from pytorch_lightning.loggers import WandbLogger #TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.wandb import WandbLoggerCallback
import torch
from typing import List, Union

from kosmoss import ARTIFACTS_PATH, LOGS_PATH
from kosmoss.hyperopt.data import LitGNNDataModule
from kosmoss.hyperopt.models import LitGAT


def main():
    
    def train_gnns(config,
              num_epochs=10,
              num_gpus=1):
        
        datamodule = LitGNNDataModule(batch_size=512)
        model = LitGAT(
            in_channels=20,
            out_channels=4,
            hidden_channels=config['hidden_channels'],
            edge_dim=config['edge_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            heads=config['heads'],
            lr=config['lr']
        )
    
        kwargs = {
            # "logger": WandbLoggerCallback(project=os.environ['wandb_project']),
            "gpus": num_gpus,
            "strategy": "ddp",
            "max_epochs": num_epochs,
            "callbacks": [
                TuneReportCallback({
                    "loss": "train_loss"
                }, on="batch_end"),
                # TuneReportCallback({
                #     "loss": "val_loss"
                # }, on="validation_end"),
            ],
            "progress_bar_refresh_rate": 0
        }

        trainer = Trainer(**kwargs)
        trainer.fit(model, datamodule)
        # trainer.test(model, datamodule)
    
    def tune_gnns_asha(num_samples=10, num_epochs=10, gpus_per_trial=1):
    
        config = {
            "hidden_channels": tune.choice([2 ** k for k in range(4)]),
            "num_layers": tune.randint(4, 10),
            "lr": tune.loguniform(1e-4, 1e-1),
            "dropout": tune.uniform(0, 1),
            "heads": tune.randint(4, 8)
        }

        scheduler = ASHAScheduler(
            max_t=num_epochs,
            grace_period=10,
            reduction_factor=2)

        # reporter = CLIReporter(
        #     parameter_columns=list(config.keys()),
        #     metric_columns=["loss", "training_iteration"])

        train_fn_with_parameters = tune.with_parameters(train_gnns,
                                                        num_epochs=num_epochs,
                                                        num_gpus=gpus_per_trial)
        resources_per_trial = {"cpu": 4, "gpu": gpus_per_trial}

        analysis = tune.run(
            train_fn_with_parameters,
            resources_per_trial=resources_per_trial,
            metric="loss",
            mode="min",
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            # progress_reporter=reporter,
            name="tune_gnns_asha",
            callbacks=[WandbLoggerCallback(project=os.environ['wandb_project'])],
            max_concurrent_trials=None
        )

        print("Best hyperparameters found were: ", analysis.best_config)
        
    tune_gnns_asha()


if __name__ == '__main__':
    
    main()
