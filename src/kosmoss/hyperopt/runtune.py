import json
import numpy as np
import os
import os.path as osp
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers.wandb import WandbLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hebo import HEBOSearch
import torch
from typing import List, Union

from kosmoss import ARTIFACTS_PATH, LOGS_PATH
from kosmoss.hyperopt.data import LitGNNDataModule
from kosmoss.hyperopt.models import LitGAT


def main() -> None:
    
    project = os.environ['wandb_project']
    
    def train_gnns(config: dict,
                   num_epochs: int = 10,
                   num_gpus: int = 0) -> None:
        
        datamodule = LitGNNDataModule(**config['data'])
        model = LitGAT(**config["model"])
    
        kwargs = {
            # "logger": WandbLogger(project=project),
            "gpus": num_gpus,
            "strategy": "ddp",
            "max_epochs": num_epochs,
            "callbacks": [
                TuneReportCallback({
                    "train_loss": "train_loss",
                    "val_loss": "val_loss"
                }, on=[
                    "batch_end",
                    "validation_end"
                ]),
            ],
            "progress_bar_refresh_rate": 0
        }
        trainer = Trainer(**kwargs)
        trainer.fit(model, datamodule)
    
    def tune_gnns_asha(num_samples: int = 4, 
                       num_epochs: int = 10, 
                       gpus_per_trial: int = 0) -> dict:
    
        config = {
            "data": {"batch_size": 256},
            "model": {
                # Fixed ins-and-outs
                "in_channels": 20,
                "out_channels": 4,
                
                # HPs, each sampled from its own search space
                "hidden_channels": int(tune.choice([2 ** k for k in np.arange(4, 6)]).sample()),
                "edge_dim": tune.choice([2 ** k for k in np.arange(4, 6)]),
                "num_layers": tune.randint(4, 10),
                "lr": tune.loguniform(1e-4, 1e-1),
                "dropout": tune.uniform(0, 1),
                "heads": int(tune.choice([4, 8]).sample())
            }
        }

        scheduler = ASHAScheduler(
            max_t=num_epochs,
            grace_period=num_epochs,
            reduction_factor=2)

        # Add on-the-fly named parameters to the set, as extra
        train_with_params_fn = tune.with_parameters(
            train_gnns,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial)

        # Run the execution flow
        analysis = tune.run(
            # Set the callable an resources to mobilize for every trial
            run_or_experiment=train_with_params_fn,
            resources_per_trial={
                "cpu": 4, 
                "gpu": gpus_per_trial
            },
            
            # Set the metric to watch, and how to consider metric progress
            metric="val_loss",
            mode="min",
            
            # Set the config dict of all HPs
            config=config,
            
            # Set the total number of tries
            num_samples=num_samples,
            
            # Set the execution scheduler
            scheduler=scheduler,
            
            # Set the search algorithm for sampling HPs
            search_alg=HEBOSearch(), #metric="val_loss", mode="min"),
            
            # Give a name prefix to all experiments
            name="tune_gnns_asha",
            
            # Set the logger to push results to your favorite logger, and local log dir
            callbacks=[WandbLoggerCallback(project=project, dir=LOGS_PATH)],
            local_dir=LOGS_PATH,
            # max_concurrent_trials=None
            
            # And Sh**-**!
            verbose=0
        )
        
        return analysis.best_config
    
    best_config = tune_gnns_asha()
    print("Best HP-set yet: ", best_config)


if __name__ == '__main__':
    
    # Turn on this flag to allow the report callback to fail gracefully if all metrics are not populated at each 'on' step
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = 1
    main()