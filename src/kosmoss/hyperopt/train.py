import json
import os
import os.path as osp
import numpy as np
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks.base import Callback
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.wandb import WandbLoggerCallback
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_optimizer as optim
from torch_geometric.data import Dataset
import torchmetrics.functional as F
from typing import List, Union

from kosmoss import ARTIFACTS_PATH, CONFIG, DATA_PATH, LOGS_PATH, METADATA
from kosmoss.dataproc.flows import BuildGraphsFlow



class LitGNNDataModule(LightningDataModule):
    
    class GNNDataset(Dataset):

        def __init__(self) -> None:

            self.timestep = str(CONFIG['timestep'])
            self.params = METADATA[str(self.timestep)]['features']
            self.num_shards = self.params['num_shards']

            super().__init__(DATA_PATH)

        @property
        def raw_file_names(self) -> list:
            return [""]

        @property
        def processed_file_names(self) -> List[str]:
            return [osp.join(f"graphs-{self.timestep}", f"data-{shard}.pt") 
                    for shard in np.arange(self.num_shards)]


        def download(self) -> None:
            raise Exception("Execute the Notebooks in this Bootcamp following the order defined by the Readme.")


        def process(self) -> None:
            BuildGraphsFlow()

        def len(self):
            return self.params['dataset_len']

        def get(self, idx):

            shard_size = self.len() // self.num_shards
            fileidx = idx // shard_size
            rowidx = idx % shard_size

            data_list = torch.load(osp.join(self.processed_dir, f"graphs-{self.timestep}", f'data-{fileidx}.pt'))
            data = data_list[rowidx]

            return data
        
    
    def __init__(self, batch_size: int) -> None:
        
        self.bs = batch_size
        super().__init__()
        
        
    def prepare_data(self) -> None:
        pass
    
    
    def setup(self, stage: str) -> None:
        dataset = LitGNNDataModule.GNNDataset().shuffle()
        length = len(dataset)
        
        self.testds = dataset[int(length * .9):]
        self.valds = dataset[int(length * .8):int(length * .9)]
        self.trainds = dataset[:int(length * .8)]
    
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return pyg.loader.DataLoader(self.trainds, batch_size=self.bs, num_workers=4, shuffle=True)
    
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return pyg.loader.DataLoader(self.valds, batch_size=self.bs, num_workers=4)
    
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return pyg.loader.DataLoader(self.testds, batch_size=self.bs, num_workers=4)

    
    
class LitGAT(LightningModule):
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config['lr']
        import pprint
        pprint.pprint(config)
        self.net = pyg.nn.GAT(
            in_channels=config['in_feats'], 
            out_channels=config['out_feats'],
            hidden_channels=config['hidden_feats'],
            num_layers=config['num_layers'],
            edge_dim=config['edge_dim'],
            dropout=config['dropout'],
            heads=config['heads'],
            act=config['act']
        )

    def forward(self, x, edge_index):
        return self.net(x, edge_index)
    
    def _common_step(self, 
                     batch, 
                     batch_idx, 
                     stage: Union['train', 'val', 'test'] = "train") -> List[torch.Tensor]:
        
        y_hat = self(batch.x, batch.edge_index)
        loss = F.mean_squared_error(y_hat, batch.y)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        
        return y_hat, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, loss = self._common_step(batch, batch_idx, "val")
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        y_hat, _ = self._common_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)
    


def main():
    
    def train_gnns(config,
                   num_epochs=10,
                   num_gpus=1):
        
        datamodule = LitGNNDataModule(config['batch_size'])
        model = LitGAT(config)
    
        kwargs = {
            # "logger": WandbLoggerCallback(project=os.environ['wandb_project']),
            "gpus": num_gpus,
            "strategy": "ddp",
            "max_epochs": num_epochs,
            "callbacks": [
                TuneReportCallback({
                    "loss": "train_loss"
                }, on="batch_end"),
            ],
            "progress_bar_refresh_rate": 0
        }

        trainer = Trainer(**kwargs)
        trainer.fit(model, datamodule)
    
    def tune_gnns_asha(num_samples=10, 
                       num_epochs=10, 
                       gpus_per_trial=1):
    
        config = {
            # Fixed set of HParams
            "batch_size": 512,
            "in_feats": 20,
            "out_feats": 4,
            "act": nn.SiLU(inplace=True),
            
            # HPO sampled by Ray.Tune
            "hidden_feats": tune.choice([2 ** k for k in np.arange(2, 6)]),
            "edge_dim": tune.choice([2 ** k for k in np.arange(2, 6)]),
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
