import numpy as np
import os.path as osp
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset
from typing import List

from kosmoss import CONFIG, DATA_PATH, PARAMS
from kosmoss.dataproc.flows import BuildGraphsFlow


class GNNDataset(InMemoryDataset):
    
    def __init__(self, root: str) -> None:
        
        self.timestep = CONFIG['timestep']
        self.params = PARAMS[str(self.timestep)]['features']
        self.num_shards = self.params['num_shards']
        
        super().__init__(root)
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> None:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        return [f"data-{self.timestep}.{shard}.pt" for shard in np.arange(self.num_shards)]
    
    
    def download(self) -> None:
        raise Exception("Execute the Notebooks in this Bootcamp following natural order.")

        
    def process(self) -> None:
        
        BuildGraphsFlow()


# @DATAMODULE_REGISTRY
class LitGNNDataModule(LightningDataModule):
    
    def __init__(self, batch_size: int) -> None:
        
        self.batch_size = batch_size
        super().__init__()
        
        
    def prepare_data(self) -> None:
        pass
    
    
    def setup(self, stage: str) -> None:
        dataset = GNNDataset(DATA_PATH).shuffle()
        length = len(dataset)
        
        self.test_dataset = dataset[1000000:]
        self.val_dataset = dataset[900000:1000000]
        self.train_dataset = dataset[:900000]
    
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        
        return pyg.loader.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True)
    
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        
        return pyg.loader.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size)
    
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        
        return pyg.loader.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size)