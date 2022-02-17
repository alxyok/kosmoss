import numpy as np
import os.path as osp
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import sys
import torch
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset

from kosmoss import CONFIG, DATA_PATH, PARAMS
from kosmoss.dataproc.flows import BuildGraphsFlow


class ThreeDCorrectionDataset(InMemoryDataset):
    
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


@DATAMODULE_REGISTRY
class LitThreeDCorrectionDataModule(LightningDataModule):
    
    def __init__(self, 
                 batch_size: int,
                 num_workers: int) -> None:
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()
        
        
    def prepare_data(self) -> None:
        pass
    
    
    def setup(self, stage: str) -> None:
        dataset = ThreeDCorrectionDataset(DATA_PATH).shuffle()
        length = len(dataset)
        
        self.test_dataset = dataset[1000000:]
        self.val_dataset = dataset[900000:1000000]
        self.train_dataset = dataset[:900000]
    
    
    def train_dataloader(self) -> torch.data.loader.DataLoader:
        
        return pyg.loader.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers)
    
    
    def val_dataloader(self) -> torch.data.loader.DataLoader:
        
        return pyg.loader.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers)
    
    
    def test_dataloader(self) -> torch.data.loader.DataLoader:
        
        return pyg.loader.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers)