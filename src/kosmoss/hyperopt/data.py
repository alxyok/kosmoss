import numpy as np
import os.path as osp
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset
from typing import List

from kosmoss import CONFIG, DATA_PATH, METADATA
from kosmoss.dataproc.flows import BuildGraphsFlow


class GNNDataset(Dataset):
    
    def __init__(self) -> None:
        
        self.timestep = str(CONFIG['timestep'])
        self.params = METADATA[str(self.timestep)]['features']
        self.num_shards = self.params['num_shards']
        
        super().__init__(DATA_PATH)

    @property
    def raw_file_names(self) -> None:
        return [""]

    @property
    def processed_file_names(self) -> List[str]:
        return [osp.join(f"graphs-{self.timestep}", f"data-{shard}.pt") 
                for shard in np.arange(self.num_shards)]
    
    
    def download(self) -> None:
        raise Exception("Execute the Notebooks in this Bootcamp following the order defined.")

        
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


# @DATAMODULE_REGISTRY
class LitGNNDataModule(LightningDataModule):
    
    def __init__(self, batch_size: int) -> None:
        
        self.batch_size = batch_size
        super().__init__()
        
        
    def prepare_data(self) -> None:
        pass
    
    
    def setup(self, stage: str) -> None:
        dataset = GNNDataset().shuffle()
        length = len(dataset)
        
        self.test_dataset = dataset[int(length * .9):]
        self.val_dataset = dataset[int(length * .8):int(length * .9)]
        self.train_dataset = dataset[:int(length * .8)]
    
    
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