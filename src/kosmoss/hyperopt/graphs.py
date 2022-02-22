import numpy as np
import os.path as osp
from pytorch_lightning import LightningDataModule
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from typing import List, Tuple

from kosmoss import CONFIG, DATA_PATH, METADATA
from kosmoss.dataproc.flows import BuildGraphsFlow


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
        
    def len(self) -> int:
        return self.params['dataset_len']

    def get(self, idx: int) -> Tuple[torch.Tensor]:
        
        shard_size = self.len() // self.num_shards
        fileidx = idx // shard_size
        rowidx = idx % shard_size
        
        data_list = torch.load(osp.join(self.processed_dir, f"graphs-{self.timestep}", f'data-{fileidx}.pt'))
        data = data_list[rowidx]
        
        return data


class LitGNNDataModule(LightningDataModule):
    
    def __init__(self, batch_size: int) -> None:
        self.bs = batch_size
        super().__init__()
        
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: str) -> None:
        dataset = GNNDataset().shuffle()
        length = len(dataset)
        
        self.testds = dataset[int(length * .9):]
        self.valds = dataset[int(length * .8):int(length * .9)]
        self.trainds = dataset[:int(length * .8)]
    
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainds, batch_size=self.bs, num_workers=4, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valds, batch_size=self.bs, num_workers=4)
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.testds, batch_size=self.bs, num_workers=4)
