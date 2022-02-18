import numpy as np
import os.path as osp
from pytorch_lightning import LightningDataModule
import torch
from typing import Tuple, Union

from kosmoss import CONFIG, METADATA, PROCESSED_DATA_PATH

class FlattenedDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 step: int, 
                 mode: Union['efficient', 'controlled'] = 'controlled') -> None:
        super().__init__()
        self.step = step
        self.mode = mode
        self.params = METADATA[str(self.step)]['flattened']
    
    def __len__(self) -> int:
        
        return self.params['dataset_len']
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        
        shard_size = len(self) // self.params['num_shards']
        fileidx = idx // shard_size
        rowidx = idx % shard_size
        
        def _load(name: Union['x', 'y']) -> Tuple[torch.Tensor]:
            main_path = osp.join(PROCESSED_DATA_PATH, f"flattened-{self.step}")
            
            if self.mode == 'efficient':
                data = np.lib.format.open_memmap(
                    mode='r',
                    dtype = self.params['dtype'],
                    filename=osp.join(main_path, name, f'{fileidx}.npy'), 
                    shape=tuple(self.params[f'{name}_shape']) 
                )
                
            else:
                data = np.load(osp.join(main_path, name, f'{fileidx}.npy'))
                
            tensor = torch.squeeze(torch.tensor(data[rowidx, ...]))
            return tensor
        
        x = _load('x')
        y = _load('y')
        
        return x, y
    

class FlattenedDataModule(LightningDataModule):
    
    def __init__(self, 
                 batch_size: int,
                 num_workers: int) -> None:
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loading_mode = CONFIG['loading_mode']
        self.timestep = CONFIG['timestep']
        super().__init__()
        
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: str) -> None:
        dataset = FlattenedDataset(self.timestep, mode=self.loading_mode)
        length = len(dataset)
        
        self.train, self.val, self.test = torch.utils.data.random_split(
            dataset,
            [
                int(length * .8), 
                int(length * .1), 
                int(length * .1)
            ])
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        
        return torch.utils.data.DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers)
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        
        return torch.utils.data.DataLoader(
            self.val, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers)
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        
        return torch.utils.data.DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers)