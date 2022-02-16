# MIT License
# 
# Copyright (c) 2022 alxyok
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import os.path as osp
import pytorch_lightning as pl
import torch
from typing import Tuple, Union
import sys

sys.path.append(osp.abspath('..'))

import config

class FlattenedDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 step: int, 
                 mode: Union['efficient', 'controlled'] = 'controlled') -> None:
        super().__init__()
        self.step = step
        self.mode = mode
        self.params = config.params[str(self.step)]['flattened']
    
    def __len__(self) -> int:
        
        return self.params['dataset_len']
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        
        shard_size = len(self) // self.params['num_shards']
        fileidx = idx // shard_size
        rowidx = idx % shard_size
        
        def _load(name: Union['x', 'y']) -> Tuple[torch.Tensor]:
            main_path = osp.join(config.processed_data_path, f"flattened-{self.step}")
            
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
    

class FlattenedDataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 batch_size: int,
                 num_workers: int) -> None:
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loading_mode = config.config['loading_mode']
        self.timestep = config.config['timestep']
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