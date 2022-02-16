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
import os
import os.path as osp
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import sys
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_optimizer as optim

sys.path.append(osp.abspath('..'))

import config

class ThreeDCorrectionDataset(pyg.data.InMemoryDataset):
    
    def __init__(self, root: str) -> None:
        
        self.timestep = config.config['timestep']
        self.params = config.params[str(self.timestep)]['features']
        self.num_shards = self.params['num_shards']
        
        super().__init__(root)
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> None:
        pass
        # features_path = osp.join(config.processed_data_path, f"features-{self.timestep}")
        # x_file = osp.join(features_path, 'x')
        # return [osp.join(features_path, 'x', f'{shard}.npy') for shard in 

    @property
    def processed_file_names(self) -> List[str]:
        return [f"data-{self.timestep}.{shard}.pt" for shard in np.arange(self.num_shards)]
    
    
    def download(self) -> None:
        raise Exception("Execute the Notebooks in this Bootcamp following natural order.")

        
    def process(self) -> None:
        
        ßimport flows
        flows.BuildGraphsFlow()


@DATAMODULE_REGISTRY
class LitThreeDCorrectionDataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 batch_size: int,
                 num_workers: int) -> None:
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()
        
        
    def prepare_data(self) -> None:
        pass
    
    
    def setup(self, stage: str) -> None:
        dataset = ThreeDCorrectionDataset(config.data_path).shuffle()
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