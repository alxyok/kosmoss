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
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import sys
import torch
import torch_geometric as pyg

import kosmoss as km


class ThreeDCorrectionDataset(pyg.data.InMemoryDataset):
    
    def __init__(self, root: str) -> None:
        
        self.timestep = km.CONFIG['timestep']
        self.params = km.PARAMS[str(self.timestep)]['features']
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
        
        km.dataproc.flows.BuildGraphsFlow()


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
        dataset = ThreeDCorrectionDataset(km.DATA_PATH).shuffle()
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