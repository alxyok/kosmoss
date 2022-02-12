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

import h5py
from memory_profiler import profile
import networkx as nx
import numpy as np
import os
import os.path as osp
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_optimizer as optim
import yaml
import netCDF4
from tqdm import tqdm

import config
from build_graphs import BuildGraphsFlow

class GraphDataset(torch.utils.data.Dataset):
    
    def __init__(self, mode="train"):
        super().__init__()
        
        directed_index = np.array([[*range(1, 138)], [*range(137)]])
        undirected_index = np.hstack((
            directed_index, 
            directed_index[[1, 0], :]
        ))
        self.undirected_index = torch.tensor(undirected_index, dtype=torch.long)
    
    def raw_file_names(self):
        return None
    
    def processed_file_names(self):
        return [f"data-{config.params['timestep']}.{k}.pt" 
                for k in np.arange(config.params['num_shards'])]
    
    def download(self):
        pass
    
    def process(self):
        BuildGraphsFlow()

class GraphDataset(pyg.data.Dataset):
    
    def __init__(self, mode="train"):
        super().__init__()
        
        directed_index = np.array([[*range(1, 138)], [*range(137)]])
        undirected_index = np.hstack((
            directed_index, 
            directed_index[[1, 0], :]
        ))
        self.undirected_index = torch.tensor(undirected_index, dtype=torch.long)
    
    def __len__(self):
        return config.params['dataset_len']
    
    def __getitem__(self, idx):
        fileidx = idx // config.params['num_shards']
        rowidx = idx % config.params['num_shards']
        
        path = osp.join(config.processed_data_path, f"feats-{config.params['timestep']}", f'{fileidx}.npy')
        feats = np.memmap(
            path, 
            dtype = config.params['dtype'],
            mode='r',
            shape=config.params['shard_shape']
        )
        
        x = torch.squeeze(torch.tensor(feats[rowidx, :, :20]))
        y = torch.squeeze(torch.tensor(feats[rowidx, :, 20:]))

        graph = pyg.data.Data(x=x, edge_index=self.undirected_index, y=y)
        
        return graph

# class RawDataset(torch.utils.data.Dataset):
    
#     def __len__(self):
#         return config.params['dataset_len']
    
#     def __getitem__(self, idx):
#         fileidx = idx // config.params['num_shards']
#         rowidx = idx % config.params['num_shards']
        
#         path = osp.join(config.processed_data_path, f'feats-{config.params['timestep']}', f'{fileidx}.npy')
#         feats = np.memmap(
#             path, 
#             dtype = config.params['dtype'],
#             mode='r',
#             shape=config.params['shape']
#         )
        
#         x = torch.tensor(feats[rowidx, :, :20])
#         y = torch.tensor(feats[rowidx, :, 20:])
        
#         return x, y


@DATAMODULE_REGISTRY
class LitThreeDCorrectionDataModule(pl.LightningDataModule):
    """Creates datasets for the """
    
    def __init__(self, 
                 timestep: int,
                 batch_size: int,
                 num_workers: int):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__()
        
    def prepare_data(self):
        GraphDataset()
    
    def setup(self, stage):
        dataset = GraphDataset()#.shuffle()
        length = len(dataset)
        
        # self.test_dataset = GraphDataset("test")
        # self.val_dataset = GraphDataset("val")
        # self.train_dataset = GraphDataset("train")
        
        # self.test_dataset = dataset[int(length * .9):]
        # self.val_dataset = dataset[int(length * .8):int(length * .9)]
        # self.train_dataset = dataset[:int(length * .8)]
        self.train, self.val, self.test = torch.utils.data.random_split(
            dataset,
            [int(length * .8), int(length * .1), int(length * .1)])
    
    def train_dataloader(self):
        return pyg.loader.DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers)
    
    def val_dataloader(self):
        return pyg.loader.DataLoader(
            self.val, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers)
    
    def test_dataloader(self):
        return pyg.loader.DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers)