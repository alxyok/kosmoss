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

from metaflow import FlowSpec, Parameter, step
import os
import os.path as osp
import time


class BuildGraphsFlow(FlowSpec):
        
    @step
    def start(self):
        """
        Create the constants for the rest of the Flow.
        """
        
        import numpy as np
        import yaml
        
        # Split the Flow and do slice_and_save for each subset
        step = 250
        self.shards_path = osp.join(config.processed_data_path, f'feats-{step}', 'concat')
        self.shard = np.arange(53 * 2 ** 3)
        self.next(self.slice_and_save, foreach="shard")
                  
                  
    @step
    def slice_and_save(self):
        """
        1) Read the raw data.
        2) Feature engineer x and y.
        3) Compute and save normalization factors for later use.
        3) Sequentially iterate the sharded subset to create and save the graph for each row.
        """
        
        from netCDF4 import Dataset
        import torch
        import torch.nn.functional as F
        import torch_geometric as pyg
        
        # Read the raw data file and extract the desired features
        in_path = osp.join(self.raw_data_path, f"data-{self.params['timestep']}.nc")
        np.memmap(
        
        # Compute normalization factors if dataset is complete
        # Serialize stats on disk
        self.normalization_time = '-'
        if self.params['num_shards'] == 1:
            stats_path = os.path.join(self.data_path, f"stats-{self.params['timestep']}.pt")
            if not os.path.isfile(stats_path) or self.params['force']:
                tic = time.perf_counter()
                stats = {
                    "x_mean" : torch.mean(x, dim=0),
                    "y_mean" : torch.mean(y, dim=0),
                    "x_std" : torch.std(x, dim=0),
                    "y_std" : torch.std(y, dim=0)
                }
                tac = time.perf_counter()
                torch.save(stats, stats_path)
                self.normalization_time = int(tac - tic)
            
        # Build a graph based on x, y, edge attribute features and connectivity matrix
        data_list = []
        for idx in range(x.shape[0]):
            x_ = torch.squeeze(x[idx, ...])
            y_ = torch.squeeze(y[idx, ...])

            edge_attr = torch.squeeze(sca_inputs_[idx, ...])

            data = pyg.data.Data(
                x=x_,
                edge_attr=edge_attr,
                edge_index=self.undirected_index,
                y=y_,
            )

            data_list.append(data)
        
        # Save the bulk of graph in a separate file
        out_path = osp.join(self.processed_data_path, f"data-{self.params['timestep']}.{self.input}.pt")
        torch.save(data_list, out_path)
        
        self.next(self.join)
        
        
    @step
    def join(self, inputs):
        """
        Join the parallel branches. Compile results, print execution time.
        """
        
        import json
        
        # Gather properties defined in previous steps
        self.merge_artifacts(inputs, include=[
            'start_time',
            'artifacts_path',
            'dataset_size',
            'normalization_time',
            'params'
        ])
        
        self.end_time = time.perf_counter()
        exec_time = int(self.end_time - self.start_time)
        
        print(f'Total execution time: ~{exec_time}s.')
        
        # Print execution time in a JSON file
        with open(osp.join(self.artifacts_path, 'results.json'), 'w') as file:
            json.dump({
                'max_workers': int(os.environ['MAX_WORKERS']),
                'dataset_size': self.dataset_size,
                'num_shards': self.params['num_shards'],
                'exec_time': exec_time,
                'normalization_time': self.normalization_time
            }, file)
        
        self.next(self.end)
        
        
    @step
    def end(self):
        """
        End the flow.
        """
        
        pass
    

    

if __name__ == '__main__':
    
    BuildGraphsFlow()