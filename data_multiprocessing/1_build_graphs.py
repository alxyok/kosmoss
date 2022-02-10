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
    
    num_shards = Parameter(
        'num_shards',
        help="Desired number of shards.",
        default=2 * 53)
    
    timestep = Parameter(
        'timestep',
        help="Dataset timestep.",
        default=1000)
        
    @step
    def start(self):
        """
        Create the constants for the rest of the Flow.
        """
        
        import numpy as np
        import randomname
        import torch
        
        self.start_time = time.perf_counter()

        root_path = osp.dirname(osp.realpath(__file__))
        self.data_path = osp.join(root_path, 'data')
        self.raw_data_path = osp.join(self.data_path, 'raw')
        self.processed_data_path = osp.join(self.data_path, 'processed')
        xps_path = osp.join(root_path, 'experiments')

        # Create all path for the current experiment
        os.makedirs(xps_path, exist_ok=True)
        existing_xps = os.listdir(xps_path)

        # Generate experiment name
        _randomize = True
        while _randomize:
            name = randomname.get_name()
            if name not in xps_path: _randomize = False
        xp_path = osp.join(xps_path, name)
        self.artifacts_path = osp.join(xp_path, 'artifacts')
        
        # Make directories that do not exist
        for p in [self.processed_data_path, xp_path, self.artifacts_path]:
            os.makedirs(p, exist_ok=True)

        # Build the graph connectivity matrix
        _directed_index = np.array([[*range(1, 138)], [*range(137)]])
        _undirected_index = np.hstack((
            _directed_index, 
            _directed_index[[1, 0], :]
        ))
        self.undirected_index = torch.tensor(_undirected_index, dtype=torch.long)
        
        # Split the Flow and do slice_and_save for each subset
        self.shard = np.arange(self.num_shards)
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

        # Feature engineering, node feature alignment
        def broadcast_features(tensor):
            t = torch.unsqueeze(tensor, -1)
            t = t.repeat((1, 1, 138))
            t = t.moveaxis(1, -1)
            return t
        def pad_tensor(tensor):
            return F.pad(tensor, (0, 0, 1, 1, 0, 0))

        # Calculate slice start and end indices
        def slice_indices(dataset_size, num_shards, idx=0):
            rows_per_shard = dataset_size // num_shards
            start = idx * rows_per_shard
            end = start + rows_per_shard
            return start, end
        
        # Read the raw data file and extract the desired features
        in_path = osp.join(self.raw_data_path, f'data-{self.timestep}.nc')
        with Dataset(in_path, "r", format="NETCDF4") as file:
            self.dataset_size = file.dimensions['column'].size
            start, end = slice_indices(self.dataset_size, self.num_shards, self.input)
            
            sca_inputs = torch.tensor(file['sca_inputs'][start:end])
            col_inputs = torch.tensor(file['col_inputs'][start:end])
            hl_inputs = torch.tensor(file['hl_inputs'][start:end])
            inter_inputs = torch.tensor(file['inter_inputs'][start:end])

            flux_dn_sw = torch.tensor(file['flux_dn_sw'][start:end])
            flux_up_sw = torch.tensor(file['flux_up_sw'][start:end])
            flux_dn_lw = torch.tensor(file['flux_dn_lw'][start:end])
            flux_up_lw = torch.tensor(file['flux_up_lw'][start:end])

        inter_inputs_ = pad_tensor(inter_inputs)
        sca_inputs_ = broadcast_features(sca_inputs)

        # Feature engineering, build an input x with 20 features
        x = torch.cat([
            hl_inputs,
            inter_inputs_,
            sca_inputs_
        ], dim=-1)

        # Feature engineering, build ground truth with 4 features
        y = torch.cat([
            torch.unsqueeze(flux_dn_sw, -1),
            torch.unsqueeze(flux_up_sw, -1),
            torch.unsqueeze(flux_dn_lw, -1),
            torch.unsqueeze(flux_up_lw, -1),
        ], dim=-1)
        
        # Compute normalization factors if dataset is complete
        # Serialize stats on disk
        if self.num_shards == 1:
            stats_path = os.path.join(self.data_path, f"stats-{self.timestep}.pt")
            if not os.path.isfile(stats_path):
                stats = {
                    "x_mean" : torch.mean(x, dim=0),
                    "y_mean" : torch.mean(y, dim=0),
                    "x_std" : torch.std(x, dim=0),
                    "y_std" : torch.std(y, dim=0)
                }
                torch.save(stats, stats_path)
            
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
        out_path = osp.join(self.processed_data_path, f'data-{self.timestep}.{self.input}.pt')
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
            'dataset_size'
        ])
        
        self.end_time = time.perf_counter()
        exec_time = int(self.end_time - self.start_time)
        
        print(f'Total execution time: ~{exec_time}s.')
        
        # Print execution time in a JSON file
        with open(osp.join(self.artifacts_path, 'results.json'), 'w') as file:
            json.dump({
                'max_workers': int(os.environ['MAX_WORKERS']),
                'dataset_size': self.dataset_size,
                'num_shards': self.num_shards,
                'exec_time': exec_time
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