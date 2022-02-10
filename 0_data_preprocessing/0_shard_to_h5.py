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


class ShardToH5Flow(FlowSpec):
    
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

        root_path = osp.join(osp.dirname(osp.realpath(__file__)), '..')
        self.data_path = osp.join(root_path, 'data')
        self.raw_data_path = osp.join(self.data_path, 'raw')
        self.processed_data_path = osp.join(self.data_path, 'processed', 'shards')
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
        
        # Split the Flow and do slice_and_save for each subset
        self.shard = np.arange(self.num_shards)
        self.next(self.slice_and_save, foreach="shard")
                  
                  
    @step
    def slice_and_save(self):
        """
        1) Read a subset of the raw data.
        2) Write the content to an HDF5 file. 
        """
        
        from h5py import File
        from netCDF4 import Dataset
        import numpy as np
        import torch
        import torch.nn.functional as F
        import torch_geometric as pyg

        # Calculate slice start and end indices
        def slice_indices(dataset_size, num_shards, idx=0):
            rows_per_shard = dataset_size // num_shards
            start = idx * rows_per_shard
            end = start + rows_per_shard
            return start, end
        
        # Read the raw data file and extract the desired features
        in_path = osp.join(self.raw_data_path, f'data-{self.timestep}.nc')
        out_path = osp.join(self.processed_data_path, f'data-{self.timestep}.{self.input}.h5')
        # if osp.exists(out_path): os.remove(out_path)
        
        variable_names = [
            'sca_inputs',
            'col_inputs',
            'hl_inputs',
            'inter_inputs',
            'flux_dn_sw',
            'flux_up_sw',
            'flux_dn_lw',
            'flux_up_lw',
        ]
        with Dataset(in_path, "r", format="NETCDF4") as in_file:
            with File(out_path, "w") as out_file:
                self.dataset_size = in_file.dimensions['column'].size
                start, end = slice_indices(self.dataset_size, self.num_shards, self.input)

                for name in variable_names:
                    data = np.squeeze(in_file[name][start:end])
                    out_file.create_dataset(name, data=data)
        
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
    
    ShardToH5Flow()