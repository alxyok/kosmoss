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

import config


class BuildGraphsFlow(FlowSpec):
    
    
    # myfile = IncludeFile(
    #     'myfile',
    #     is_text=False,
    #     help='My input',
    #     default='/Users/bob/myinput.bin')
        
    @step
    def start(self):
        """
        Create the constants for the rest of the Flow.
        """
        
        import numpy as np
        import json
        
        # Split the Flow and do slice_and_save for each subset
        self.shards_path = osp.join(config.processed_data_path, f'feats-{step}', 'concat')
        
        with open(osp.join(config.root_path, "params.json"), "r") as stream:
            self.params = json.load(stream)
            
        self.shard = np.arange(self.params['num_shards'])
        self.next(self.slice_and_save, foreach="shard")
                  
                  
    @step
    def slice_and_save(self):
        """
        1) Read the raw data.
        2) Feature engineer x and y.
        3) Compute and save normalization factors for later use.
        3) Sequentially iterate the sharded subset to create and save the graph for each row.
        """
        
        import numpy as np
        import torch
        import torch.nn.functional as F
        import torch_geometric as pyg
        
        # Read the raw data file and extract the desired features
        filepath = osp.join(config.processed_data_path, f"feats-{self.params['timestep']}", "concat", f"{self.input}.npy")
        
        data = np.memmap(
            filepath, 
            dtype=self.params['dtype'],
            mode='r',
            shape=tuple(self.params['shard_shape']))
        
        x = torch.tensor(data[..., :20])
        y = torch.tensor(data[..., 20:])
        
        data_list = []
        
        directed_index = np.array([[*range(1, 138)], [*range(137)]])
        undirected_index = np.hstack((
            directed_index, 
            directed_index[[1, 0], :]
        ))
        undirected_index = torch.tensor(undirected_index, dtype=torch.long)
        for idx in range(len(data)):
            x_ = torch.squeeze(x[idx, ...])
            y_ = torch.squeeze(y[idx, ...])

            # edge_attr = torch.squeeze(sca_inputs_[idx, ...])

            data = pyg.data.Data(
                x=x_,
                # edge_attr=edge_attr,
                edge_index=undirected_index,
                y=y_,
            )

            data_list.append(data)
        
        # Save the bulk of graph in a separate file
        out_path = osp.join(config.processed_data_path, f"data-{self.params['timestep']}.{self.input}.pt")
        torch.save(data_list, out_path)
        
        self.next(self.join)
        
        
    @step
    def join(self, inputs):
        """
        Join the parallel branches. Compile results, print execution time.
        """
        
        self.next(self.end)
        
        
    @step
    def end(self):
        """
        End the flow.
        """
        
        pass
    

    

if __name__ == '__main__':
    
    BuildGraphsFlow()