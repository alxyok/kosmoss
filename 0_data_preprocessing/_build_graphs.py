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
        self.next(self.build_graphs, foreach="shard")
                  
                  
    @step
    def build_graphs(self):
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
        main_dir = osp.join(config.processed_data_path, f"features-{self.params['timestep']}")
        
        def load(name):
            return torch.tensor(
                np.memmap(
                    osp.join(main_dir, name, f"{self.input}.npy"), 
                    dtype=self.params['dtype'], 
                    mode='r', 
                    shape=tuple(self.params[f'{name}_shape'])))
                
        x, y, edge = load("x"), load("y"), load("edge")
        
        data_list = []
        
        directed_idx = np.array([[*range(1, 138)], [*range(137)]])
        undirected_idx = np.hstack((
            directed_idx, 
            directed_idx[[1, 0], :]
        ))
        undirected_idx = torch.tensor(undirected_idx, dtype=torch.long)
        for idx in range(len(x)):
            x_ = torch.squeeze(x[idx, ...])
            y_ = torch.squeeze(y[idx, ...])
            edge_ = torch.squeeze(edge[idx, ...])

            data = pyg.data.Data(x=x_, edge_attr=edge_, edge_index=undirected_idx, y=y_,)

            data_list.append(data)
            
        out_path = osp.join(
            config.processed_data_path, 
            f"graphs-{self.params['timestep']}",
            f"data-{self.params['timestep']}.{self.input}.pt")
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