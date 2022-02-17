from metaflow import FlowSpec, Parameter, step
import os
import os.path as osp
import sys
    
import kosmoss as km


class BuildGraphsFlow(FlowSpec):
        
    @step
    def start(self):
        """
        Create the constants for the rest of the Flow.
        """
        
        import numpy as np
        import yaml
        
        self.timestep = km.CONFIG['timestep']
        self.params = km.PARAMS[str(self.timestep)]['features']
        self.out_dir = km.utils.purgedirs(osp.join(km.PROCESSED_DATA_PATH, f"graphs-{self.timestep}"))
            
        self.shard = np.arange(km.PARAMS['num_shards'])
        self.next(self.build_graphs, foreach="shard")
                  
                  
    @step
    def build_graphs(self):
        """
        1) Load the raw data.
        2) Extract features for x and y.
        3) Sequentially iterate the sharded subset sample to create and save the graph for each row.
        """
        
        import numpy as np
        import torch
        import torch.nn.functional as F
        import torch_geometric as pyg
        from typing import Union
        
        main_dir = osp.join(km.PROCESSED_DATA_PATH, f"features-{self.timestep}")
        
        def load(name: Union['x', 'y', 'edge']) -> torch.Tensor:
            return torch.tensor(
                np.lib.format.open_memmap(
                    mode='r', 
                    dtype=self.params['dtype'], 
                    filename=osp.join(main_dir, name, f"{self.input}.npy"), 
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
            
        out_path = osp.join(self.out_dir, f"data-{self.timestep}.{self.input}.pt")
        torch.save(data_list, out_path)
        
        self.next(self.join)
        
        
    @step
    def join(self, inputs):
        """
        Join the parallel branches.
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