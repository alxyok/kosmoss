from metaflow import FlowSpec, Parameter, step
import os
import os.path as osp
import shutil

class BuildGraphsFlow(FlowSpec):
    
    # In addition to the standard class properties...
    PROCESSED_DATA_PATH = osp.join(os.environ['HOME'], ".kosmoss", "data", "processed")

    # ...you can just add parameters to be read from the command line
    timestep = Parameter('timestep', help='Temporal sampling step', default=1000)
    num_shards = Parameter('num_shards', help='Number of shards', default=848)
    dtype = Parameter('dtype', help="NumPy's dtype", default='float32')
    x_shape = Parameter('x_shape', help='Shape for x', default=(1280, 136, 17))
    y_shape = Parameter('y_shape', help='Shape for y', default=(1280, 138, 1))
    edge_shape = Parameter('edge_shape', help='Shape for edge', default=(1280, 137, 27))
        
    @step
    def start(self):
        """
        Create the constants for the rest of the Flow.
        """
        
        import numpy as np
        
        # Each 'common' step can store a shared property
        self.out_dir = osp.join(self.PROCESSED_DATA_PATH, f"graphs-{self.timestep}")
        if osp.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
            os.makedirs(self.out_dir)
        
        # To launch in thread in parallel, just call the next step over an attribute's list
        self.shard = np.arange(self.num_shards)
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
        
        main_dir = osp.join(self.PROCESSED_DATA_PATH, f"features-{self.timestep}")
        
        def load(name: Union['x', 'y', 'edge']) -> torch.Tensor:
            return torch.tensor(
                np.lib.format.open_memmap(
                    mode='r', 
                    dtype=self.dtype, 
                    filename=osp.join(main_dir, name, f"{self.input}.npy"), 
                    shape=getattr(self, f'{name}_shape')))
                
        x, y, edge = load("x"), load("y"), load("edge")
        
        data_list = []
        
        # Build the both-ways connectivity matrix
        directed_idx = np.array([[*range(1, 138)], [*range(137)]])
        undirected_idx = np.hstack((
            directed_idx, 
            directed_idx[[1, 0], :]
        ))
        undirected_idx = torch.tensor(undirected_idx, dtype=torch.long)
        
        # Iterate over the rows of the sharded file
        for idx in range(len(x)):
            
            # For each element, simply extract:
            # The nodes features (input x, output y)
            x_ = torch.squeeze(x[idx, ...])
            y_ = torch.squeeze(y[idx, ...])
            
            # The edge attributes
            edge_ = torch.squeeze(edge[idx, ...])

            # Build a graph for that element
            data = pyg.data.Data(x=x_, edge_attr=edge_, edge_index=undirected_idx, y=y_,)
            
            # Append the data to a list
            data_list.append(data)
            
        # Save the list with torch.save()
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