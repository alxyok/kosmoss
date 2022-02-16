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
import psutil
import pytorch_lightning as pl
import torch
from typing import Union
import sys

def main(batch_size: int,
         lr: float,
         strategy: Union['ddp', 'horovod'],
         gpus: int,
         num_nodes: int) -> None:
    
    sys.path.append(osp.abspath('..'))

    import config
    import data
    import models

    pl.seed_everything(42, workers=True)
    
    step = config.config['timestep']
    params = config.params[str(step)]['flattened']

    x_feats = params['x_shape'][-1]
    y_feats = params['y_shape'][-1]

    mlp = models.LitMLP(
        in_channels=x_feats,
        hidden_channels=100,
        out_channels=y_feats,
        
        # Adjust the learning rate accordingly to account for the increase in total batch size
        # Or use a Lightning LR Finder functionality, or any other framework's finder
        lr=lr,
    )

    cores = psutil.cpu_count(logical=False)
    datamodule = data.FlattenedDataModule(
        
        # Adjust the total batch size with regards to the number of nodes and GPUs per node
        batch_size=batch_size // (num_nodes * gpus),
        num_workers=cores
    )

    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=config.logs_path,
        name='flattened_mlp_logs',
        log_graph=True
    )

    gpu_trainer = pl.Trainer(
        strategy=strategy,
        gpus=gpus,
        
        # If you're using a batch normalization layer
        # The following flag can allow sync accros total batch instead of local minibatch
        # sync_batchnorm=True,
        
        max_epochs=1,
        logger=logger,
        deterministic=True,
        num_sanity_val_steps=0,
        num_nodes=num_nodes
        
        # Uncomment if you want to use Nvidia's own implementation of AMP called APEX
        # amp_backend='apex',
        
        # Useful if you want to profile your training for debug purposes
        # profiler="simple"
    )
    gpu_trainer.fit(model=mlp, datamodule=datamodule)
    gpu_trainer.test(model=mlp, datamodule=datamodule)

    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', 
                        type=int,
                        help='Total batch size', 
                        default=16)
    parser.add_argument('--lr', 
                        type=float,
                        help='Learning Rate', 
                        default=1e-4)
    parser.add_argument('--strategy', 
                        type=str,
                        help='Data Distributed Strategy', 
                        default='ddp')
    parser.add_argument('--gpus', 
                        type=int,
                        help='Number of GPUs to accelerate over', 
                        default=1)
    parser.add_argument('--num-nodes', 
                        type=int,
                        help='Number of nodes to accelerate over', 
                        default=1)
    args = parser.parse_args()
    
    main(
        args.batch_size, 
        args.lr,
        args.strategy, 
        args.gpus,
        args.num_nodes
    )