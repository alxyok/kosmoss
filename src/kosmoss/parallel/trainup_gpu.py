import os.path as osp
import psutil
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from typing import Union

from kosmoss import CONFIG, LOGS_PATH, PARAMS
from kosmoss.parallel.data import FlattenedDataModule
from kosmoss.parallel.models import LitMLP

def main(batch_size: int,
         lr: float,
         strategy: Union['ddp', 'horovod'],
         gpus: int,
         num_nodes: int) -> None:

    seed_everything(42, workers=True)
    
    step = CONFIG['timestep']
    params = PARAMS[str(step)]['flattened']

    x_feats = params['x_shape'][-1]
    y_feats = params['y_shape'][-1]

    mlp = LitMLP(
        in_channels=x_feats,
        hidden_channels=100,
        out_channels=y_feats,
        
        # Adjust the learning rate accordingly to account for the increase in total batch size
        # Or use a Lightning LR Finder functionality, or any other framework's finder
        lr=lr,
    )

    cores = psutil.cpu_count(logical=False)
    datamodule = FlattenedDataModule(
        
        # Adjust the total batch size with regards to the number of nodes and GPUs per node
        batch_size=batch_size // (num_nodes * gpus),
        num_workers=cores
    )

    logger = TensorBoardLogger(
        save_dir=LOGS_PATH,
        name='flattened_mlp_logs',
        log_graph=True
    )

    gpu_trainer = Trainer(
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