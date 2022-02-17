import os.path as osp
import psutil
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from typing import Union

from kosmoss import CONFIG, LOGS_PATH, PARAMS
from kosmoss.parallel.data import FlattenedDataModule
from kosmoss.parallel.models import LitMLP

def main(batch_size: int,
         num_processes: int) -> None:

    seed_everything(42, workers=True)
    
    step = CONFIG['timestep']
    params = PARAMS[str(step)]['flattened']

    x_feats = params['x_shape'][-1]
    y_feats = params['y_shape'][-1]

    mlp = LitMLP(
        in_channels=x_feats,
        hidden_channels=100,
        out_channels=y_feats
    )

    cores = psutil.cpu_count(logical=False)
    datamodule = FlattenedDataModule(
        batch_size=batch_size // num_processes,
        num_workers=cores
    )

    logger = TensorBoardLogger(
        save_dir=LOGS_PATH,
        name='flattened_mlp_logs',
        log_graph=False
    )

    gpu_trainer = Trainer(
        max_epochs=1,
        logger=logger,
        deterministic=True,
        num_sanity_val_steps=0,
        num_processes=num_processes
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
    parser.add_argument('--num-processes', 
                        type=int,
                        help='Number of processes to distribute over', 
                        default=1)
    args = parser.parse_args()
    
    main(
        args.batch_size, 
        args.num_processes
    )
