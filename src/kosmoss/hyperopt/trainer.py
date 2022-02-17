import json
import os.path as osp
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import LightningCLI
import torch
from typing import List, Union

from kosmoss import ARTIFACTS_PATH, LOGS_PATH

class Trainer(Trainer):
    
    def __init__(self, 
                 accelerator: Union[str, Accelerator, None], 
                 devices: Union[List[int], str, int, None], 
                 max_epochs: int, 
                 fast_dev_run: Union[int, bool] = False, 
                 callbacks: Union[List[Callback], Callback, None] = None) -> None:
        
        if accelerator == 'cpu': 
            devices = None
            
        logger = TensorBoardLogger(LOGS_PATH, name=None)
        super().__init__(
            default_root_dir=LOGS_PATH,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            # TODO: for some reason, a forward pass happens in the model before datamodule creation.
            num_sanity_val_steps=0)
    
    def test(self, **kwargs) -> None:
        results = super().test(**kwargs)[0]
        
        with open(osp.join(ARTIFACTS_PATH, "results.json"), "w") as f:
            json.dump(results, f)
        
        torch.save(self.model.net, osp.join(ARTIFACTS_PATH, 'model.pth'))
        
        
def main():
    
    cli = LightningCLI(trainer_class=Trainer)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)
    
    
if __name__ == '__main__':
    
    main()