import json
import os.path as osp
from pytorch_lightning import LightningDataModule, LightningModule, Trainer as T
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, MODEL_REGISTRY, LightningCLI
import torch
from typing import List, Union

from kosmoss import ARTIFACTS_PATH, LOGS_PATH
from kosmoss.hyperopt import data, models
from kosmoss.hyperopt.data import LitGNNDataModule
from kosmoss.hyperopt.models import LitGAT

class Trainer(T):
    
    def __init__(self, 
                 max_epochs: int, 
                 gpus: Union[int, str, List[int], None],
                 fast_dev_run: Union[int, bool] = False, 
                 callbacks: Union[List[Callback], Callback, None] = None) -> None:
            
        logger = TensorBoardLogger(LOGS_PATH, name=None)
        super().__init__(
            default_root_dir=LOGS_PATH,
            logger=logger,
            gpus=gpus,
            max_epochs=max_epochs,
            # TODO: for some reason, a forward pass happens in the model before datamodule creation.
            num_sanity_val_steps=0)
    
    def test(self, **kwargs) -> None:
        results = super().test(**kwargs)[0]
        
        with open(osp.join(ARTIFACTS_PATH, "results.json"), "w") as f:
            json.dump(results, f)
        
        torch.save(self.model.net, osp.join(ARTIFACTS_PATH, 'model.pth'))
        
        
def main():
    
    DATAMODULE_REGISTRY.register_classes(data, LitGNNDataModule, override=True)
    MODEL_REGISTRY.register_classes(models, LitGAT, override=True)
    
    cli = LightningCLI(trainer_class=Trainer)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)
    
    
if __name__ == '__main__':
    
    main()