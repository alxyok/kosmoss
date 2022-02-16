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

import json
import os.path as osp
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.cli import LightningCLI
import sys
import torch
from typing import List, Union

import kosmoss as km

class Trainer(pl.Trainer):
    
    def __init__(self, 
                 accelerator: Union[str, pl.accelerators.Accelerator, None], 
                 devices: Union[List[int], str, int, None], 
                 max_epochs: int, 
                 fast_dev_run: Union[int, bool] = False, 
                 callbacks: Union[List[Callback], Callback, None] = None) -> None:
        
        if accelerator == 'cpu': 
            devices = None
            
        logger = pl.loggers.TensorBoardLogger(km.LOGS_PATH, name=None)
        super().__init__(
            default_root_dir=km.LOGS_PATH,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            # TODO: for some reason, a forward pass happens in the model before datamodule creation.
            num_sanity_val_steps=0)
    
    def test(self, **kwargs) -> None:
        results = super().test(**kwargs)[0]
        
        with open(osp.join(km.ARTIFACTS_PATH, "results.json"), "w") as f:
            json.dump(results, f)
        
        torch.save(self.model.net, osp.join(km.ARTIFACTS_PATH, 'model.pth'))
        
        
def main():
    
    cli = LightningCLI(trainer_class=Trainer)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)
    
    
if __name__ == '__main__':
    
    main()