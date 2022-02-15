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

import os.path as osp
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_optimizer as optim
import torchmetrics.functional as F
from typing import List, Tuple, Union
import sys

sys.path.append(osp.abspath('..'))

import config

class ThreeDCorrectionModule(pl.LightningModule):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def _common_step(self, 
                     batch: List[torch.Tensor], 
                     batch_idx: int, 
                     stage: Union['train', 'test', 'val']) -> Tuple[torch.Tensor]:
        
        x, y = batch
        y_hat = self(x)
        
        # Note: by using a metric from TorchMetrics, you don't have to manually set the sync_dist option, 'ensures that each GPU worker has the same behaviour when tracking model checkpoints, which is important for later downstream tasks such as testing the best checkpoint across all workers'.
        loss = F.mean_squared_error(y_hat, y)
        self.log(
            f"{stage}_loss", 
            loss, prog_bar=True, 
            on_step=True, 
            batch_size=len(batch),
            # Uncomment, for the 'test' and 'val' stages if you're not using TorchMetrics
            # sync_dist = True
        )
        
        return y_hat, loss
    
    def training_step(self, 
                      batch: List[torch.Tensor], 
                      batch_idx: int) -> torch.Tensor:
        
        _, loss = self._common_step(batch, batch_idx, "train")
        
        return loss

    def validation_step(self, 
                        batch: List[torch.Tensor], 
                        batch_idx: int) -> Tuple[torch.Tensor]:
        
        y_hat, _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, 
                  batch: List[torch.Tensor], 
                  batch_idx: int) -> Tuple[torch.Tensor]:
        y_hat, _ = self._common_step(batch, batch_idx, "test")
    
    def configure_optimizers(self) -> optim.Optimizer:
        
        return optim.AdamP(self.parameters(), lr=self.lr)
    
    
class LitMLP(ThreeDCorrectionModule):
    
    class Normalize(nn.Module):
        
        def __init__(self, module_instance) -> None:
            
            super().__init__()
            
            # 'The LightningModule knows what device it is on. You can access the reference via self.device. Sometimes it is necessary to store tensors as module attributes. However, if they are not parameters they will remain on the CPU even if the module gets moved to a new device. To prevent that and remain device agnostic, register the tensor as a buffer in your modules’s __init__ method with register_buffer().'
            self._module = module_instance
            self._module.register_buffer("epsilon", torch.tensor(1.e-8))
            
            step = config.config['timestep']
            stats = torch.load(osp.join(config.data_path, f"stats-flattened-{step}.pt"))
            
            self.x_mean = stats["x_mean"]
            self.x_std = stats["x_std"]
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            
            # Adjusting the tensors to scale to other devices. 'When you need to create a new tensor, use type_as. This will make your code scale to any arbitrary number of GPUs or TPUs with Lightning.'
            mean = self.x_mean.type_as(x)
            std = self.x_std.type_as(x)

            return (x - mean) / (std + self._module.epsilon)
    
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 lr: float = 1e-4) -> None:
        
        super().__init__()
        
        self.lr = lr
        
        self.net = nn.Sequential(
            LitMLP.Normalize(self),
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )