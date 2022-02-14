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
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_optimizer as optim
import torchmetrics.functional as F
import os

import config

class ThreeDCorrectionModule(pl.LightningModule):

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def _common_step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x, edge_index)
        loss = F.mean_squared_error(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=len(batch))
        
        return y_hat, loss
    
    def training_step(self, batch, batch_idx):
        _, loss = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        y_hat, _ = self._common_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)


@MODEL_REGISTRY
class LitMLP(ThreeDCorrectionModule):
    
    class Normalize(nn.Module):
        
        def __init__(self):
            super().__init__()
            self.epsilon = torch.tensor(1.e-8)
            
            stats = torch.load(os.path.join(config.data_path, f"stats-{config.params['timestep']}.pt"))
            
            self.x_mean = stats["x_mean"]
            self.x_std = stats["x_std"]
            
        def forward(self, x: torch.Tensor):

            return (x - self.x_mean) / (self.x_std + self.epsilon)
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 lr: float):
        super().__init__()
        
        self.lr = lr
        self.net = nn.Sequential(
            self.Normalize(),
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )