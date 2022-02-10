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

import h5py
import numpy as np
import os
import os.path as osp
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch_optimizer as optim
import torchmetrics.functional as F

import config

class NOGWDModule(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        
        stats = torch.load(osp.join(config.data_path, 'stats.pt'))
        
        self.x_mean = stats['x_mean'].to(self.device)
        self.x_std = stats['x_std'].to(self.device)
        self.y_std = stats['y_std'].max().to(self.device)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def _common_step(self, batch: torch.Tensor, batch_idx: int, stage: str):
        
        x, y = batch
        x = (x - self.x_mean.to(self.device)) / self.x_std.to(self.device)
        y = y / self.y_std.to(self.device)
        y_hat = self(x)
        
        loss = F.mean_squared_error(y_hat, y)
        # TODO: Add epsilon in TSS for R2 score computation.
        # Currently returns NaN sometimes.
        r2 = F.r2_score(y_hat, y)
    
        self.log(f"{stage}_loss", loss, on_step=True, prog_bar=True)
        self.log(f"{stage}_r2", self.r2, on_step=True, prog_bar=True)

        return y_hat, loss, r2
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        _, loss, _ = self._common_step(batch, batch_idx, "train")
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        _, loss, _ = self._common_step(batch, batch_idx, "val")
        
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        _, loss, _ = self._common_step(batch, batch_idx, "test")
        
    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)


@MODEL_REGISTRY
class LitMLP(NOGWDModule):
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, lr: float):
        super().__init__()
        
        self.lr = lr
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )