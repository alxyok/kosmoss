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

import kosmoss as km

class ThreeDCorrectionModule(pl.LightningModule):

    def forward(self, x, edge_index):
        return self.net(x, edge_index)
        
    def _normalize(self, batch, batch_size):
        stats = torch.load(os.path.join(km.DATA_PATH, f"stats-{self.step}.pt"))
        device = self.device
        x_mean = stats["x_mean"].to(device)
        y_mean = stats["y_mean"].to(device)
        x_std = stats["x_std"].to(device)
        y_std = stats["y_std"].to(device)
        
        num_output_features = batch.y.size()[-1]
        
        x = (batch.x.reshape((batch_size, -1, batch.num_features)) - x_mean) / (x_std + torch.tensor(1.e-8))
        y = (batch.y.reshape((batch_size, -1, num_output_features)) - y_mean) / (y_std + torch.tensor(1.e-8))
        
        x = x.reshape(-1, batch.num_features)
        y = y.reshape(-1, num_output_features)
        
        return x, batch.y, batch.edge_index
    
    def _common_step(self, 
                     batch, 
                     batch_idx, 
                     stage: Union['train', 'val', 'test'] = "train") -> List[torch.Tensor]:
        batch_size = batch.ptr.size()[0]-1
        
        if normalize:
            x, y, edge_index = self._normalize(batch, batch_size)
        else:
            x, y, edge_index = batch.x, batch.y, batch.edge_index
        
        y_hat = self(x, edge_index)
        loss = F.mean_squared_error(y_hat, y)
        r2 = F.r2_score(y_hat, y)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, batch_size=batch_size)
        self.log(f"{stage}_r2", r2,  prog_bar=True, on_step=True, batch_size=batch_size)
        
        return y_hat, loss, r2

    def training_step(self, batch, batch_idx):
        _, loss, _ = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, _, _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        y_hat, _, _ = self._common_step(batch, batch_idx, "test")


@MODEL_REGISTRY
class LitGAT(ThreeDCorrectionModule):
    
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 heads,
                 jk,
                 lr,
                 timestep,
                 norm):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.step = timestep
        self.norm = norm
        self.net = pyg.nn.GAT(in_channels=in_channels, 
                              hidden_channels=hidden_channels,
                              out_channels=out_channels,
                              num_layers=num_layers,
                              dropout=dropout,
                              act=nn.SiLU(inplace=True),
                              heads=heads,
                              jk=jk)
        
    
    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)