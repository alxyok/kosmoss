import numpy as np
import os.path as osp
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_optimizer as optim
import torchmetrics.functional as F
from typing import List, Union

from kosmoss import DATA_PATH

class CommonModule(LightningModule):

    def forward(self, x, edge_index):
        return self.net(x, edge_index)
    
    def _common_step(self, 
                     batch, 
                     batch_idx, 
                     stage: Union['train', 'val', 'test'] = "train") -> List[torch.Tensor]:
        
        y_hat = self(batch.x, batch.edge_index)
        loss = F.mean_squared_error(y_hat, batch.y)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        
        return y_hat, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, loss = self._common_step(batch, batch_idx, "val")
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        y_hat, _ = self._common_step(batch, batch_idx, "test")


# @MODEL_REGISTRY
class LitGAT(CommonModule):
    
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 heads,
                 lr):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.net = pyg.nn.GAT(in_channels=in_channels, 
                              hidden_channels=hidden_channels,
                              out_channels=out_channels,
                              num_layers=num_layers,
                              edge_dim=32,
                              dropout=dropout,
                              act=nn.SiLU(inplace=True),
                              heads=heads)
        
    
    def configure_optimizers(self):
        return optim.AdamP(self.parameters(), lr=self.lr)