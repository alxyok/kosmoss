from pytorch_lightning import LightningModule
import torch
from torch_geometric.data import Batch
from torch_geometric.nn.models import GAT
import torch_optimizer as optim
import torchmetrics.functional as F
from typing import Dict, List, Union


class LitGAT(LightningModule):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = kwargs.pop('lr')
        self.net = GAT(**kwargs)

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        
        return self.net(x, edge_index)
    
    def _common_step(self, 
                     batch: Batch, 
                     batch_idx: int, 
                     stage: Union['train', 'val', 'test'] = "train") -> List[torch.Tensor]:
        
        y_hat = self(batch.x, batch.edge_index)
        loss = F.mean_squared_error(y_hat, batch.y)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        return y_hat, loss

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        _, loss = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        _, _ = self._common_step(batch, batch_idx, "val")
        # return {'val_loss': loss}

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        _, _ = self._common_step(batch, batch_idx, "test")
        
    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamP(self.parameters(), lr=self.lr)