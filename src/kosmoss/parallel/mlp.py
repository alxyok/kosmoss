import os.path as osp
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch_optimizer as optim
import torchmetrics.functional as F
from typing import List, Tuple, Union

from kosmoss import CONFIG, DATA_PATH

class ThreeDCorrectionModule(LightningModule):
    
    def __init__(self):
        super().__init__()
        
        # 'The LightningModule knows what device it is on. You can access the reference via self.device. Sometimes it is necessary to store tensors as module attributes. However, if they are not parameters they will remain on the CPU even if the module gets moved to a new device. To prevent that and remain device agnostic, register the tensor as a buffer in your modules’s __init__ method with register_buffer().'
        self.register_buffer("epsilon", torch.tensor(1.e-8))

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

        def __init__(self, epsilon: torch.Tensor) -> None:

            super().__init__()

            self.epsilon = epsilon

            step = CONFIG['timestep']
            stats = torch.load(osp.join(DATA_PATH, f"stats-flattened-{step}.pt"))

            self.x_mean = stats["x_mean"]
            self.x_std = stats["x_std"]

        def forward(self, x: torch.Tensor) -> torch.Tensor:

            # Adjusting the tensors to scale to other devices. 'When you need to create a new tensor, use type_as. This will make your code scale to any arbitrary number of GPUs or TPUs with Lightning.'
            mean = self.x_mean.type_as(x)
            std = self.x_std.type_as(x)
            epsilon = torch.tensor(1.e-8).type_as(x)

            return (x - mean) / (std + self.epsilon)

    
    # Doing the feature selection in the model is maybe less efficient from a data loading perspecitve, but way more flexible. 
    # That way, you can train a model
    class SelectFeatures(nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            start_idx = 276 + 136 + 17
    
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 lr: float = 1e-4) -> None:
        
        super().__init__()
        
        self.lr = lr
        
        self.normalization_layer = LitMLP.Normalize(self.epsilon)
        self.feature_selection_layer = LitMLP.SelectFeatures()
        
        self.net = nn.Sequential(
            self.normalization_layer,
            self.feature_selection_layer,
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )