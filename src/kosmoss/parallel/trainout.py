from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import SLURMEnvironment

from kosmoss.parallel.data import FlattenedDataModule
from kosmoss.parallel.models import LitMLP

# This file is for launching a training with an srun. To perform a run with a SlurmCluster object, follow the guide at https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html#building-slurm-scripts. We are not however much interested in the grid search promoted by the guide since we will be using the Ray Tune framwork in the last part.

def main(config):
    
    datamodule = FlattenedDataModule(**config["data"])
    model = LitMLP(**config["models"])
    
    trainer = Trainer(
        gpus=8, 
        num_nodes=4, 
        strategy="ddp",
        
        # By default, a failed sbatch will be resubmitted
        # To deactivate the behavior, configure the SLURM environment with no auto_requeue
        plugins=[SLURMEnvironment(auto_requeue=False)]
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    
    config = {
        "data": {
            "batch_size": 256,
            "num_workers": 16,
        },
        "models": {
            "in_channels": 20,
            "hidden_channels": 100,
            "out_channels": 4,
            "lr": 1e-4
        }
    }

    main(config)
