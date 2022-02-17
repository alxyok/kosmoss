import argparse
import os.path as osp
from pytorch_lightning import Trainer

from kosmoss.parallel.models import LitMLP

# This file is for launching a training with an srun. To perform a run with a SlurmCluster object, follow the guide at https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html#building-slurm-scripts. We are not however much interested in the grid search promoted by the guide since we will be using the Ray.tune framwork in the last part.

def main(hparams):
    
    model = LitMLP(hparams)
    trainer = Trainer(gpus=8, num_nodes=4, strategy="ddp")
    trainer.fit(model)


if __name__ == "__main__":
    root_dir = osp.dirname(osp.realpath(__file__))
    parent_parser = argparse.ArgumentParser(add_help=False)
    hyperparams = parser.parse_args()

    # TRAIN
    main(hyperparams)