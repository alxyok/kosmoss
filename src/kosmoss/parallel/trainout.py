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

import argparse
import os.path as osp
import pytorch_lightning as ps
import sys

import kosmoss as km

# This file is for launching a training with an srun. To perform a run with a SlurmCluster object, follow the guide at https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html#building-slurm-scripts. We are not however much interested in the grid search promoted by the guide since we will be using the Ray.tune framwork in the last part.

def main(hparams):
    
    model = km.parallel.models.LitMLP(hparams)
    trainer = pl.Trainer(gpus=8, num_nodes=4, strategy="ddp")
    trainer.fit(model)


if __name__ == "__main__":
    root_dir = osp.dirname(osp.realpath(__file__))
    parent_parser = argparse.ArgumentParser(add_help=False)
    hyperparams = parser.parse_args()

    # TRAIN
    main(hyperparams)