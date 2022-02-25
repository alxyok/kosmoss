#!/bin/bash -l

#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --time=0-00:30:00
#SBATCH -p milan7513_A100

# activate conda env
source activate $1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# select the torch distributed backend. 'nccl' or 'gloo'
export PL_TORCH_DISTRIBUTED_BACKEND='nccl'

# run script from above
srun python trainout.py