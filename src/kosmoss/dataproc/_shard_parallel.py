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
from mpi4py import MPI
import numpy as np
import os.path as osp
import sys

import kosmoss as km

def main() -> None:
    
    sys.path.append(osp.abspath('..'))

    timestep = km.CONFIG['timestep']
    h5_path = osp.join(km.PROCESSED_DATA_PATH, f'features-{timestep}.h5')

    rank = MPI.COMM_WORLD.rank
    print(f'worker of rank {rank} started.')

    for subidx in np.arange(53):
        
        print(f'processing slice {subidx} for rank {rank}.')
        start = rank * 53 * 2 ** 4 + subidx * 4800
        end = start + 4800

        with h5py.File(h5_path, 'r') as feats:

            sharded_path = osp.join(
                km.PROCESSED_DATA_PATH, 
                f'feats-{timestep}.{rank}.{subidx}.h5')
            with h5py.File(sharded_path, 'w') as sharded:

                sharded.create_dataset("/x", data=feats['/x'][start:end])
                sharded.create_dataset("/y", data=feats['/y'][start:end])
                sharded.create_dataset("/edge", data=feats['/edge'][start:end])
                
    print(f'ending session for worker of rank {rank}.')
    
    
if __name__ == "__main__":
    
    main()