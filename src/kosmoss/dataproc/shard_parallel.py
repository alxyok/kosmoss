import h5py
from mpi4py import MPI
import numpy as np
import os.path as osp
import sys

from kosmoss import (CONFIG,
                     PROCESSED_DATA_PATH)

def main() -> None:
    
    sys.path.append(osp.abspath('..'))

    timestep = CONFIG['timestep']
    h5_path = osp.join(PROCESSED_DATA_PATH, f'features-{timestep}.h5')

    rank = MPI.COMM_WORLD.rank
    print(f'worker of rank {rank} started.')

    for subidx in np.arange(53):
        
        print(f'processing slice {subidx} for rank {rank}.')
        start = rank * 53 * 2 ** 4 + subidx * 4800
        end = start + 4800

        with h5py.File(h5_path, 'r') as feats:

            sharded_path = osp.join(
                PROCESSED_DATA_PATH, 
                f'feats-{timestep}.{rank}.{subidx}.h5')
            with h5py.File(sharded_path, 'w') as sharded:

                sharded.create_dataset("/x", data=feats['/x'][start:end])
                sharded.create_dataset("/y", data=feats['/y'][start:end])
                sharded.create_dataset("/edge", data=feats['/edge'][start:end])
                
    print(f'ending session for worker of rank {rank}.')
    
    
if __name__ == "__main__":
    
    main()