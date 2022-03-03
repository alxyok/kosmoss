import h5py
from mpi4py import MPI
import numpy as np
import os
import os.path as osp

from kosmoss import PROCESSED_DATA_PATH

def main() -> None:

    # Being multiprocessed, threads can't open the same file for concurrency-related issues
    # So we don't read the timestep variable from the config.yaml file and instead, have to fix the value
    timestep = 1000
    h5_path = osp.join(PROCESSED_DATA_PATH, f'features-{timestep}.h5')
    out_dir = osp.join(PROCESSED_DATA_PATH, f"features-{timestep}")
    os.makedirs(out_dir, exist_ok=True)

    # The MPI Rank uniquely identify each process
    rank = MPI.COMM_WORLD.rank
    print(f'worker of rank {rank} started.')

    # Each process will produce 53 files
    for subidx in np.arange(53):
        
        print(f'processing slice {subidx} for rank {rank}.')
        
        # Each file holding 4800 records
        start = rank * 53 * 2 ** 4 + subidx * 4800
        end = start + 4800

        # h5py is not built for concurrency, and os error can occur
        # So we have to loop until the lock is released
        while True:
            try:
                with h5py.File(h5_path, 'r') as feats:
                    x = feats['/x'][start:end]
                    y = feats['/y'][start:end]
                    edge = feats['/edge'][start:end]
                break
                
            except BlockingIOError:
                pass
                    
        # Give the output file a unique name to avoid overriting
        name = (rank + 1) * (subidx + 1)
        sharded_path = osp.join(out_dir, f'features-{name}.h5')
        
        with h5py.File(sharded_path, 'w') as sharded:
            sharded.create_dataset("/x", data=x)
            sharded.create_dataset("/y", data=y)
            sharded.create_dataset("/edge", data=edge)
                
    print(f'ending session for worker of rank {rank}.')
    
    
if __name__ == "__main__":
    
    main()