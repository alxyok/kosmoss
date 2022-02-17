import climetlab as cml
import dask
import dask.array as da
from glob import glob
import numpy as np
import os.path as osp
import xarray as xr

import kosmoss as km

def main():

    step = km.CONFIG['timestep']
    
    cml.settings.set("cache-directory", km.CACHE_DATA_PATH)
    cmlds = cml.load_dataset(
        'maelstrom-radiation', 
        dataset='3dcorrection', 
        raw_inputs=False, 
        timestep=list(range(0, 3501, step)), 
        minimal_outputs=False,
        patch=list(range(0, 16, 1)),
        hr_units='K d-1',
    )
    

if __name__ == "__main__":
    
    main()