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

import dask
import dask.array as da
from glob import glob
from h5py import File
import json
import numpy as np
import os.path as osp
import time
import xarray as xr

import config

def main():
    start_time = time.perf_counter()

    shards = glob(osp.join(config.processed_data_path, 'shards', '*.h5'))
    with xr.open_mfdataset(shards, chunks=-1, combine="nested", concat_dim="concat_dim", parallel=True) as dataset:

        

        # still...
        x_mean = da.mean(x, axis=0)
        y_mean = da.mean(y, axis=0)
        x_std = da.std(x, axis=0)
        y_std = da.std(y, axis=0)

        stats_path = osp.join(config.data_path, f"stats-{config.params['timestep']}.h5")
        if not osp.isfile(stats_path) or config.params['force']:
            with File(stats_path, 'w') as file:
                tic = time.perf_counter()
                file.create_dataset("x_mean", data=x_mean)
                file.create_dataset("y_mean", data=y_mean)
                file.create_dataset("x_std", data=x_std)
                file.create_dataset("y_std", data=y_std)
                tac = time.perf_counter()

            end_time = time.perf_counter()
            exec_time = int(end_time - start_time)
            normalization_time = int(tac - tic)

            print(f'Total execution time: ~{exec_time}s.')

            # Print execution time in a JSON file
            with open(osp.join(config.artifacts_path, 'results.json'), 'w') as file:
                json.dump({
                    'exec_time': exec_time,
                    'normalization_time': normalization_time
                }, file)
                
        else:
            print(f'File: {stats_path} exists, force with option `force` at `True` in config.yaml')
        
        
if __name__ == '__main__':
    
    main()