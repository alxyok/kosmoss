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
from typing import Union
import xarray as xr

import config

def broadcast_features(tensor):
    t = da.repeat(tensor, 138, axis=-1)
    t = da.moveaxis(t, 1, -1)
    return t

def pad_tensor(tensor):
    return da.pad(tensor, ((0, 0), (1, 1), (0, 0)))

def perform(save_format: Union['npy', 'h5']):
    
    start_time = time.perf_counter()
    
    features = [
        'sca_inputs',
        'col_inputs',
        'hl_inputs',
        'inter_inputs',
        'flux_dn_sw',
        'flux_up_sw',
        'flux_dn_lw',
        'flux_up_lw',
    ]

    data = {}

    shards = glob(osp.join(config.processed_data_path, 'shards_h5', '*.h5'))
    with xr.open_mfdataset(shards, chunks=-1, combine="nested", concat_dim="concat_dim", parallel=True) as dataset:

        # all this is lazy
        for feat in features:
            shape = dataset[feat].shape
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                var = da.reshape(dataset[feat].data, (shape[0] * shape[1], *shape[2:]), merge_chunks=False)
            data.update({feat: var})

        # still lazy
        x = da.concatenate([
            data['hl_inputs'],
            pad_tensor(data['inter_inputs'][..., np.newaxis]),
            broadcast_features(data['sca_inputs'][..., np.newaxis])
        ], axis=-1)

        y = da.concatenate([
            data['flux_dn_sw'][..., np.newaxis],
            data['flux_up_sw'][..., np.newaxis],
            data['flux_dn_lw'][..., np.newaxis],
            data['flux_up_lw'][..., np.newaxis],
        ], axis=-1)
        
        if save_format == 'npy':
            out_path = osp.join(config.processed_data_path, 'feats_npy')
            da.to_npy_stack(osp.join(out_path, 'x'), x)
            da.to_npy_stack(osp.join(out_path, 'y'), y)
        elif save_format == 'h5':
            out_path = osp.join(config.processed_data_path, 'feats_h5')
            x.to_hdf5(out_path, '/x')
            y.to_hdf5(out_path, '/y')

    return x, y
        
if __name__ == '__main__':
    
    perform(config.params['shard_save_format'])