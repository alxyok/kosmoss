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

    xr_array = cmlds.to_xarray()
    xr_array.to_netcdf(osp.join(km.RAW_DATA_PATH, f'data-{step}.nc'))
    

if __name__ == "__main__":
    
    main()