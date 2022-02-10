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

import os
import os.path as osp
import logging
import randomname

root_path = osp.dirname(os.path.realpath(__file__))

data_path = osp.join(root_path, 'data')
cache_data_path = osp.join(data_path, 'cache')
raw_data_path = osp.join(data_path, 'raw')
processed_data_path = osp.join(data_path, 'processed')

# Create all path for the current experiment
experiments_path = osp.join(root_path, 'experiments')
os.makedirs(experiments_path, exist_ok=True)
_existing_xps = os.listdir(experiments_path)

# Generate experiment name
_randomize_name = True
while _randomize_name:
    _experiment_name = randomname.get_name()
    if _experiment_name not in _existing_xps:
        break
experiment_path = osp.join(experiments_path, _experiment_name)

logs_path = osp.join(experiment_path, 'logs')
artifacts_path = osp.join(experiment_path, 'artifacts')
plots_path = osp.join(experiment_path, 'plots')

_paths = [
    experiment_path, 
    logs_path, 
    artifacts_path, 
    plots_path
]
for path in _paths:
    os.makedirs(path, exist_ok=True)
    
logging.basicConfig(filename=osp.join(logs_path, f'{_experiment_name}.log'), 
                    filemode='w', 
                    format='%(name)s - %(levelname)s - %(message)s')


step = 1000