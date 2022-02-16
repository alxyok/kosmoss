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

import json
import os.path as osp
import sys
import yaml

import utils

root_path = osp.join(osp.dirname(osp.realpath(__file__)))

data_path, cache_data_path, raw_data_path, processed_data_path, artifacts_path, logs_path = utils.makedirs([
    osp.join(root_path, '_data'),
    osp.join(root_path, '_data', 'cache'),
    osp.join(root_path, '_data', 'raw'),
    osp.join(root_path, '_data', 'processed'),
    osp.join(root_path, '_artifacts'),
    osp.join(root_path, '_logs'),
])
    
logger = logging.getLogger("")
logger.addHandler(logging.StreamHandler(sys.stdout))

# Export initial config parameters for access in all modules
with open(osp.join(root_path, "config.yaml"), "r") as stream:
    config = yaml.safe_load(stream)

# Export artifacts params for all training modules
params_path = osp.join(root_path, "params.json")
if osp.isfile(params_path):
    with open(params_path, "r") as stream:
        params = json.load(stream)
