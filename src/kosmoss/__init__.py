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
import logging
import os
import os.path as osp
import sys
import yaml

import kosmoss.utils as utils

EGG_PATH = osp.join(osp.dirname(osp.realpath(__file__)))
ROOT_PATH = osp.join(os.environ['HOME'], '.kosmoss')
os.makedirs(ROOT_PATH, exist_ok=True)

DATA_PATH, CACHE_DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, ARTIFACTS_PATH, LOGS_PATH = utils.makedirs([
    osp.join(ROOT_PATH, 'data'),
    osp.join(ROOT_PATH, 'data', 'cache'),
    osp.join(ROOT_PATH, 'data', 'raw'),
    osp.join(ROOT_PATH, 'data', 'processed'),
    osp.join(ROOT_PATH, 'artifacts'),
    osp.join(ROOT_PATH, 'logs'),
])
    
LOGGER = logging.getLogger("")
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

# Export initial config parameters for access in all modules
with open(osp.join(EGG_PATH, "config.yaml"), "r") as stream:
    CONFIG = yaml.safe_load(stream)

# Export artifacts params for all training modules
_params_path = osp.join(EGG_PATH, "params.json")
if osp.isfile(_params_path):
    with open(_params_path, "r") as stream:
        PARAMS = json.load(stream)
