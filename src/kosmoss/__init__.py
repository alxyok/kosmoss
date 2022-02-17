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
