import json
import logging
import os
import os.path as osp
import shutil
import sys
import yaml

from kosmoss import utils

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

shutil.copy(osp.join(EGG_PATH, "config", "config.yaml"), ROOT_PATH)
shutil.copy(osp.join(EGG_PATH, "config", "params.json"), ROOT_PATH)
CONFIG = utils.load_attr(osp.join(ROOT_PATH, "config.yaml"), 'yaml')
PARAMS = utils.load_attr(osp.join(ROOT_PATH, "params.json"), 'json')