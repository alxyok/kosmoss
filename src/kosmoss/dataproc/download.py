import climetlab as cml
from distutils.errors import DistutilsArgError
import os
import os.path as osp
from pathlib import Path
from setuptools import Command

from kosmoss import CACHED_DATA_PATH, DATA_PATH
        
cml.settings.set("cache-directory", CACHED_DATA_PATH)
cml.settings.set("number-of-download-threads", 16)

class Download(Command):
    
    user_options = [
        ('timestep=', 't', 'Temporal sampling step.'),
    ]
    
    def initialize_options(self):
        self.timestep = None
        
    def finalize_options(self):
        if not self.timestep:
            raise DistutilsArgError("You must specify --timestep option")
    
    def run(self):
        
        cml.load_dataset(
            name="maelstrom-radiation",
            dataset="3dcorrection",
            timestep=list(range(0, 3501, int(self.timestep))),
            raw_inputs=False,
        )

class ConvertTFRecord(Command):
    
    user_options = [
        ('timestep=', 't', 'Temporal sampling step.'),
        ('batchsize=', 'b', 'Batch size.'),
    ]
    
    def initialize_options(self):
        self.timestep = None
        self.batchsize = 256
        
    def finalize_options(self):
        if not self.timestep or int(self.timestep) < 500:
            raise DistutilsArgError("You must specify --timestep option >= 500")
    
    def run(self):
        
        cmlds = cml.load_dataset(
            name="maelstrom-radiation-tf",
            dataset="3dcorrection",
            timestep=list(range(0, 3501, int(self.timestep))),
            filenum=list(range(5)),
            hr_units="K d-1",
        )
        cmlds.to_tfdataset(batch_size=self.batchsize)

# class LinkCache(Command):
    
#     user_options = [
#         ('srcpath=', 'p', "Path for the source 'cached' data dir."),
#     ]
    
#     def initialize_options(self):
#         self.srcpath = None
        
#     def finalize_options(self):
#         if not self.srcpath:
#             raise DistutilsArgError("You must specify --srcpath option")
    
#     def run(self):
        
#         os.makedirs(DATA_PATH, exist_ok=True)
        
#         if osp.isdir(CACHED_DATA_PATH):
#             print(f"directory: {CACHED_DATA_PATH} already exists, failing gracefully...")
#             return
            
#         path = Path(self.srcpath).resolve()
#         os.symlink(path, CACHED_DATA_PATH, target_is_directory=True)