import climetlab as cml
from distutils.errors import DistutilsArgError
from setuptools import Command

from kosmoss import CACHE_DATA_PATH
        
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
        cml.settings.set("cache-directory", CACHE_DATA_PATH)
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
        if not self.timestep or self.timestep < 500:
            raise DistutilsArgError("You must specify --timestep option >= 500")
    
    def run(self):
        cml.settings.set("cache-directory", CACHE_DATA_PATH)
        cmlds = cml.load_dataset(
            name="maelstrom-radiation-tf",
            dataset="3dcorrection",
            timestep=list(range(0, 3501, int(self.timestep))),
            filenum=list(range(5)),
            hr_units="K d-1",
        )
        cmlds.to_tfdataset(batch_size=self.batchsize)