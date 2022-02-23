import climetlab as cml
from distutils.errors import DistutilsArgError
from setuptools import Command

from kosmoss import CACHE_DATA_PATH


def download(timestep):
    cml.settings.set("cache-directory", CACHE_DATA_PATH)
    cmlds = cml.load_dataset(
        'maelstrom-radiation', 
        dataset='3dcorrection', 
        raw_inputs=False, 
        timestep=list(range(0, 3501, timestep)), 
        minimal_outputs=False,
        patch=list(range(0, 16, 1)),
        hr_units='K d-1',
    )
    return cmlds
        
        
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
        download(int(self.timestep))

        
class ConvertTFRecord(Command):
    
    user_options = [
        ('timestep=', 't', 'Temporal sampling step.'),
        ('batch_size=', 'bs', 'Batch size.'),
        ('repeat=', 'r', 'Repeat.'),
    ]
    
    def initialize_options(self):
        self.timestep = None
        
    def finalize_options(self):
        if not self.timestep:
            raise DistutilsArgError("You must specify --timestep option")
        if not self.batch_size:
            self.batch_size = 256
        if not self.repeat:
            self.repeat = False
    
    def run(self):
        cmlds = download(int(self.timestep))
        cmlds.to_tfdataset(batch_size=self.batch_size, repeat=self.repeat)