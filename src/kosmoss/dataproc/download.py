import climetlab as cml
import os
import os.path as osp

from kosmoss import CACHED_DATA_PATH

cml.settings.set("cache-directory", CACHED_DATA_PATH)
cml.settings.set("number-of-download-threads", 16)

def download():
    cml.load_dataset(
        'maelstrom-radiation', 
        dataset='3dcorrection', 
        raw_inputs=False, 
        timestep=list(range(0, 3501, 250)), 
        minimal_outputs=False,
        patch=list(range(0, 16, 1)),
        hr_units='K d-1',
    )

def download_tfrecords():
    cmlds = cml.load_dataset(
        'maelstrom-radiation-tf',
        dataset='3dcorrection',
        timestep=list(range(0, 3501, 500)), 
        filenum=list(range(5)),
        norm=True,
        hr_units="K d-1",
    )
    cmlds.to_tfdataset(batch_size=256)
    
if __name__ == "__main__":
    
    download()
    download_tfrecords()