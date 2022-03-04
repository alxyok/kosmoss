# kosmoss

Prerequisites: valid [CUDA install](https://developer.nvidia.com/cuda-downloads) and [Poetry](https://python-poetry.org/docs/master/#installing-with-the-official-installer).

To install:
* `git clone` this repo and `cd` into it
* `poetry install` to install this project as a library with its dependencies
* `poetry run download` to download data for PyTorch + Lightning content
* `poetry run download_tfrecords` to download data for TensorFlow + Keras content

One note
1. All of the material is based on the work of AI4sim team on a real-world use-case for ECMWF
2. This is just an introduction, not a comprehensive, in-depth, hands-on material
3. No low-level optimizations