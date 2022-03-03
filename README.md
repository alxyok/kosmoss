# kosmoss

Prerequisites: valid CUDA install.

To install:
* `git clone` this repo
* `cd kosmoss` to enter project directory
* `python -m pip install -r requirements.txt` to install project dependencies
* `python -m pip install .` to install the project as a library
* `python setup.py download -t 250` to download a substantial dataset for all PyTorch + Lightning content
* Optionally, `python setup.py convert -t 500` to download and convert a reduced dataset for optional TensorFlow + Keras content

One note
1. All of the material is based on the work of AI4sim team on a real-world use-case for ECMWF
2. This is just an introduction, not a comprehensive, in-depth, hands-on material
3. No low-level optimizations