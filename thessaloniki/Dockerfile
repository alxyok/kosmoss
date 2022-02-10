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

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

LABEL maintainer="mlops"
LABEL name="Thessaloniki"
LABEL type="Development Image"

ARG PROXY

ENV https_proxy ${PROXY}
ENV http_proxy ${PROXY}

ENV HTTPS_PROXY ${PROXY}
ENV HTTP_PROXY ${PROXY}


# ---------------------------------------
# Install low-level dependencies
RUN echo "Acquire::http::Proxy \"${http_proxy}\";" >> /etc/apt/apt.conf
RUN apt update -y && apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Paris apt -y install tzdata
RUN apt install -y build-essential \
                   python3.9 \
                   python3.9-dev \
                   python3.9-full \
                   diffutils \
                   libblas-dev \
                   liblapack-dev \
                   libatlas-base-dev \
                   libglib2.0-0 \
                   cmake \
                   gfortran \
                   man-db \
                   git \
                   htop \
                   wget \
                   curl \
                   sudo \
                   vim \
                   zsh \
    && apt clean all \
    && rm -rf /var/cache/apt/*

RUN ln -snf python3 /usr/bin/python
RUN ln -snf pip3 /usr/bin/pip


# ---------------------------------------
# Upgrade pip and install wheel
RUN echo "http_proxy = ${http_proxy}" >> ${HOME}/.wgetrc \
    && echo "https_proxy = ${http_proxy}" >> ${HOME}/.wgetrc \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py
RUN pip install --upgrade pip
RUN pip install wheel


# ---------------------------------------
# Install PyTorch and PyG
RUN pip install torch==1.10.2+cu113 \
                torchvision==0.11.3+cu113 \
                torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip install torch-scatter \
                torch-sparse \
                torch-cluster \
                torch-spline-conv \
                torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html


# ---------------------------------------
# Install TensorFlow + TensorBoard
RUN pip install tensorflow


# ---------------------------------------
# Install a few other usefull packages for scientific-ML
WORKDIR /tmp/
ENV ECCODES "eccodes-2.24.0-Source"
RUN wget https://confluence.ecmwf.int/download/attachments/45757960/${ECCODES}.tar.gz?api=v2 -O ${ECCODES}.tar.gz \
    && tar zxf ${ECCODES}.tar.gz \
    && mkdir build; cd build \
    && cmake ../${ECCODES} \
    && make \
    && ctest --rerun-failed --output-on-failure \
    && make install
RUN C_INCLUDE_PATH=/usr/include/python3.9 pip install bottleneck
RUN pip install climetlab \
                climetlab-maelstrom-nogwd \
                climetlab-maelstrom-radiation \
                dask \
                h5py \
                matplotlib \
                metaflow \
                memory_profiler \
                netCDF4 \
                jupyterlab \
                numpy \
                nvitop \
                pandas \
                plotly \
                pytorch-lightning \
                PyYAML \
                randomname \
                scipy \
                sklearn \
                torchmetrics \
                torch_optimizer \
                xarray


# ---------------------------------------
# Misc configuration
RUN mkdir ${HOME}/.jupyter
COPY jupyter_server_config.py ${HOME}/.jupyter/
COPY .pylintrc /etc/pylintrc

CMD ["jupyter", "lab"]
