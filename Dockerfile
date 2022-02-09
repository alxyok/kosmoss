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

LABEL maintainer="mluser"
LABEL name="Thessaloniki"
LABEL type="Development Image"

ARG PROXY

ENV https_proxy ${PROXY}
ENV http_proxy ${PROXY}

ENV HTTPS_PROXY ${PROXY}
ENV HTTP_PROXY ${PROXY}


# ---------------------------------------
# Install Python 3.9, Git and others
RUN echo "Acquire::http::Proxy \"${http_proxy}\";" >> /etc/apt/apt.conf
RUN apt update -y && apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Paris apt -y install tzdata
RUN apt install -y build-essential \
                   python3.9 \
                   python3.9-dev \
                   python3.9-full \
                   libblas-dev \
                   liblapack-dev \
                   libatlas-base-dev \
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

RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

RUN ln -snf python3 /usr/bin/python
RUN ln -snf pip3 /usr/bin/pip


# ---------------------------------------
# Create mluser and make it sudoer
ENV USER mluser
ENV HOME /home/${USER}

RUN adduser --disabled-password --gecos "" ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${USER}
WORKDIR ${HOME}
ENV PATH "${HOME}/.local/bin:${PATH}"

RUN echo "http_proxy = ${http_proxy}" >> ~/.wgetrc \
    && echo "https_proxy = ${http_proxy}" >> ~/.wgetrc \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py


# ---------------------------------------
# Upgrade pip and install wheel
RUN pip install --upgrade pip
RUN pip install --user wheel


# ---------------------------------------
# Install PyTorch and PyG
RUN pip install --user torch==1.10.2+cu113 \
                       torchvision==0.11.3+cu113 \
                       torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip install --user torch-scatter \
                       torch-sparse \
                       torch-cluster \
                       torch-spline-conv \
                       torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html


# ---------------------------------------
# Install TensorFlow, TensorBoard and TFX
RUN pip install --user tensorflow


# ---------------------------------------
# Install a few other usefull packages for scientific-ML
RUN C_INCLUDE_PATH=/usr/include/python3.9 pip install --user bottleneck
RUN pip install --user climetlab \
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
# Quick config for Jupyter
RUN mkdir ${HOME}/.jupyter
COPY resources/jupyter_server_config.py ${HOME}/.jupyter/

CMD ["bash"]
