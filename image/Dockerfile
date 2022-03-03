FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

LABEL maintainer="mlops"
LABEL name="Kosmoss-Ubuntu"
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
RUN apt install -y apt-utils \
                   python3.8 \
                   python3.8-dev \
                   python3.8-full \
                   build-essential \
                   diffutils \
                   libblas-dev \
                   liblapack-dev \
                   libatlas-base-dev \
                   libglib2.0-0 \
                   libhdf5-mpi-dev \
                   cmake \
                   gfortran \
                   man-db \
                   htop \
                   wget \
                   git \
                   curl \
                   sudo \
                   vim \
    && apt clean all \
    && rm -rf /var/cache/apt/*

RUN ln -sf python3 /usr/bin/python
RUN ln -sf pip3 /usr/bin/pip


# ---------------------------------------
# Upgrade pip and install wheel
RUN echo "http_proxy=${http_proxy}" >> ${HOME}/.wgetrc \
    && echo "https_proxy=${http_proxy}" >> ${HOME}/.wgetrc \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && rm get-pip.py
RUN python -m pip install --upgrade pip
RUN python -m pip install wheel


# ---------------------------------------
# Install cuPy
RUN python -m pip install cupy-cuda113


# ---------------------------------------
# Install PyTorch and PyG
RUN python -m pip install torch==1.10.2+cu113 \
                          torchvision==0.11.3+cu113 \
                          torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN python -m pip install torch-scatter \
                          torch-sparse \
                          torch-cluster \
                          torch-spline-conv \
                          torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
   
   
# ---------------------------------------
# Install TensorFlow + TensorBoard
RUN python -m pip install tensorflow


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
RUN export CC=mpicc \
    && HDF5_MPI="ON" \
    && pip install mpi4py \
    && pip install --no-binary=h5py h5py
RUN C_INCLUDE_PATH=/usr/include/python3.9 pip install bottleneck



# ---------------------------------------
# Install requirements for this project
COPY ../requirements.txt ${HOME}
RUN python -m pip install -r ${HOME}/requirements.txt



# ---------------------------------------
# Misc configuration
RUN mkdir ${HOME}/.jupyter
COPY jupyter_server_config.py ${HOME}/.jupyter/
COPY .pylintrc /etc/pylintrc

EXPOSE 6006

CMD ["jupyter", "lab"]
