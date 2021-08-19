FROM nvidia/cuda:11.1-devel-ubuntu18.04 as base

ENV WORKDIR=/tp3
WORKDIR $WORKDIR

# Install wget for the miniconda installation, and
# dependencies for python-fcl
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && apt-get install -y --no-install-recommends libfcl-dev\
    && apt-get install -y --no-install-recommends liboctomap-dev \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

# Need to set bash shell for conda
# https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["/bin/bash", "--login", "-c"]

# Install miniconda
# https://towardsdatascience.com/conda-pip-and-docker-ftw-d64fe638dc45
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# make non-activate conda commands available
ENV PATH=/root/miniconda3/bin:$PATH

# make conda activate command available
RUN echo ". /root/miniconda3/etc/profile.d/conda.sh" >> ~/.profile \
    && conda init bash

# conda dependencies
RUN conda install -n base -c conda-forge mamba -y

COPY grasp_refinement_env.yaml grasp_refinement_env.yaml
RUN mamba env create -f grasp_refinement_env.yaml \
    && mamba clean -afy \
    && echo "conda activate gref" >> ~/.bashrc

# pip dependencies
# removed tensorflow because it was downgrading numpy and breaking others
RUN conda activate gref \
    && /root/miniconda3/envs/gref/bin/pip install --no-cache-dir --compile pyrender \
    # && /root/miniconda3/envs/gref/bin/pip install --no-cache-dir --compile tensorflow \
    && /root/miniconda3/envs/gref/bin/pip install --no-cache-dir --compile python-fcl \
    && /root/miniconda3/envs/gref/bin/pip install --no-cache-dir --compile hydra-core \
    && rm -rf /root/.cache \
    && mamba clean -afy

# separate install for torch-geometric packages to debug environment vars
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV CPATH=/usr/local/cuda/include:$CPATH
RUN conda activate gref \
    && FORCE_CUDA=1 /root/miniconda3/envs/gref/bin/pip install --no-cache-dir --compile torch-points-kernels \
    && /root/miniconda3/envs/gref/bin/pip install --no-cache-dir --compile \
    torch-scatter torch-sparse torch-cluster -f \
    https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html \
    && /root/miniconda3/envs/gref/bin/pip install --no-cache-dir --compile torch-geometric \
    && /root/miniconda3/envs/gref/bin/pip install --no-cache-dir --compile pytorch-metric-learning \
    && /root/miniconda3/envs/gref/bin/pip install --no-cache-dir --compile torchnet \
    && rm -rf /root/.cache \
    && mamba clean -afy

