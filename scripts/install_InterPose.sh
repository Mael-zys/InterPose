#!/bin/bash

conda create -n interpose python=3.8

source ~/miniconda3/etc/profile.d/conda.sh

conda activate interpose

# Install PyTorch.
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y


# Install PyTorch3D.
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt200/download.html

# Install human_body_prior.
cd ..
git clone https://github.com/nghorbani/human_body_prior.git
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior/
python setup.py develop

cd InterPose
# Install BPS.
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch

# Install other dependencies.
pip install -r requirements.txt 

