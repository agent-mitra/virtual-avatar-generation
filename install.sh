#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
mamba create -n sabr python=3.10 -y
bash ./prepare_data.sh
cd pipeline/track
wget -qO- cli.runpod.net | sudo bash
conda activate sabr && {
    mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
    pip install -e .[all]
    apt-get update
    apt install libglu1-mesa libx11-dev libegl1-mesa-dev libgl1-mesa-dev -y
    mamba update ffmpeg -y
    pip install firebase-admin==6.1.0
    pip install google-cloud-storage==2.8.0
    pip install -U openmim
    mim install mmcv-full
    pip install mmdet==2.28.1
    pip install mmpose==0.24.0
    pip install timm==0.4.9 einops
    pip install scikit-image
    pip install git+https://github.com/facebookresearch/segment-anything.git
    pip install wandb cvbase imageio==2.31.4 gdown botocore boto3 ultralytics==8.0.99
    pip install imageio-ffmpeg==0.4.9
    pip install opencv-python
    pip install hydra-core
    cd ..
    cd context
    bash ./install.sh
    bash ./download_ckpt.sh
    cd ..
    pip install packaging
    pip install ninja pyrender json_tricks PyOpenGL==3.1.0 munkres pyrootutils hydra_colorlog --upgrade hydra-submitit-launcher --upgrade onnxruntime-gpu
    pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/ #if on CUDA 12.2
    pip install flash-attn --no-build-isolation
    mamba install xformers -c xformers -y
    pip install git+https://github.com/facebookresearch/detectron2.git
    cd ..
    echo "Installation complete."
}
