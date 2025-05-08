#!/bin/bash

# Record the current directory
CURRENT_DIR=$(pwd)
source /home/aiscuser/anaconda3/etc/profile.d/conda.sh

conda create -y -n rstar python==3.12
conda activate rstar

cd ~
git clone https://github.com/pytorch/tensordict.git
cd tensordict
git checkout ecdde0b1b23374f8b2f438b396f3cf7fdfa6e741
pip install .
cd ..

# install verl
cd ~/rStar-RL/verl
pip install -e .
cd ../..

# patch ray for AMD device
bash rStar-RL/patches/ray_patch/patch.sh

# patch sglang to support AMD device release kv cache after rollout
sudo cp rStar-RL/patches/sglang_046_patch/scheduler.py /sgl-workspace/sglang/python/sglang/srt/managers/
sudo cp rStar-RL/patches/sglang_046_patch/verl_engine.py /sgl-workspace/sglang/python/sglang/srt/entrypoints/

# pre-init aiter, or it will case error at first time running experiment
python -c "from sglang.srt.entrypoints.verl_engine import VerlEngine"

# install code server
sudo apt-get update -y && sudo apt-get install redis -y
pip install redis
pip install "fastapi[standard]"

pip install sympy
pip install numpy
pip install scipy