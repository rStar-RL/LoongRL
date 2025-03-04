#!/bin/bash

# Record the current directory
CURRENT_DIR=$(pwd)

# Set up conda environment
cd ~
source /opt/conda/etc/profile.d/conda.sh 
conda create -y -n rstar python==3.12
conda activate rstar

# Install tensordict
cd ~
git clone https://github.com/pytorch/tensordict.git
cd tensordict
pip install .
cd ..

# Install vllm and flash-attn
pip install vllm
pip install flash-attn

# Clone and install rStar-RL
# git clone https://github.com/rStar-RL/rStar-RL.git
# cd rStar-RL/verl
cd "$CURRENT_DIR"
pip install -e .
cd ..

# Install pandas and wandb
pip install pandas
pip install wandb

# Apply ray patch
bash patches/ray_patch/patch.sh

# Return to the original directory