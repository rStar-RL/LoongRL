cd ~
source /opt/conda/etc/profile.d/conda.sh 
conda create -y -n rstar python==3.12
conda activate rstar

# tensordict
cd ~
git clone https://github.com/pytorch/tensordict.git
cd tensordict
pip install .
cd ..

# vllm
pip install vllm

# rStar-RL
git clone https://github.com/rStar-RL/rStar-RL.git
cd rStar-RL/verl
pip install -e .
cd ..

bash patches/ray_patch/patch.sh