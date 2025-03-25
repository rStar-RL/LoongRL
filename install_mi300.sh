# This script is for
#
# image: rocmshared/vllm-rocm:vllm_0.6.3_rocm6.2_ubuntu20.04_torch2.6.0_py3.9
# registry: srgxacr.azurecr.io
#
# with sglang rollout.

conda init
bash

conda create -y -n rstar --clone py_3.9
conda activate rstar

git clone https://github.com/rStar-RL/rStar-RL.git

# install tensordict
pip uninstall -y tensordict
git clone https://github.com/pytorch/tensordict.git
cd tensordict
pip install .
cd ..

# install a requirement modified vllm v0.7.3
bash /scratch/nishang/rStar-RL/patches/vllm_patch/install_v0_7_3.sh

# install aiter
git clone https://github.com/ROCm/aiter.git
cd aiter
git checkout e70ee4d948fd8455e4d665ebcc6fa2654bad6137
git submodule update --init --recursive
PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py develop
cd ..

# install sglang
git clone -b v0.4.4.post1 https://github.com/sgl-project/sglang.git
cp rStar-RL/patches/sglang_patch/pyproject.toml sglang/python/
cp rStar-RL/patches/sglang_patch/scheduler.py sglang/python/sglang/srt/managers/
cp rStar-RL/patches/sglang_patch/custom_all_reduce.py sglang/python/sglang/srt/distributed/device_communicators/
cd sglang/sgl-kernel
python setup_rocm.py install
cd ..
pip install -e "python[all_hip]"
cd ..

# patch torchao because the torch v2.6 in this image infact don't include some features used by torchao
python rStar-RL/patches/torchao_patch/downgrade_version_mi300.py

# install verl
cd rStar-RL/verl
pip install -e .
cd ../..

# patch ray for AMD device
bash rStar-RL/patches/ray_patch/patch.sh

echo "main dependence of RL training installed, please view useful commands after exit in this script."

exit

# download some models
huggingface-cli download Qwen/Qwen2.5-7B --local-dir Qwen2.5-7B
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-32B --local-dir Qwen2.5-32B
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir Qwen2.5-32B-Instruct

# install code server
conda create -y -n server python=3.10
conda activate server
sudo apt-get update && sudo apt-get install redis
pip install redis
pip install "fastapi[standard]"

git clone https://github.com/0xWJ/code-judge.git
cd code-judge
pip install -r requirements.txt

# start redis
redis-server --daemonize yes

# start fastapi
tmux new-session -t server
bash
conda activate server
cd /scratch/nishang/code-judge
export REDIS_URI="redis://localhost:6379"
REDIS_URI=$REDIS_URI RUN_WORKERS=0 ~/.conda/envs/server/bin/fastapi run --workers 4 app/main.py

# start worker
tmux new-session -t worker
bash
conda activate server
cd /scratch/nishang/code-judge
export REDIS_URI="redis://localhost:6379"
REDIS_URI=$REDIS_URI MAX_WORKERS=64 python run_workers.py

# start ray
ray start --head --port 0
ray start --address="node-0:6379"
