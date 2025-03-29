# rStar-RL

## Dev Note  
**Please don't directly commit on the main branch.**

If you need to merge new features or fixes into the main branch, please submit a pull request (PR) to the main branch. Additionally, to keep the commit history clean, it's recommended to use the **Squash Merge** option. This way, all commits will be combined into a single commit, making the commit history on the main branch clearer and easier to track.  

### Verl

`verl` is based on v0.2 branch.

#### Preparation

```bash
# suggest create a new conda environment or clone from an vllm installed environment

# conda create -y -n rstar python=3.12
# or
# conda create -y -n rstar --clone py_3.9
# if your base environment is not in a conda environment, maybe command like following will work for cloning the base pkgs
# cp -r /usr/local/lib/python3.12/dist-packages/* ./miniconda3/envs/rstar/lib/python3.12/site-packages/

# install tensordict

git clone https://github.com/pytorch/tensordict.git
cd tensordict
pip install .
cd ..
```

SGLang have a better rollout performance compared with vLLM.

If you are using NVIDIA devices, please follow the official install method of SGLang/vLLM.

If you are using AMD devices, please follow the following steps.

Chosen 1 on MI300X: sglang v0.4.4.post1

```bash
# please make sure the vllm version is v0.7.3 or using the SGLang suggested docker image,
# other version may also work, but not tested.
# 
# Install a patched vllm v0.7.3 version to avoid dependency conflict with SGLang,
# or you may install a wrong cuda SGLang at the end.  
git clone https://github.com/rStar-RL/rStar-RL.git
bash rStar-RL/patches/vllm_patch/install_v0_7_3.sh

git clone -b v0.4.4.post1 https://github.com/sgl-project/sglang.git
# if using vllm version v0.7.3, patch SGLang pyproject.toml
cp rStar-RL/patches/sglang_patch/pyproject.toml sglang/python/
# patch SGLang scheduler to enable clear kv cache on rocm
cp rStar-RL/patches/sglang_patch/scheduler.py sglang/python/sglang/srt/managers/

pip install --upgrade pip
cd sglang/sgl-kernel
python setup_rocm.py install
cd ..
pip install -e "python[all_hip]"

# aiter is required by sglang on rocm
git clone https://github.com/ROCm/aiter.git
cd aiter
git checkout e70ee4d948fd8455e4d665ebcc6fa2654bad6137
git submodule update --init --recursive
PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py develop

# if you are using image: rocmshared/vllm-rocm:vllm_0.6.3_rocm6.2_ubuntu20.04_torch2.6.0_py3.9
# need to patch torchao to avoid it using some torch2.6 feature, because in this image is a special rocm torch2.6
python rStar-RL/patches/torchao_patch/downgrade_version_mi300.py
```

Chosen 2 on MI300X: vllm v0.6.3

```bash
# install vllm, on rocm, suggest install from source
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.3
# if you encounter ccache permission problem
export CCACHE_DIR=/scratch/ccahe 
python setup.py develop
cd ..
```

#### Install

```bash
git clone https://github.com/rStar-RL/rStar-RL.git
cd rStar-RL/verl
pip install -e .
cd ../..

# patch ray to enable ray set visible devices on AMD devices
bash rStar-RL/patches/ray_patch/patch.sh
```

### Launch Reward Server

We found it helpful to compute the reward score in a remote server, since
- the python environment is independent to the training process
- the server handles exceptions itself
- decouple the system, make the performance optimization easier

Note that the server only supports judging python and cpp code currently.

You can set up the server on a single node by following commands

```bash
# the python version should >= 3.10
conda create -y -n server python=3.10
sudo apt-get update && sudo apt-get install redis
pip install redis
pip install "fastapi[standard]"
redis-server --daemonize yes
git clone https://github.com/0xWJ/code-judge.git
cd code-judge
REDIS_URI=redis://localhost:6379 RUN_WORKERS=0 ~/.conda/envs/server/bin/fastapi run --workers 4 app/main.py
REDIS_URI=redis://localhost:6379 python run_workers.py
```

When calculating advantages becomes the bottle neck of the training system, you can try to deploy the server on
multiple nodes that have resources the server needs, like CPU cores. Here `REMOTE_REDIS_INFO` is the address and
port of the azure redis cache service.

```bash
# on the master node in Ray's training
REDIS_URI=REMOTE_REDIS_INFO RUN_WORKERS=0 fastapi run --workers 32 app/main.py --host 0.0.0.0

# on all worker nodes
REDIS_URI=REMOTE_REDIS_INFO python run_workers.py
```

### Multi-Node Training

```bash
# execute on node-0, can specify port by passing --port PORT_NUM
ray start --head
# execute on remaining nodes
ray start --address="NODE_0_IP:PORT_NUM"

# then execute the bash script on node-0
```
