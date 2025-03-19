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

# install vllm, on rocm, suggest install from source

git clone https://github.com/vllm-project/vllm.git
cd vllm
# if nvidia devices are used, can install main branch for better performance
git checkout v0.6.3
# export CCACHE_DIR=/scratch/ccahe if you encounter ccache permission problem
python setup.py develop
cd ..
```

#### Install

```bash
git clone https://github.com/rStar-RL/rStar-RL.git
cd rStar-RL/verl
pip install -e .
cd ..

# patch ray to enable ray set visible devices on AMD devices
bash patches/ray_patch/patch.sh
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
