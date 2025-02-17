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
