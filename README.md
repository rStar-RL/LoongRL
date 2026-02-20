# LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts **[ICLR 2026 Oral]**

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026%20Oral-blue)](https://arxiv.org/abs/2510.19363)
[![arXiv](https://img.shields.io/badge/arXiv-2510.19363-b31b1b.svg)](https://arxiv.org/abs/2510.19363)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm-dark.svg)](https://huggingface.co/papers/2510.19363)
<a href="https://huggingface.co/datasets/OldKingMeister/LoongRL-Train-Data"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg" height="20"></a>

This repository contains the official implementation of **LoongRL**, a reinforcement learning framework for training large language models on long-context question answering and mathematical reasoning tasks. The project uses a customized version of [veRL (HybridFlow)](https://github.com/volcengine/verl) optimized for **Group Relative Policy Optimization (GRPO)** with support for AMD MI300X and NVIDIA GPUs.

**Paper**: [LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](https://arxiv.org/abs/2510.19363)

**Data Creation**: Looking for our long-context training data synthesis pipeline? Check out [**KeyChain**](https://github.com/Wangmerlyn/KeyChain) — our UUID-driven data creation pipeline that easily constructs high-quality multi-hop QA instances over long contexts with customized multi-level distractors, enabling fine-grained control over difficulty.

## Key Features

- **GRPO Training**: Critic-free reinforcement learning with group-wise advantage estimation
- **Long Context QA**: RULER, NIAH (needle-in-haystack), and extended context reasoning tasks
- **Math Reasoning**: Verifiable reward functions for GSM8K, MATH, MathQA, AIME datasets
- **Multi-node Distributed Training**: Scales to 7B-32B models across GPU clusters
- **AMD MI300X Optimized**: ROCm patches for vLLM 0.7.3 and SGLang 0.4.4.post1
- **Flexible Reward Functions**: 16+ reward implementations including answer verification, process rewards, and execution-based evaluation

## Main Results

LoongRL delivers frontier-level long-context reasoning at much smaller scales (7B/14B), rivaling o3-mini and DeepSeek-R1, while preserving general short-context abilities across all scales.

| | Long-Context Reasoning | | | | | | General & Short Reasoning | | | |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model** | **Avg.** | **HotpotQA** | **2WikiMQA** | **MuSiQue** | **NarrativeQA** | **QASPER** | **Avg.** | **MMLU** | **MATH** | **IFEval** |
| o3-mini (medium) | 74.5 | 83.0 | 89.0 | 64.0 | 60.7 | 60.5 | **92.1** | 86.9 | **98.0** | **91.5** |
| DeepSeek-R1 | **74.9** | 82.7 | 91.3 | **72.2** | **66.9** | 61.4 | 90.5 | **90.8** | 97.3 | 83.3 |
| GPT-4o | 64.7 | 82.5 | 78.0 | 54.0 | 60.5 | 48.5 | 82.5 | 88.7 | 74.6 | 84.3 |
| QwQ-32B | 69.6 | 78.5 | 87.4 | 62.7 | 61.1 | 58.5 | 85.9 | 75.7 | **98.0** | 83.9 |
| R1-Distill-LLaMA-70B | 65.4 | 76.1 | 85.0 | 61.9 | 53.4 | 50.5 | 85.4 | 82.4 | 94.5 | 79.3 |
| | | | | | | | | | | |
| Qwen2.5-7B-Instruct | 48.9 | 69.5 | 50.5 | 34.0 | 44.5 | 46.0 | 73.5 | 73.4 | 76.0 | **71.2** |
| R1-Distill-Qwen-7B | 31.2 | 40.2 | 53.3 | 11.1 | 8.9 | 42.5 | 69.9 | 62.3 | **92.8** | 54.7 |
| **LoongRL-7B** | **72.4** | **83.1** | **91.1** | **65.6** | **58.4** | **63.6** | **75.0** | **76.2** | 78.0 | 70.9 |
| | | | | | | | | | | |
| Qwen2.5-14B-Instruct | 53.1 | 74.0 | 60.5 | 36.5 | 48.5 | 46.0 | 81.3 | 79.4 | 83.4 | **81.0** |
| R1-Distill-Qwen-14B | 64.9 | 77.5 | 87.0 | 58.0 | 51.0 | 51.0 | 81.0 | 76.6 | 93.9 | 72.6 |
| R1-Distill-Qwen-32B | 65.5 | 76.3 | 87.6 | 59.8 | 52.7 | 50.9 | 82.4 | **80.5** | 94.3 | 72.5 |
| QwenLong-L1-32B | 70.1 | 80.7 | 89.1 | 65.2 | 58.6 | 56.7 | **84.1** | 78.5 | **95.2** | 78.6 |
| **LoongRL-14B** | **74.2** | **82.2** | **93.3** | **67.5** | **63.4** | **64.5** | 80.7 | **80.5** | 83.2 | 78.4 |

## Repository Structure

```
LoongRL/
├── verl/                    # Customized veRL framework (see verl/README.md)
│   ├── examples/
│   │   ├── data_preprocess/     # 21 dataset preprocessing scripts
│   │   ├── grpo_trainer/         # 20+ GRPO configurations
│   │   ├── ppo_trainer/          # PPO configurations
│   │   └── sft/                  # Supervised fine-tuning
│   ├── verl/
│   │   ├── trainer/              # Training entry points
│   │   ├── workers/              # FSDP/Megatron workers
│   │   └── utils/reward_score/   # Reward function implementations
│   └── scripts/                  # Installation scripts
├── patches/                 # AMD/ROCm patches for vLLM, SGLang, Ray
└── install_mi300.sh        # AMD MI300X installation script
```

## Installation

### Quick Start (AMD MI300X)

```bash
git clone https://github.com/rStar-RL/rStar-RL.git
cd rStar-RL
bash install_mi300.sh  # Requires ROCm 6.2, PyTorch 2.6.0, Python 3.9
```

The installation script sets up:
- tensordict
- vLLM 0.7.3 (patched for ROCm)
- aiter (AMD GPU support)
- SGLang 0.4.4.post1 (patched)
- AMD-specific patches for Ray and torchao

### Manual Installation

#### Environment Setup

```bash
# Create conda environment
conda create -y -n rstar python=3.9
conda activate rstar

# Install tensordict
git clone https://github.com/pytorch/tensordict.git
cd tensordict
pip install .
cd ..
```

#### Inference Engine: SGLang (Recommended for AMD)

SGLang provides better rollout performance compared to vLLM on AMD MI300X.

```bash
# Install patched vLLM v0.7.3 to avoid dependency conflicts
git clone https://github.com/rStar-RL/rStar-RL.git
bash rStar-RL/patches/vllm_patch/install_v0_7_3.sh

# Install SGLang v0.4.4.post1 with patches
git clone -b v0.4.4.post1 https://github.com/sgl-project/sglang.git
cp rStar-RL/patches/sglang_patch/pyproject.toml sglang/python/
cp rStar-RL/patches/sglang_patch/scheduler.py sglang/python/sglang/srt/managers/
cp rStar-RL/patches/sglang_patch/custom_all_reduce.py sglang/python/sglang/srt/distributed/device_communicators/

cd sglang/sgl-kernel
python setup_rocm.py install
cd ..
pip install -e "python[all_hip]"
cd ..

# Install aiter (required by SGLang on ROCm)
git clone https://github.com/ROCm/aiter.git
cd aiter
git checkout e70ee4d948fd8455e4d665ebcc6fa2654bad6137
git submodule update --init --recursive
PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py develop
cd ..

# Patch torchao if using rocm-torch 2.6 docker image
python rStar-RL/patches/torchao_patch/downgrade_version_mi300.py
```

#### Alternative: vLLM v0.6.3

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.3
export CCACHE_DIR=/scratch/ccache  # If encountering permission issues
python setup.py develop
cd ..
```

#### Install veRL

```bash
cd rStar-RL/verl
pip install -e .
cd ../..

# Patch Ray for AMD device support
bash rStar-RL/patches/ray_patch/patch.sh
```

### NVIDIA A100 Installation

```bash
git clone https://github.com/rStar-RL/rStar-RL.git
cd rStar-RL
bash verl/scripts/install_a100x8.sh
```

## Quick Start

### 1. Data Preprocessing

Prepare datasets with reward-compatible formatting. See [verl/examples/data_preprocess/](verl/examples/data_preprocess/) for all preprocessing scripts.

**Long Context QA (RULER, NIAH)**:
```bash
# RULER benchmark (needle-in-haystack)
python verl/examples/data_preprocess/ruler_niah_dataset_system.py \
    --data_source <path> --output_path <output>

# General long context QA
python verl/examples/data_preprocess/longcontextqa_like_dataset_system.py \
    --data_source <path> --output_path <output>

# Sentence-level needle tasks
python verl/examples/data_preprocess/sentence_needle_dataset_system.py \
    --data_source <path> --output_path <output>
```

**Math Datasets (GSM8K, MATH, MathQA, AIME)**:
```bash
# Math with system prompts
python verl/examples/data_preprocess/math_like_dataset_system.py \
    --data_source <path> --output_path <output>

# MathQA specific
python verl/examples/data_preprocess/mathqa_dataset_system.py \
    --data_source <path> --output_path <output>
```

### 2. Launch Reward Server (Optional)

For code generation tasks requiring execution-based rewards:

```bash
# Create server environment (Python 3.10+)
conda create -y -n server python=3.10
conda activate server
sudo apt-get update && sudo apt-get install redis
pip install redis "fastapi[standard]"

# Setup code-judge server
git clone https://github.com/0xWJ/code-judge.git
cd code-judge
pip install -r requirements.txt

# Start Redis
redis-server --daemonize yes

# Start server
export REDIS_URI="redis://localhost:6379"
REDIS_URI=$REDIS_URI RUN_WORKERS=0 fastapi run --workers 4 app/main.py

# Start workers (in another terminal)
REDIS_URI=$REDIS_URI MAX_WORKERS=64 python run_workers.py
```

**Benefits of remote reward server**:
- Independent Python environment from training
- Exception handling isolation
- Easier performance optimization
- Supports distributed deployment for high-throughput advantage calculation

**Multi-node deployment** (when advantage computation becomes bottleneck):
```bash
# Master node
REDIS_URI=<REMOTE_REDIS_INFO> RUN_WORKERS=0 fastapi run --workers 32 app/main.py --host 0.0.0.0

# Worker nodes
REDIS_URI=<REMOTE_REDIS_INFO> python run_workers.py
```

### 3. Multi-node Ray Cluster

```bash
# Head node (node-0)
ray start --head --port 6379

# Worker nodes
ray start --address="<node-0-ip>:6379"

# Verify cluster
ray status
```

### 4. Run GRPO Training

**Long Context QA**:
```bash
# Qwen2-7B with sequence balancing
bash verl/examples/grpo_trainer/run_qwen2-7b_seq_balance_longcontext.sh

# Llama 3.1-8B
bash verl/examples/grpo_trainer/run_llama31-8b_seq_balance_longcontext.sh
```

**Math Reasoning**:
```bash
# Qwen2.5-32B on mixed math datasets (DeepScale-R + OpenR1)
bash verl/examples/grpo_trainer/run_qwen2.5-32b_math-mix1.sh

# Qwen2-7B math
bash verl/examples/grpo_trainer/run_qwen2-7b_math.sh
```

**Custom GRPO Configuration**:
```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=<train_parquet> \
    data.val_files=<val_parquet> \
    data.train_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=<model_path> \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    reward_model.reward_manager=longcontext_qa \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.total_epochs=20
```

See [verl/README.md](verl/README.md) for detailed GRPO configurations and more algorithms (PPO, PRIME, ReMax, RLOO).

## Training Entry Points

```bash
# GRPO/PPO training
python -m verl.trainer.main_ppo [algorithm.adv_estimator=grpo]

# Supervised fine-tuning
python -m verl.trainer.fsdp_sft_trainer

# Evaluation
python -m verl.trainer.main_eval

# Generation
python -m verl.trainer.main_generation
```

## Environment Variables

```bash
# AMD GPUs
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=0

# Logging
export WANDB_PROJECT=<project_name>
export WANDB_API_KEY=<your_key>
```

## Troubleshooting

**OOM Errors**: Enable `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True` or reduce `actor_rollout_ref.rollout.n` (group size).

**vLLM Issues**: Adjust `actor_rollout_ref.rollout.gpu_memory_utilization=0.5-0.75`, increase `swap_space`, or switch to SGLang with `actor_rollout_ref.rollout.name=sglang`.

**Long Context**: For sequences >8K tokens, enable sequence packing and adjust `data.max_prompt_length` based on GPU memory.

**Multi-node**: Verify Ray cluster with `ray status`, ensure `trainer.nnodes` matches actual cluster size.

## Available Reward Functions

- **Long Context**: `longcontext_qa`, `longcontext_choice`, `sentence_needle`, `ruler_multi`
- **Math**: `math` (boxed answer), `math_verify`, `gsm8k`, `mathqa_choice`, `prime` (process reward)
- **Code**: `code_server` (execution-based)
- **General**: `docmath`, `docqa`

## Citation

If you use LoongRL in your research, please cite our paper:

```bibtex
@misc{wang2025loongrlreinforcementlearningadvanced,
      title={LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts}, 
      author={Siyuan Wang and Gaokai Zhang and Li Lyna Zhang and Ning Shang and Fan Yang and Dongyao Chen and Mao Yang},
      year={2025},
      eprint={2510.19363},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.19363}, 
}
```

## Acknowledgement

This project is based on [veRL (HybridFlow)](https://github.com/volcengine/verl) by the ByteDance Seed team. veRL is inspired by Nemo-Aligner, DeepSpeed-Chat, and OpenRLHF. See their original documentation at [verl/README.md](verl/README.md).
