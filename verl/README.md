# veRL: Reinforcement Learning Framework for LLM Reasoning

This repository contains a customized version of [veRL (HybridFlow)](https://github.com/volcengine/verl) for large-scale reinforcement learning on long-context question answering and mathematical reasoning tasks. The framework is optimized for **Group Relative Policy Optimization (GRPO)** and supports distributed training on AMD MI300X and NVIDIA GPUs.

## Overview

This framework provides:

- **GRPO Implementation**: Critic-free reinforcement learning with group-wise advantage estimation
- **Long Context QA**: RULER, NIAH (needle-in-haystack), and extended context reasoning tasks
- **Math Reasoning**: Verifiable reward functions for GSM8K, MATH, MathQA, AIME datasets
- **AMD MI300X Support**: ROCm-optimized with vLLM 0.7.3 and SGLang 0.4.4.post1
- **Multi-node Training**: Distributed GRPO on 7B-32B models across GPU clusters

## Installation

### AMD MI300X GPUs (ROCm 6.2)

```bash
git clone https://github.com/rStar-RL/rStar-RL.git
cd rStar-RL
bash install_mi300.sh  # Requires PyTorch 2.6.0, Python 3.9
```

This installs: tensordict, vLLM 0.7.3 (patched), aiter, SGLang 0.4.4.post1 (patched), and AMD-specific patches for Ray/torchao.

### NVIDIA A100 GPUs

```bash
bash verl/scripts/install_a100x8.sh
```

## Quick Start

### 1. Data Preprocessing

Prepare datasets with reward-compatible formatting:

**Long Context QA**:
```bash
# RULER benchmark (needle-in-haystack)
python examples/data_preprocess/ruler_niah_dataset_system.py \
    --data_source <path> --output_path <output>

python examples/data_preprocess/ruler_niah_dataset.py \
    --data_source <path> --output_path <output>

# General long context QA
python examples/data_preprocess/longcontextqa_like_dataset_system.py \
    --data_source <path> --output_path <output>

python examples/data_preprocess/longcontextqa_like_dataset.py \
    --data_source <path> --output_path <output>

# Needle-in-haystack variants
python examples/data_preprocess/longcontextqa_needle_like_dataset_system.py \
    --data_source <path> --output_path <output>

python examples/data_preprocess/longcontextqa_needle_like_dataset_system_1line.py \
    --data_source <path> --output_path <output>

# Multiple choice format
python examples/data_preprocess/longcontext_choice_system.py \
    --data_source <path> --output_path <output>

# Sentence-level needle tasks
python examples/data_preprocess/sentence_needle_dataset_system.py \
    --data_source <path> --output_path <output>

python examples/data_preprocess/sentence_needle_dataset.py \
    --data_source <path> --output_path <output>
```

**Math Datasets** (GSM8K, MATH, MathQA, AIME):
```bash
# With system prompts
python examples/data_preprocess/math_like_dataset_system.py \
    --data_source <path> --output_path <output>

# Without system prompts
python examples/data_preprocess/math_like_dataset.py \
    --data_source <path> --output_path <output>

# MathQA specific
python examples/data_preprocess/mathqa_dataset_system.py \
    --data_source <path> --output_path <output>

# DAPO 17k dataset
python examples/data_preprocess/dapo17k_dataset_system.py \
    --data_source <path> --output_path <output>
```

See [examples/data_preprocess/](examples/data_preprocess/) for all 21 preprocessing scripts.

**Available reward functions**: `longcontext_qa`, `longcontext_choice`, `sentence_needle`, `ruler_multi`, `math` (boxed answer), `math_verify`, `gsm8k`, `mathqa_choice`, `prime` (process reward), `docmath`, `docqa`.

### 2. Multi-node Ray Cluster

```bash
# Head node
ray start --head --port 6379

# Worker nodes
ray start --address="<head_node_ip>:6379"
```

## GRPO Training

GRPO (Group Relative Policy Optimization) eliminates the critic network and computes advantages from grouped samples. All GRPO experiments use `algorithm.adv_estimator=grpo` with the unified PPO trainer.

### Core GRPO Parameters

```bash
algorithm.adv_estimator=grpo                    # Enable GRPO mode
actor_rollout_ref.rollout.n=16                  # Samples per prompt (group size)
actor_rollout_ref.rollout.temperature=1.0       # Sampling temperature
actor_rollout_ref.actor.use_kl_loss=True        # KL regularization
actor_rollout_ref.actor.kl_loss_coef=0.001      # KL coefficient
actor_rollout_ref.actor.kl_loss_type=low_var_kl # Low-variance KL estimator
reward_model.reward_manager=<type>              # longcontext_qa | ruler_multi | math | prime
trainer.critic_warmup=0                         # No critic needed for GRPO
```

### Long Context QA Experiments

**Qwen2-7B Long Context**:
```bash
bash examples/grpo_trainer/run_qwen2-7b_seq_balance_longcontext.sh
```

Configuration: Sequence balancing for efficient long-context processing.

**Llama 3.1-8B Long Context**:
```bash
bash examples/grpo_trainer/run_llama31-8b_seq_balance_longcontext.sh
```

Configuration: Optimized for extended context windows with sequence balancing.

### Math Reasoning Experiments

**Qwen2.5-32B on Mixed Math Datasets**:
```bash
# With KL regularization (kl_coef=0.0001)
bash examples/grpo_trainer/run_qwen2.5-32b_math-mix1.sh

# Without KL regularization
bash examples/grpo_trainer/run_qwen2.5-32b_math-mix1-nokl.sh
```

**Dataset**: DeepScale-R + OpenR1 (130K examples)
**Configuration**: 8 samples/prompt, temperature 0.6, PRIME process rewards, 2 nodes × 8 GPUs

**Qwen2.5-32B Math+Code (DAPO)**:
```bash
bash examples/grpo_trainer/run_qwen2.5-32b-ins_math-code-orz-dapo.sh
```

**Qwen2-7B Math Experiments**:
```bash
# Standard GRPO
bash examples/grpo_trainer/run_qwen2-7b_math.sh

# With sequence balancing
bash examples/grpo_trainer/run_qwen2-7b_seq_balance.sh

# Megatron-LM backend
bash examples/grpo_trainer/run_qwen2-7b_math_megatron.sh
```

### DeepSeek Models

```bash
# DeepSeek-7B base
bash examples/grpo_trainer/run_deepseek7b_llm.sh

# DeepSeek-7B math
bash examples/grpo_trainer/run_deepseek7b_llm_math.sh

# With sequence balancing
bash examples/grpo_trainer/run_deepseek7b_llm_seq_balance.sh

# Megatron backend
bash examples/grpo_trainer/run_deepseek7b_llm_megatron.sh
bash examples/grpo_trainer/run_deepseek7b_llm_math_megatron.sh
```

### Custom GRPO Configuration

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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    reward_model.reward_manager=longcontext_qa \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.total_epochs=20
```

See [examples/grpo_trainer/](examples/grpo_trainer/) for 20+ pre-configured experiments.

## Additional Algorithms

### PPO (Proximal Policy Optimization)

Standard PPO with critic networks:

```bash
# Math tasks
bash examples/ppo_trainer/run_qwen2.5-7b_math-orz.sh
bash examples/ppo_trainer/run_qwen2.5-32b_math-mix1.sh

# With SGLang inference
bash examples/ppo_trainer/run_qwen2.5-7b_math-orz-sglang.sh
bash examples/ppo_trainer/run_qwen2.5-32b-ins_math-code-orz-sglang.sh
```

See [examples/ppo_trainer/](examples/ppo_trainer/) for 20+ PPO configurations.

### PRIME (Process Reward Model)

Process-level reward guidance:

```bash
cd recipe/prime
bash run_prime_qwen.sh
```

### Other Trainers

- **ReMax**: [examples/remax_trainer/](examples/remax_trainer/)
- **RLOO**: [examples/rloo_trainer/](examples/rloo_trainer/)
- **SFT**: `bash examples/sft/gsm8k/run_qwen_05.sh`

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

Configuration via Hydra: `verl/verl/trainer/config/ppo_trainer.yaml`

## Framework Features

- **Training Backends**: FSDP (gradient/optimizer offload), Megatron-LM
- **Inference Engines**: vLLM 0.7.3, SGLang 0.4.4.post1, HuggingFace Transformers
- **Model Support**: Qwen2/2.5 (7B-32B), Llama 3.1, DeepSeek, Gemma2
- **RL Algorithms**: GRPO, PPO, PRIME, ReMax, RLOO
- **Task Support**: Long-context QA, RULER/NIAH benchmarks, math reasoning, code generation
- **Reward Functions**: Answer verification, process rewards, LLM-as-judge, execution-based
- **Optimizations**: Flash Attention 2, sequence packing, sequence parallelism (Ulysses), gradient checkpointing, LoRA
- **Distributed**: Multi-node Ray clusters, tensor parallelism, expert parallelism

## Repository Structure

```
verl/
├── examples/
│   ├── data_preprocess/     # 21 dataset preprocessing scripts
│   ├── grpo_trainer/         # 20+ GRPO configurations
│   ├── ppo_trainer/          # 20+ PPO configurations
│   ├── rloo_trainer/         # RLOO experiments
│   ├── remax_trainer/        # ReMax experiments
│   └── sft/                  # Supervised fine-tuning
├── recipe/
│   └── prime/                # PRIME algorithm (process rewards)
├── verl/
│   ├── trainer/              # main_ppo.py, fsdp_sft_trainer.py
│   ├── workers/              # FSDP/Megatron workers
│   ├── utils/reward_score/   # 16 reward function implementations
│   └── third_party/          # vLLM/SGLang integrations
└── scripts/                  # Installation scripts
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

**vLLM Issues**: Adjust `actor_rollout_ref.rollout.gpu_memory_utilization=0.5-0.75`, increase `swap_space`, or use SGLang backend with `actor_rollout_ref.rollout.name=sglang`.

**Multi-node Training**: Verify Ray cluster status with `ray status`, ensure `trainer.nnodes` matches actual cluster size.

**Long Context**: For sequences >8K tokens, enable sequence packing and adjust `data.max_prompt_length` based on GPU memory.

## Documentation

For detailed documentation on the base veRL framework:
- [veRL Documentation](https://verl.readthedocs.io/en/latest/)
- [HybridFlow Paper (EuroSys 2025)](https://arxiv.org/abs/2409.19256v2)

## Citation

If you use this framework, please cite:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}

@inproceedings{sheng2024nl2code,
  title     = {A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization},
  author    = {Guangming Sheng and Mrinal Anand and Jie M. Zhang and Alex Serban and Chuan Wu},
  booktitle = {NL2Code Workshop at ICSE},
  year      = {2024}
}
```

## Acknowledgement

This framework is based on [veRL (HybridFlow)](https://github.com/volcengine/verl) by the ByteDance Seed team. veRL is inspired by Nemo-Aligner, DeepSpeed-Chat, and OpenRLHF. The original veRL project is supported by Bytedance, Anyscale, LMSys.org, Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, and many contributors from the community.
