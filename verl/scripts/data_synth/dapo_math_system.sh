n_samples=5000
dataset_path=/mnt/longcontext/models/siyuan/rl_datasets/DAPO-Math-17k.jsonl

python examples/data_preprocess/dapo17k_dataset_system.py --data_source $dataset_path --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/DAPO-Math-17k_$n_samples  --start_index 0 --end_index $n_samples