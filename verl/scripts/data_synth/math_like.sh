
n_samples=5000
dataset_path="/mnt/longcontext/models/siyuan/rl_datasets/merged_data_deepscaler_openr1_130k.json"

python examples/data_preprocess/math_like_dataset_no_system.py --data_source $dataset_path --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/merged_data_deepscaler_openr1_130k_$n_samples  --start_index 0 --end_index $n_samples