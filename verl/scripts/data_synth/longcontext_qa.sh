n_samples=10000
datasets=(
    "hotpotqa"
    "musique"
    "2wikimqa"
)
seq_lens=(8192 16384 32768)

for dataset in "${datasets[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        echo "Generating dataset: $dataset with sequence length: $seq_len"
        python examples/data_preprocess/longcontextqa_like_dataset.py --data_source /mnt/longcontext/models/siyuan/test_code/longcontext_syth/$dataset/validation-llama-3.1-8B-instruct-num_sample_10000-max_seq_$seq_len.jsonl --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/${dataset}${n_samples}_seq${seq_len} --start_index 0 --end_index $n_samples 
    done
done