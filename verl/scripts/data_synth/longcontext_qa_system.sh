n_samples_list=(1000 3000 5000 10000)
datasets=(
    "hotpotqa"
    "musique"
    "2wikimqa"
)
seq_lens=(8192 16384 32768)

for n_samples in "${n_samples_list[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seq_len in "${seq_lens[@]}"; do
            start_index=0
            total_samples=10000  # This should be the total number of samples available, adjust as needed

            # Loop over start_index with a stride equal to n_samples
            while [ $start_index -lt $total_samples ]; do
                # Ensure the end_index does not exceed the total number of samples
                end_index=$((start_index + n_samples))
                if [ $end_index -gt $total_samples ]; then
                    end_index=$total_samples
                fi

                echo "Generating dataset: $dataset with sequence length: $seq_len, start_index: $start_index, end_index: $end_index"
                python examples/data_preprocess/longcontextqa_like_dataset_system.py \
                    --data_source /mnt/longcontext/models/siyuan/test_code/longcontext_syth/$dataset/validation-llama-3.1-8B-instruct-num_sample_10000-max_seq_$seq_len.jsonl \
                    --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/${dataset}${n_samples}_seq${seq_len} \
                    --start_index $start_index \
                    --end_index $end_index

                # Increment start_index by n_samples
                start_index=$((start_index + n_samples))
            done
        done
    done
done