n_samples_list=(1000 3000 5000)
datasets=(
    "musique"
)
seq_lens=(8192 16384 32768)

for n_samples in "${n_samples_list[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seq_len in "${seq_lens[@]}"; do
            start_index=0
            total_samples=5000  # This should be the total number of samples available, adjust as needed

            # Loop over start_index with a stride equal to n_samples
            while [ $start_index -lt $total_samples ]; do
                # Ensure the end_index does not exceed the total number of samples
                end_index=$((start_index + n_samples))
                if [ $end_index -gt $total_samples ]; then
                    end_index=$total_samples
                fi

                echo "Generating dataset: $dataset with sequence length: $seq_len, start_index: $start_index, end_index: $end_index"
                # /mnt/longcontext/models/siyuan/test_code/longcontext_syth/musique/hard-llama-3.1-8B-instruct-num_sample_5000-max_seq_8192.jsonl 
                python examples/data_preprocess/longcontextqa_like_dataset_system.py \
                    --data_source /mnt/longcontext/models/siyuan/test_code/longcontext_syth/$dataset/hard-llama-3.1-8B-instruct-num_sample_5000-max_seq_$seq_len.jsonl \
                    --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/hard_${dataset}_start_idx${start_index}_end_idx${end_index}_seq${seq_len} \
                    --start_index $start_index \
                    --end_index $end_index

                # Increment start_index by n_samples
                start_index=$((start_index + n_samples))
            done
        done
    done
done