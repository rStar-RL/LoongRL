n_samples_list=(2500 5000)
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
            # total_samples are different for different datasets
            # hotpotqa: 24287
            # musique: 8517
            # 2wikimqa: 39679
            if [ "$dataset" == "hotpotqa" ]; then
                total_samples=24287
            elif [ "$dataset" == "musique" ]; then
                total_samples=8517
            elif [ "$dataset" == "2wikimqa" ]; then
                total_samples=39679
            fi

            # Loop over start_index with a stride equal to n_samples
            while [ $start_index -lt $total_samples ]; do
                # Ensure the end_index does not exceed the total number of samples
                end_index=$((start_index + n_samples))
                if [ $end_index -gt $total_samples ]; then
                    end_index=$total_samples
                fi

                echo "Generating dataset: $dataset with sequence length: $seq_len, start_index: $start_index, end_index: $end_index"
                #input file like hotpotqa_qwen_filtered-num_sample_24287-max_seq_32768.jsonl
                input_file="${dataset}_qwen_filtered_core_gaussian-num_sample_${total_samples}-max_seq_$seq_len.jsonl"
                echo "input file: /mnt/longcontext/models/siyuan/test_code/longcontext_syth/$dataset/${input_file}"
                python examples/data_preprocess/longcontextqa_like_dataset_system.py \
                    --data_source /mnt/longcontext/models/siyuan/test_code/longcontext_syth/$dataset/$input_file \
                    --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/${dataset}_qwen_filtered_core_gaussian_start_idx${start_index}_end_idx${end_index}_seq${seq_len} \
                    --start_index $start_index \
                    --end_index $end_index &

                # Increment start_index by n_samples
                start_index=$((start_index + n_samples))
                break
            done
        done
        wait
    done
done