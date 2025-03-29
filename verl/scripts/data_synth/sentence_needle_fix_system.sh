# n_samples=5000
n_samples_list=(1000 3000 5000 10000)
seq_lens=(4096 8192 16384 32768)
datasets=("/mnt/longcontext/models/siyuan/test_code/longcontext_syth/books3-hf")

# Set the total number of available samples (adjust as needed)
total_samples=10000

for n_samples in "${n_samples_list[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seq_len in "${seq_lens[@]}"; do
            start_index=0

            # Loop over start_index with a stride equal to n_samples
            while [ $start_index -lt $total_samples ]; do
                # Ensure the end_index does not exceed the total number of samples
                end_index=$((start_index + n_samples))
                if [ $end_index -gt $total_samples ]; then
                    end_index=$total_samples
                fi

                echo "Generating dataset: $dataset with sequence length: $seq_len, start_index: $start_index, end_index: $end_index"
                echo "save to /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/sentence_needle_fix_${n_samples}_start_index_${start_index}_end_index_${end_index}_seq_${seq_len}"

                # Run the Python script with the appropriate parameters
                python examples/data_preprocess/sentence_needle_dataset_system.py \
                    --data_source $dataset/sentence_needle_llama_${seq_len}_10000samples.jsonl \
                    --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/sentence_needle_fix_${n_samples}_start_index_${start_index}_end_index_${end_index}_seq_${seq_len} \
                    --start_index $start_index \
                    --end_index $end_index

                # Increment start_index by n_samples
                start_index=$((start_index + n_samples))
            done
        done
    done
done