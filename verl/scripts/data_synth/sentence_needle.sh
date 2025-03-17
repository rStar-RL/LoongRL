# n_samples=5000
n_samples_list=(1000 3000 5000 10000)
seq_lens=(4096 8192 16384 32768)
datasets=("/mnt/longcontext/models/siyuan/test_code/longcontext_syth/books3-hf")
for n_samples in "${n_samples_list[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seq_len in "${seq_lens[@]}"; do
            echo "Generating dataset: $dataset with sequence length: $seq_len"
            start_index=0
            end_index=$n_samples
            echo "start_index: $start_index, end_index: $end_index"
            python examples/data_preprocess/sentence_needle_dataset.py --data_source $dataset/sentence_needle_llama_${seq_len}_10000samples.jsonl  --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/sentence_needle_${n_samples}_start_index_${start_index}_end_index_${end_index}_seq_${seq_len} --start_index $start_index --end_index $end_index 
        done
    done
done