# n_samples=5000
n_samples_list=(1000 3000 5000 10000)
seq_lens=("4k" "8k" "16k" "32k" "64k" "128k")
datasets=("/mnt/longcontext/data/gaozhang/book-haystacked-ruler/multikey/multikey" "/mnt/longcontext/data/gaozhang/book-haystacked-ruler/multivalue/multivalue")
for n_samples in "${n_samples_list[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seq_len in "${seq_lens[@]}"; do
            echo "Generating dataset: $dataset with sequence length: $seq_len"
            start_index=0
            end_index=$n_samples
            echo "start_index: $start_index, end_index: $end_index"
            echo "save to /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/$(basename $dataset)_${n_samples}_start_index_${start_index}_end_index_${end_index}"
            python examples/data_preprocess/ruler_niah_dataset.py --data_source ${dataset}_${seq_len}/validation.jsonl  --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/$(basename $dataset)_${n_samples}_start_index_${start_index}_end_index_${end_index}_seqlen${seq_len} --start_index $start_index --end_index $end_index 
        done
    done
done