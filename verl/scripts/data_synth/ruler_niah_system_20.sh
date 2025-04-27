all_data_folder=/mnt/longcontext/data/gaozhang/book_20key_10k

n_samples_list=(1024 3072 5120 10240)
seq_lens=("4k" "8k" "16k" "32k")
datasets=("/mnt/longcontext/data/gaozhang/book_20key_10k" "/mnt/longcontext/data/gaozhang/book_20value_10k")
# datasets=("/mnt/longcontext/data/gaozhang/book_20value_10k")


for n_samples in "${n_samples_list[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seq_len in "${seq_lens[@]}"; do
            start_index=0
            total_samples=10240  
            
            # Loop over start_index with a stride equal to n_samples
            while [ $start_index -lt $total_samples ]; do
                # Ensure the end_index does not exceed the total number of samples
                end_index=$((start_index + n_samples))
                if [ $end_index -gt $total_samples ]; then
                    end_index=$total_samples
                fi

                echo "Generating dataset: $dataset with sequence length: $seq_len, start_index: $start_index, end_index: $end_index"
                echo "save to /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/$(basename $dataset)_${n_samples}_start_index_${start_index}_end_index_${end_index}"

                # if key in data, then the datasource folder should be ${dataset}/book_20_key_10k
                # if value in data, then the datasource folder should be ${dataset}/book_20_value_10k
                if [[ $dataset == *"key"* ]]; then
                    data_source=${dataset}/book_multikey_${seq_len}/book_multikey_${seq_len}_20key_10k.jsonl
                elif [[ $dataset == *"value"* ]]; then
                    data_source=${dataset}/book_multivalue_${seq_len}/book_multivalue_${seq_len}_20value_10k.jsonl
                else
                    echo "Unknown dataset type"
                    exit 1
                fi

                # assert the data_source exists
                if [ ! -f $data_source ]; then
                    echo "Data source file does not exist: $data_source"
                    # echo something in big red to warn the user
                    echo -e "\033[31mData source file does not exist: $data_source\033[0m"
                    exit 1
                fi
                # output a log to tell the user the data_source exists
                # echo something in green to tell the user the data_source exists
                echo -e "\033[32mData source file exists: $data_source\033[0m"
                echo "Data source file exists: $data_source"
                echo "local_dir: /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/$(basename $dataset)_${n_samples}_start_index_${start_index}_end_index_${end_index}_seqlen${seq_len}"
                
                # Run the Python script with the appropriate parameters
                python examples/data_preprocess/ruler_niah_dataset_system.py \
                    --data_source $data_source \
                    --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/$(basename $dataset)_${n_samples}_start_index_${start_index}_end_index_${end_index}_seqlen${seq_len} \
                    --start_index $start_index \
                    --end_index $end_index &

                # Increment start_index by n_samples
                start_index=$((start_index + n_samples))
            done
        done
    done
done