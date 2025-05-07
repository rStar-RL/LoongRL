dataset_folder="/mnt/longcontext/models/siyuan/test_code/longcontext_syth/longbench1/"
dataset_names=("hotpotqa" "musique" "2wikimqa")
for dataset_name in "${dataset_names[@]}"; do
    echo "Processing dataset: $dataset_name"
    python examples/data_preprocess/longcontextqa_like_dataset_system.py \
        --data_source "${dataset_folder}${dataset_name}.jsonl" \
        --local_dir "/mnt/longcontext/models/siyuan/rl_datasets/rl_three/${dataset_name}_valid" 
done