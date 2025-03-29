
start_idx=0
end_idx=5000

python examples/data_preprocess/mathqa_dataset_system.py --data_source /mnt/longcontext/models/siyuan/test_code/longcontext_syth/mathqa/math_qa_train_formatted.jsonl \
    --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/mathqa_choice_start_idx${start_idx}_end_idx${end_idx} \
    --start_index $start_idx \
    --end_index $end_idx 


start_idx=5000
end_idx=10000

python examples/data_preprocess/mathqa_dataset_system.py --data_source /mnt/longcontext/models/siyuan/test_code/longcontext_syth/mathqa/math_qa_train_formatted.jsonl \
    --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/mathqa_choice_start_idx${start_idx}_end_idx${end_idx} \
    --start_index $start_idx \
    --end_index $end_idx 