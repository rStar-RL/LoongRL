data_path="/home/aiscuser/Red_split_long"
dataset_name="arxiv_middle_chars_43008_65536_segments"
l=7
min_tokens=2048
suffix="no_context"
overall_start_idx=0
overall_end_idx=10000
stride=1000

# Construct the save file path using string concatenation
save_file_path="${data_path}/${dataset_name}_l${l}_min_tokens${min_tokens}_start_idx_${overall_start_idx}_end_idx_${overall_end_idx}_${suffix}_overall_question_merged.jsonl"

# Output the save file path to check
echo "Save file path: $save_file_path"
# assert this file exists
if [ ! -f $save_file_path ]; then
    echo "File does not exist: $save_file_path"
    exit 1
fi
echo "File exists: $save_file_path"
python examples/data_preprocess/longcontext_choice_system.py \
    --data_source $save_file_path \
    --local_dir /mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/${dataset_name}_l${l}_min_tokens${min_tokens}_start_idx_${overall_start_idx}_end_idx_${overall_end_idx}_${suffix} \
    --start_index $overall_start_idx \
    --end_index $overall_end_idx 