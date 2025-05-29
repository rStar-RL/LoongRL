import os
import datasets
from pathlib import Path

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source')
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=-1, type=int)

    args = parser.parse_args()

    dataset = datasets.Dataset.from_json(args.data_source)
    if args.end_index == -1:
        args.end_index = len(dataset)
    dataset = dataset.select(range(args.start_index, args.end_index))
    data_source = Path(args.data_source).stem

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            input = example.pop('Problem')
            option_list = example.pop('options')
            assert len(option_list) == 5, f"options length is {len(option_list)}"
            options = f"(A) {option_list[0]}\n(B) {option_list[1]}\n(C) {option_list[2]}\n(D) {option_list[3]}\n(E) {option_list[4]}"
            # question = example.pop('question')
            

            solution = example.pop('correct')
            data = {
                "data_source": f"custom_mathqa_choice_{data_source}",
                "prompt": [
                    {
                    "role": "user",
                    "content": f"Question: {input}\nChoices:{options}"
                    # "content": question,
                }],
                "ability": "math_qa_choices",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = dataset.map(function=make_map_fn('test'), with_indices=True)
    # sample the first 20 examples of test dataset for validation
    valid_dataset = test_dataset.select(range(20))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    valid_dataset.to_parquet(os.path.join(local_dir, 'valid.parquet'))


    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
