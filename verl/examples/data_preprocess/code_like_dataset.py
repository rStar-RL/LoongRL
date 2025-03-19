import os
import datasets
from pathlib import Path

from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source')
    parser.add_argument('--local_dir', default='~/data/code')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    dataset = datasets.Dataset.from_json(args.data_source)
    data_source = Path(args.data_source).stem

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            prompt = example.pop('prompt')
            # for code/math mix training, apply specific prompt for code role
            # prompt = [p if p['role'] != 'user' else {'role': 'code', 'content': p['content']} for p in prompt]
            reward_model = example.pop('reward_model')
            data = {
                "data_source": f"custom_code_{data_source}",
                "prompt": prompt,
                "ability": "code",
                "reward_model": reward_model,
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_json(os.path.join(local_dir, 'train.json'))
    test_dataset.to_json(os.path.join(local_dir, 'test.json'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
