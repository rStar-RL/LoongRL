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
            input = example.pop('input')
            context = example.pop('context')
            # question = example.pop('question')
            question = f"The following are given passages.\n{context}\n\n Question: {input}"

            solution = example.pop('answers')
            data = {
                "data_source": f"custom_longcontextqa_{data_source}",
                "prompt": [
                    # {
                    #     "role": "system",
                    #     # 'content': '',
                    #     "content": "Answer the question based on the given passages following these steps: \n\nStart with a `<think>` and break down the question into key elements;\nAs you reason, use the marker `wait` to pause and reflect on details when necessary;\nProvide a clear, step-by-step explanation of your reasoning, ensuring each step is backed by the passages;\nEnd your response with a final line starting with `Answer:` followed by your answer inside \\boxed{ }.\nKeep your reasoning rigorous, precise, and succinct."
                    # },
                    {
                    "role": "user",
                    "content": f"Answer the question based on the given passages following these steps: \n\nStart with a `<think>` and break down the question into key elements;\nAs you reason, use the marker `wait` to pause and reflect on details when necessary;\nProvide a clear, step-by-step explanation of your reasoning, ensuring each step is backed by the passages;\nEnd your response with a final line starting with `Answer:` followed by your answer.\nKeep your reasoning rigorous, precise, and succinct.\n\nThe following are given passages.\n{context}\n\nQuestion: {input}"
                    # "content": question,
                }],
                "ability": "longcontext_qa",
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

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
