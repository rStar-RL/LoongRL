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
    # NOTE
    # the sentence needle raw dataset needs reordering
    # since the needle data with the same depth percent are clustered together
    num_books = 1000
    num_percents = 10
    assert num_books * num_percents == len(dataset), "dataset length should be the size of num_books * num_percents"
    # the original dataset is in the shape of dataset[percent][book].squeeze()
    # reorder the dataset to dataset[book][percent].squeeze()
    # Build the new index mapping
    new_indices = []
    for i in range(len(dataset)):
        b = i // num_books    # which book
        p = i % num_percents     # which percent
        old_index = p * num_books + b
        new_indices.append(old_index)

    # Use Dataset.select(...) to produce the re-ordered dataset
    dataset = dataset.select(new_indices)

    # Check: first n_percents elements of reordered_dataset should all be book 0 but different percents

    if args.end_index == -1:
        args.end_index = len(dataset)

    # 1) Split out the training set
    train_dataset = dataset.select(range(args.start_index, args.end_index))
    train_size = len(train_dataset)  # Number of training samples

    # 2) Calculate leftover indices
    total_size = len(dataset)        # Total number of samples in the dataset
    all_indices = set(range(total_size))
    train_indices = set(range(args.start_index, args.end_index))

    # Remove indices that are used for the training set
    # The remaining indices represent unused data
    leftover_indices = sorted(all_indices - train_indices)

    # 3) Check if leftover data is sufficient
    leftover_size = len(leftover_indices)
    if leftover_size == 0:
        print("Warning: No leftover samples to use for validation.")
        # If no leftover data, use the training set as validation
        valid_indices = train_indices
    elif leftover_size < train_size:
        print(
            f"Warning: Only {leftover_size} leftover samples, "
            f"which is less than the training size {train_size}."
        )
        # If the leftover is not enough, use all of it for validation
        valid_indices = leftover_indices
    else:
        # Otherwise, select the same number of leftover samples as training size
        valid_indices = leftover_indices[:train_size]

    # 4) Construct the validation set
    valid_dataset = dataset.select(valid_indices)

    # Display the result
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")




    data_source = Path(args.data_source).stem

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            input = example.pop('input')
            context = example.pop('context')
            solution = example.pop('answer')
            data = {
                "data_source": f"custom_sentence_needle_{data_source}",
                "prompt": [
                    {
                    "role": "user",
                    "content": f"The following are given passages.\n{context}\n\nQuestion: {input}"
                    # "content": question,
                }],
                "ability": "sentence_needle",
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

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = valid_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
