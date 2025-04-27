import os
import datasets
from pathlib import Path
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source")
    parser.add_argument("--local_dir", default="~/data/math")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument("--end_index", default=-1, type=int)

    args = parser.parse_args()
    # load the dataset with jsonl_load first
    dataset_content = []
    with open(args.data_source, "r") as f:
        dataset_content = [json.loads(line) for line in f]
    unnested_dataset = []
    for data in dataset_content:
        import pdb

        try:
            answer = data["final_questions"][0].get("options", {}).get(
                "answer"
            ) or data["final_questions"][0].get("answer")
            if answer is None:
                raise ValueError(f"Answer is None for data: {data}")
            unnested_dataset.append(
                {
                    "whole_segment": data["whole_segment"],
                    "question": data["final_questions"][0]["question"],
                    "A": data["final_questions"][0]["options"]["A"],
                    "B": data["final_questions"][0]["options"]["B"],
                    "C": data["final_questions"][0]["options"]["C"],
                    "D": data["final_questions"][0]["options"]["D"],
                    "answer": answer,
                }
            )
        except Exception as e:

            import pdb

            pdb.set_trace()
    print(len(unnested_dataset))
    # dataset = datasets.Dataset.from_json(args.data_source)
    dataset = datasets.Dataset.from_list(unnested_dataset)
    if args.end_index == -1:
        args.end_index = len(dataset)

    # 1) Split out the training set
    train_dataset = dataset.select(range(args.start_index, args.end_index))
    train_size = len(train_dataset)  # Number of training samples

    # 2) Calculate leftover indices
    total_size = len(dataset)  # Total number of samples in the dataset
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
            context = example.pop("whole_segment")
            context = "\n".join(context)
            # question = example.pop('question')
            # question = input["question"]
            # choices = input["options"]
            # answer = input["answer"]
            question = example.pop("question")
            choices = {
                "A": example.pop("A"),
                "B": example.pop("B"),
                "C": example.pop("C"),
                "D": example.pop("D"),
            }
            answer = example.pop("answer")
            # shuffle the choices and change the answer correspondingly
            # Original dictionary and answer
            # choices = {"A": "choiceA", "B": "choiceB", "C": "choiceC", "D": "choiceD"}
            # answer = "A"
            import random

            # Set the random seed for reproducibility
            random.seed(42)

            # Get the values from the dictionary and shuffle them
            values = list(choices.values())
            random.shuffle(values)  # Shuffle the values

            # Create a new dictionary with the same keys, but shuffled values
            new_choices = {
                key: value for key, value in zip(choices.keys(), values)
            }

            # Find the new answer based on the shuffled choices
            if len(answer) > 1:
                # import pdb

                # pdb.set_trace()
                print(f"answer is not a single character: {answer}")
                answer = answer[0].upper()
            new_answer = next(
                key
                for key, value in new_choices.items()
                if value == choices[answer]
            )

            # Output the results
            # print("New Choices:", new_choices)
            # print("New Answer:", new_answer)

            # # Output the results
            # print("New Choices:", new_choices)
            # print("New Answer:", new_answer)

            # format the question and the choices as question.\nA.dsad\nB.asdasd

            formatted_question = (
                question
                + "\nChoices:\n"
                + "\n".join(
                    [f"({key}) {value}" for key, value in new_choices.items()]
                )
            )

            # solution = example.pop("answers")
            data = {
                "data_source": f"custom_longcontextchoice_{data_source}",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Please read the following text and answer the questions below.\n{context}\n\nQuestion: {formatted_question}",
                        # "content": question,
                    }
                ],
                "ability": "longcontext_choice",
                "reward_model": {"style": "rule", "ground_truth": new_answer},
                "extra_info": {"split": split, "index": idx},
            }
            import pdb

            # pdb.set_trace()
            return data

        return process_fn

    train_dataset = dataset.map(
        function=make_map_fn("train"), with_indices=True
    )
    test_dataset = dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
