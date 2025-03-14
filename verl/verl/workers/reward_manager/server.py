# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score.code_server import extract_solution, validate_response_structure, batch_judge
import time
from collections import defaultdict
import random
from typing import List
from dataclasses import dataclass


@dataclass
class CodeRollout:
    """The code info.
    """

    batch_idx: int
    solution: str
    format_score: float

    valid_response_len: int
    debug_info: List[str]


class ServerRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        # when the batch size is large enough, all test cases will be judged for each submission
        self.batch_size = 4
        self.format_reward = 1.0
        self.print_num = 128

    def __call__(self, data: DataProto):

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        test_num = defaultdict(int)
        prompt2groundtruth = {}
        prompt2rollouts = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt = self.tokenizer.decode(valid_prompt_ids)

            response_ids = data_item.batch['responses']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            test_num[len(ground_truth['input'])] += 1

            debug_str = []
            debug_str.append("\n" + "="*80)
            debug_str.append(" Processing New Sample ".center(80, '='))
            # note: answer_text may be None if extract_solution fails
            answer_text, processed_str, question_str = extract_solution(sequences_str)

            if prompt in prompt2groundtruth:
                assert prompt2groundtruth[prompt] == ground_truth, f"Ground truth mismatch: {prompt2groundtruth[prompt]} != {ground_truth}"
            else:
                prompt2groundtruth[prompt] = ground_truth

            debug_str.append(f"\n[Question]\n{question_str}")
            debug_str.append(f"\n[Model Response]\n{processed_str}")
            format_correct, format_info = validate_response_structure(processed_str)
            debug_str.extend(format_info)
            format_score = self.format_reward if format_correct else -abs(self.format_reward)
            debug_str.append(f" Final Score ".center(80, '-'))
            debug_str.append(f"  Format Score: {format_score}")

            current_rollout = CodeRollout(
                batch_idx=i,
                solution=answer_text,
                format_score=format_score,
                valid_response_len=valid_response_length,
                debug_info=debug_str
            )
            prompt2rollouts[prompt].append(current_rollout)

        test_num = dict(sorted(test_num.items()))
        print('SeverRewardManager: test num info', test_num)

        print('SeverRewardManager: judge start')
        start_time = time.time()
        # Set this value carefully
        # if there are N cpu cores available to judge problems, the max_workers
        # should less than N / max(rollout_num, batch_size)
        # also note that when the concurrency degree is high, solutions whose execution
        # time is close to the timeout may receive lower scores.
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() // 4) as executor:
            futures = []
            for prompt, rollouts in prompt2rollouts.items():
                solutions = [rollout.solution for rollout in rollouts]
                ground_truth = prompt2groundtruth[prompt]
                futures.append((prompt, executor.submit(batch_judge, solutions, ground_truth, self.batch_size)))

            for prompt, future in futures:
                scores = future.result()
                for score, rollout in zip(scores, prompt2rollouts[prompt]):
                    i = rollout.batch_idx
                    answer_score = 5.0 * score
                    total_score = answer_score + rollout.format_score
                    reward_tensor[i, rollout.valid_response_len - 1] = total_score
                    rollout.debug_info.append(f"  Answer Score: {answer_score}")
                    rollout.debug_info.append(f"  Total Score: {total_score}")

        end_time = time.time()
        print('SeverRewardManager: judge time:', end_time - start_time, 's')
        # random select a rollout trace of each problem and print its debug info
        for prompt, rollouts in prompt2rollouts.items():
            random.shuffle(rollouts)
            print("\n".join(rollouts[0].debug_info))
            print("="*80)
        return reward_tensor
