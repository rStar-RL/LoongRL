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
from verl.utils.reward_score.code_server import extract_solution, validate_response_structure, run
import time
from collections import defaultdict
import random


class ServerRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        # tune this value to get better performance
        self.batch_size = 5
        self.format_reward = 1.0
        self.print_num = 128

    def __call__(self, data: DataProto):

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        info = defaultdict(int)
        solution_strs, ground_truths, valid_response_lens = [], [], []
        answer_texts, format_scores, debug_info = [], [], []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            debug_str = []
            debug_str.append("\n" + "="*80)
            debug_str.append(" Processing New Sample ".center(80, '='))
            answer_text, processed_str, question_str = extract_solution(sequences_str)
            debug_str.append(f"\n[Question]\n{question_str}")
            debug_str.append(f"\n[Model Response]\n{processed_str}")
            format_correct, format_info = validate_response_structure(processed_str)
            debug_str.extend(format_info)
            format_score = self.format_reward if format_correct else -abs(self.format_reward)
            debug_str.append(f" Final Score ".center(80, '-'))
            debug_str.append(f"  Format Score: {format_score}")
            info[len(ground_truth['input'])] += 1

            solution_strs.append(sequences_str)
            ground_truths.append(ground_truth)
            valid_response_lens.append(valid_response_length)
            answer_texts.append(answer_text)
            format_scores.append(format_score)
            debug_info.append(debug_str)

        info = dict(sorted(info.items()))
        print('SeverRewardManager: test num info', info)

        print('SeverRewardManager: judge start')
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [executor.submit(run, answer_text, ground_truth, self.batch_size) for answer_text, ground_truth in zip(answer_texts, ground_truths)]
            for i, future in enumerate(futures):
                score = future.result()
                total_score = score + format_scores[i]
                reward_tensor[i, valid_response_lens[i] - 1] = total_score
                debug_info[i].append(f"  Answer Score: {score}")
                debug_info[i].append(f"  Total Score: {total_score}")
        end_time = time.time()
        print('SeverRewardManager: judge time:', end_time - start_time, 's')
        print_info = random.sample(debug_info, min(self.print_num, len(debug_info)))
        for response_info in print_info:
            print('\n'.join(response_info))
        return reward_tensor
