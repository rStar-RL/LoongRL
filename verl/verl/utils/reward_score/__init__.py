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
# from . import gsm8k, math, prime_math, prime_code
import os


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        # from . import math
        # res = math.compute_score(solution_str, ground_truth)

        # Use Math-Verify (https://github.com/huggingface/Math-Verify) for better evaluation accuracy
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ] or data_source.startswith('custom_math_'):
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['deepmind/code_contests']:
        from . import code_server
        res = code_server.compute_score(solution_str, ground_truth)
        # directly return for now, since it includes debug str
        return res
    elif data_source.startswith('custom_longcontextqa_'):
        if os.getenv("LLM_JUDGE", False) is False:
            from . import longcontext_qa
            res = longcontext_qa.compute_score(solution_str, ground_truth)
        else:
            from . import longcontext_qa_llm_judge
            res = longcontext_qa_llm_judge.compute_score(solution_str, ground_truth, prompt_str=extra_info)['score']
    elif data_source.startswith('custom_longcontextchoice_'):
        from . import longcontext_choice
        res = longcontext_choice.compute_score(solution_str, ground_truth)
    elif data_source.startswith('custom_longcontext_needle_qa_'):
        from . import longcontext_qa
        res = longcontext_qa.compute_score(solution_str, ground_truth)
    elif data_source.startswith('custom_mathqa_choice_'):
        from . import mathqa_choice
        res = mathqa_choice.compute_score(solution_str, ground_truth)
    elif data_source.startswith('custom_sentence_needle_'):
        from . import sentence_needle
        res = sentence_needle.compute_score(solution_str, ground_truth)
    elif data_source.startswith('custom_rulerniah_'):
        from . import ruler_multi
        res = ruler_multi.compute_score(solution_str, ground_truth)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['math_dapo']:
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
        res = res['acc']
    else:
        print(f"data_source {data_source} not found")
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
