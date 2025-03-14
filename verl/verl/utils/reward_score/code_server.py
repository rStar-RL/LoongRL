import re
from typing import Dict, Tuple, Optional
import requests


def judge_batch(submissions):
    url = 'http://localhost:8000/judge/batch'
    results = [None] * len(submissions)

    try:
        while True:
            candidates = []
            idxs = []
            for idx, (result, submission) in enumerate(zip(results, submissions)):
                if result is None:
                    candidates.append(submission)
                    idxs.append(idx)

            if not candidates:
                break

            data = {
                "type": "batch",
                "submissions": candidates
            }
            response = requests.post(url, json=data, timeout=60)
            response_json = response.json()
            for idx, result in zip(idxs, response_json['results']):
                if result['success'] or result['reason'] != 'queue_timeout':
                    results[idx] = result
    except Exception as e:
        print(f"Error: {e}")
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {'success': False, 'error': str(e)}
    return results


def send_request_batch_solutions(languages, solutions, cases):
    submissions = []
    for language, solution in zip(languages, solutions):
        for in_case, out_case in cases:
            submissions.append({
                "type": language,
                "solution": solution,
                "input": in_case,
                "expected_output": out_case
            })

    batch_results = judge_batch(submissions)
    results = [[] for _ in range(len(solutions))]
    for i, result in enumerate(batch_results):
        results[i // len(cases)].append(result)

    return results


def check_language(code_string):
    # TODO: support more languages
    if code_string.find("#include") != -1:
        return "cpp"
    return "python"


def batch_judge(solutions, test_cases, batch_size):
    """
    Judge a batch of solutions against the test cases. Note solutions and test cases are
    corresponding to a same problem.
    Since the test number can be very large, we will do batch to speed up the process. 
    """
    input_case = test_cases["input"]
    output_case = test_cases["output"]
    cases = []
    for i in range(len(input_case)):
        cases.append((str(input_case[i]), str(output_case[i])))

    if not cases:
        return [0] * len(solutions)

    scores = [None] * len(solutions)
    for i, solution in enumerate(solutions):
        if solution is None:
            scores[i] = 0

    languages = []
    for solution in solutions:
        if solution is None:
            languages.append(None)
        else:
            languages.append(check_language(solution))

    test_bsz = max(1, batch_size // len(solutions))
    for i in range(0, len(cases), test_bsz):
        test_idxs = range(i, min(i + test_bsz, len(cases)))
        cur_cases = [cases[idx] for idx in test_idxs]

        cur_languages, cur_solutions, cur_idxs = [], [], []
        for idx, solution in enumerate(solutions):
            if scores[idx] is None:
                cur_languages.append(languages[idx])
                cur_solutions.append(solution)
                cur_idxs.append(idx)
        if not cur_solutions:
            break

        results = send_request_batch_solutions(cur_languages, cur_solutions, cur_cases)
        # currently we stop when one test case fails
        for test_results, idx in zip(results, cur_idxs):
            detect_fail = False
            pass_cnt = 0
            for test_idx, result in zip(test_idxs, test_results):
                if not result['success']:
                    detect_fail = True
                else:
                    pass_cnt += 1
            if detect_fail:
                scores[idx] = pass_cnt + i

    for i in range(len(scores)):
        if scores[i] is None:
            scores[i] = len(cases)

    scores = [score / len(cases) for score in scores]
    return scores


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
        question_str = solution_str.split("Assistant:", 1)[0]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
        question_str = solution_str.split("<|im_start|>assistant", 1)[0]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str, ""

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        return None, processed_str, question_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str, question_str


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    debug_str = []
    debug_str.append("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        debug_str.append(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            debug_str.append(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    legal_end_pattern1 = "</answer><|im_end|>"
    legal_end_pattern2 = "</answer><|endoftext|>"
    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        debug_str.append("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    elif not (processed_str.strip()[-len(legal_end_pattern1):] == legal_end_pattern1 or processed_str.strip()[-len(legal_end_pattern2):] == legal_end_pattern2):
        debug_str.append("  [Error] Incorrect end token")
        validation_passed = False
    elif processed_str.strip()[0:len("<think>")] != "<think>":
        debug_str.append("  [Error] Incorrect start token: Expected <think>")
        validation_passed = False
    else:
        debug_str.append("  Tag sequence validation passed")

    return validation_passed, debug_str


def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_reward: int = 1,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    debug_str = []
    debug_str.append("\n" + "="*80)
    debug_str.append(" Processing New Sample ".center(80, '='))
    
    # Extract model answer
    answer_text, processed_str, question_str = extract_solution(solution_str)
    debug_str.append(f"\n[Question]\n{question_str}")
    debug_str.append(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_correct, debug_info = validate_response_structure(processed_str)
    debug_str.extend(debug_info)
    format_score = format_reward if format_correct else -abs(format_reward)
    debug_str.append(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    debug_str.append(f"  Format score: {format_score}")

    # Validate answer content
    answer_score = run(answer_text, ground_truth, 1)

    total_score = format_score + answer_score
    debug_str.append("\n" + "-"*80)
    debug_str.append(f" Final Score ".center(80, '-'))
    debug_str.append(f"  Format: {format_score}")
    debug_str.append(f"  Answer: {answer_score}")
    debug_str.append(f"  Total: {total_score}")
    debug_str.append("="*80 + "\n")

    return total_score, "\n".join(debug_str)

if __name__ == "__main__":
    # should print [1.0] and [1.0]
    print(batch_judge(['print(sum(map(int, input().split())))'], {'input': ['4 5'], 'output': ['9']}, 5))
    print(batch_judge(["#include <bits/stdc++.h>\nusing namespace std;\nint main() {\n  string s;\n  cin >> s;\n  for(int j = 0; j < 10000000; ++j) for (int i = 0; i < s.length(); i++) {\n    if (s[i] == s[i + 1] && s[i] == s[i + 2]) {\n      cout << s[i];\n      return 0;\n    }\n  }\n  cout << -1;\n  return 0;\n}\n"], {"input": [123123123129912857127437128819329319200] * 10, "output":[-1] * 10}, 5))
