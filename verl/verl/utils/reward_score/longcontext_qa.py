import re
import os



def compute_score(solution_str, ground_truth):
    """
    Compute score for longcontext_qa
    Args:
        solution_str (str): Solution string
        ground_truth (str): Ground truth string
    Returns:
        float: Score
    """
    solution_str = solution_str.strip()
    reward_calc_type = os.getenv("REWARD_CALC_TYPE", "pure_exact_match")
    if reward_calc_type == "pure_exact_match":
        retval = _pure_exact_match_in_string(solution_str, ground_truth)
    elif reward_calc_type == "f1_score":
        raise NotImplementedError("F1 score is not implemented yet")
    elif reward_calc_type == "format_exact_match":
        retval = _format_exact_match_in_string(solution_str, ground_truth)
    else:
        raise ValueError(f"Unknown reward_calc_type: {reward_calc_type}")

    return retval
    
        
def _pure_exact_match_in_string(solution_str, ground_truth):
    if isinstance(ground_truth, str):  # More Pythonic way to check type
        ground_truth = [ground_truth]
    
    retval = 0  # Default return value
    
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if is_gt_in_pred(pred, truth):
                    retval = 1.0
                    return retval  # Early return if a match is found
        except Exception as e:
            print(f"Error encountered: {e}")
            return retval  # Return 0 if an exception occurs

    return retval  # Return 0 if no match is found

def _format_exact_match_in_string(solution_str, ground_truth):
    """
    export REWARD_CALC_TYPE=format_exact_match
    export ANSWER_OVER_FLOW_LIMIT=128
    export EOT_OVER_FLOW_LIMIT=32
    export MAX_BOXED_LIMIT=1
    export PUNISH_MULTIPLE_BRACES=1
    """
    max_boxed_limit = int(os.getenv("MAX_BOXED_LIMIT", 1))
    punish_multiple_braces = int(os.getenv("PUNISH_MULTIPLE_BRACES", 1))
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    max_retval = 0
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            retval = 0
            if max_boxed_limit > 0:
                if solution_str.count("\\boxed") > max_boxed_limit:
                    raise ValueError(f"Too many boxed parts in solution_str: {solution_str.count('\\boxed')} > {max_boxed_limit}")
                    # return 0

            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if punish_multiple_braces > 0:
                    if pred.count("{") > 1 or pred.count("}") > 1 or pred.count("\\") > 1:
                        print(f"Multiple braces found in pred: {pred}")
                        raise ValueError(f"Multiple braces found in pred: {pred}")
                        # return 0
                assert isinstance(pred, str), f"pred should be a string, got {type(pred)} instead"
                assert isinstance(truth, str), f"truth should be a string, got {type(truth)} instead"
                if is_gt_in_pred(pred, truth):
                    retval = 1.0
                    answer_over_flow_limit = int(os.getenv("ANSWER_OVER_FLOW_LIMIT", 128))
                    if len(pred) - len(truth) > answer_over_flow_limit:
                        retval -= 1.0
                    eot_over_flow_limit = int(os.getenv("EOT_OVER_FLOW_LIMIT", 32))
                    if len(solution_str) - (solution_str.rfind(pred)+len(pred)) > eot_over_flow_limit:
                        retval -= 1.0
                    max_retval = max(max_retval, retval)
                    
        except Exception as e:
            print(f"Error encountered: {e}")
            return max_retval
                
    return max_retval





def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def normalize_text(text):
    """
    Normalize text by lowercasing and removing special characters
    """
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

def is_gt_in_pred(pred, ground_truth, verbose=False):
    if pred is None and ground_truth is None:
        print("WARNING: Both None")
        return True
    if pred is None or ground_truth is None:
        return False

    try:
        # normalize string
        ss1 = normalize_text(pred)
        ss2 = normalize_text(ground_truth)
        if verbose:
            print(ss1, ss2)
        return ss2 in ss1
    except Exception:
        return ground_truth in pred

    