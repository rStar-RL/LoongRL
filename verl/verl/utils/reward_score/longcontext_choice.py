import re
import os


def compute_score(solution_str, ground_truth):
    """
    Compute score for longcontext choice in the form of choices
    Args:
        solution_str (str): Solution string
        ground_truth (str): Ground truth string
    Returns:
        float: Score
    export LONGCONTEXTCHOICE_PUNISH_MULTIPLE_BRACES=1
    export LONGCONTEXTCHOICE_ANSWER_OVER_FLOW_LIMIT=32
    export LONGCONTEXTCHOICE_MAX_BOXED_LIMIT=4
    export LONGCONTEXTCHOICE_EOT_OVER_FLOW_LIMIT=16
    """
    solution_str = solution_str.strip()
    retval = 0
    punish_multiple_braces = int(
        os.getenv("LONGCONTEXTCHOICE_PUNISH_MULTIPLE_BRACES", 1)
    )
    answer_overflow_limit = int(
        os.getenv("LONGCONTEXTCHOICE_ANSWER_OVER_FLOW_LIMIT", 256)
    )
    max_boxed_limit = int(os.getenv("LONGCONTEXTCHOICE_MAX_BOXED_LIMIT", 4))
    eot_over_flow_limit = int(
        os.getenv("LONGCONTEXTCHOICE_EOT_OVER_FLOW_LIMIT", 16)
    )
    try:
        if max_boxed_limit > 0:
            # check if there are more than max_boxed_limit boxed in solution_str
            # punish if there are more than max_boxed_limit
            if solution_str.count("\\boxed") > max_boxed_limit:
                retval = 0
                return retval
        boxed_part = last_boxed_only_string(solution_str)
        pred = remove_boxed(boxed_part)
        if eot_over_flow_limit > 0:
            # check if the distance between the last boxed and the end of solution_str is greater than eot_over_flow_limit
            if (
                len(solution_str) - (solution_str.rfind(pred) + len(pred))
                > eot_over_flow_limit
            ):
                retval = 0
        if punish_multiple_braces:
            # check if there are multiple braces in pred
            # punish if there are multiple braces
            if (
                pred.count("{") > 1
                or pred.count("}") > 1
                or pred.count("\\") > 1
            ):
                retval = 0
                return retval
        if answer_overflow_limit > 0:
            # check if the length of pred is greater than answer_overflow_limit
            if len(pred) > answer_overflow_limit:
                retval = 0
                return retval
        # extract the first option from the solution string
        # option in the form of (A)
        pred_option = re.search(r"\((\w)\)", pred)
        if pred_option:
            # check if there are multiple matches in pred_option
            # punish if there are multiple matches
            if len(pred_option.groups()) > 1:
                retval = 0
            else:
                pred_option = pred_option.group(1)
                if pred_option.lower() == ground_truth.lower():
                    retval = 1.0
                else:
                    retval = 0
        else:
            retval = 0

    except Exception as e:
        print(f"Error encountered: {e}")
        return 0

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
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    max_retval = 0
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            retval = 0
            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                assert isinstance(
                    pred, str
                ), f"pred should be a string, got {type(pred)} instead"
                assert isinstance(
                    truth, str
                ), f"truth should be a string, got {type(truth)} instead"
                if is_gt_in_pred(pred, truth):
                    retval = 1.0
                    answer_over_flow_limit = int(
                        os.getenv("ANSWER_OVER_FLOW_LIMIT", 128)
                    )
                    if len(pred) - len(truth) > answer_over_flow_limit:
                        retval -= 1.0
                    eot_over_flow_limit = int(
                        os.getenv("EOT_OVER_FLOW_LIMIT", 32)
                    )
                    if (
                        len(solution_str)
                        - (solution_str.rfind(pred) + len(pred))
                        > eot_over_flow_limit
                    ):
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
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def normalize_text(text):
    """
    Normalize text by lowercasing and removing special characters
    """
    text = re.sub(
        "[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]",
        " ",
        text.lower(),
    )
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
