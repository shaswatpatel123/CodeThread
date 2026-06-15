from typing import Any

from .parser_constants import parse_log_pytest_combined
from .python_constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)

# MARK: Utility functions
def _get_variants(case: str, sm: dict[str, str]) -> list[str]:
    """Return status values for all parameterized variants (e.g. test[True], test[False])."""
    prefix = case + "["
    return [v for k, v in sm.items() if k.startswith(prefix)]


def test_passed(case: str, sm: dict[str, str]) -> bool:
    if case in sm:
        return sm[case] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]
    # Oracle may store test_name without params; check all test_name[...] variants
    variants = _get_variants(case, sm)
    return bool(variants) and all(v in [TestStatus.PASSED.value, TestStatus.XFAIL.value] for v in variants)


def test_failed(case: str, sm: dict[str, str]) -> bool:
    if case in sm:
        return sm[case] in [TestStatus.FAILED.value, TestStatus.ERROR.value]
    # Oracle may store test_name without params; check all test_name[...] variants
    variants = _get_variants(case, sm)
    if variants:
        return any(v in [TestStatus.FAILED.value, TestStatus.ERROR.value] for v in variants)
    return True  # absent from map entirely → treat as failed


# MARK: Evaluation report functions
def get_logs_eval(test_spec: dict, log_fp: str) -> tuple[dict[str, str], bool]:
    """
    Retrieve evaluation results for a task instance from its corresponding log file

    Args:
        log_fp (str): path to log file
    Returns:
        bool: whether the patch applied successfully
        dict: status map

    TODO(john-b-yang): Check this is working properly...
    """
    log_parser = parse_log_pytest_combined

    with open(log_fp) as f:
        content = f.read()
        # TODO fix constant here
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            return {}, False
        elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
            # Test patch did not apply (should not happen at all)
            return {}, False

        # Get status map of evaluation results
        content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        return log_parser(content, test_spec), True


def get_eval_tests_report(
    eval_status_map: dict[str, str],
    gold_results: dict[str, str],
    calculate_to_fail: bool = False,
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
) -> dict[str, dict[str, list[str]]]:
    """
    Create a report based on failure/pass change from gold results to eval results.

    Args:
        eval_sm (dict): evaluation status map
        gold_results (dict): gold results
        calculate_to_fail (bool): whether to calculate metrics for "x to fail" tests
    Returns:
        report (dict): report of metrics

    Metric Definitions (Gold Result Pair + Eval Result):
    - Fail-Pass (F2P) + P: Success (Resolution)
    - Pass-Pass (P2P) + P: Success (Maintenance)
    - Fail-Pass (F2P) + F: Failure
    - Pass-Pass (P2P) + F: Failure

    Miscellaneous Definitions
    - Fail-Fail (F2F) + F: Failure Maintenance
    - Pass-Fail (P2F) + F: Not considered
    - Fail-Fail (F2F) + P: Success (Extra Credit)
    - Pass-Fail (P2F) + P: Not considered
    """

    def check_pass_and_fail(test_case, eval_status_map, success, failed):
        if test_passed(test_case, eval_status_map):
            success.append(test_case)
        elif test_failed(test_case, eval_status_map):
            failed.append(test_case)

    def check_pass_and_fail_p2p(test_case, eval_status_map, success, failed):
        # Skip P2P tests absent from the map — these failed to be collected
        # (e.g. import errors, missing fixtures) rather than actually regressing.
        # Don't penalise the agent for infrastructure / collection failures.
        in_map = test_case in eval_status_map
        has_variants = bool(_get_variants(test_case, eval_status_map))
        if not in_map and not has_variants:
            return
        if test_passed(test_case, eval_status_map):
            success.append(test_case)
        elif test_failed(test_case, eval_status_map):
            failed.append(test_case)

    def check_fail_only(test_case, eval_status_map, success, failed):
        if (
            test_case in eval_status_map
            and eval_status_map[test_case] == TestStatus.FAILED.value
        ):
            failed.append(test_case)
        else:
            success.append(test_case)

    check_f2p = (
        check_pass_and_fail if eval_type == EvalType.PASS_AND_FAIL else check_fail_only
    )
    check_p2p = (
        check_pass_and_fail_p2p if eval_type == EvalType.PASS_AND_FAIL else check_fail_only
    )

    # Calculate resolution metrics
    f2p_success = []
    f2p_failure = []
    for test_case in gold_results[FAIL_TO_PASS]:
        check_f2p(test_case, eval_status_map, f2p_success, f2p_failure)

    # Calculate maintenance metrics (only for tests that were actually run)
    p2p_success = []
    p2p_failure = []
    for test_case in gold_results[PASS_TO_PASS]:
        check_p2p(test_case, eval_status_map, p2p_success, p2p_failure)

    results = {
        FAIL_TO_PASS: {
            "success": f2p_success,
            "failure": f2p_failure,
        },
        PASS_TO_PASS: {
            "success": p2p_success,
            "failure": p2p_failure,
        },
    }

    f2f_success = []
    f2f_failure = []
    p2f_success = []
    p2f_failure = []
    if calculate_to_fail:
        # Calculate "extra credit" metrics
        for test_case in gold_results[FAIL_TO_FAIL]:
            check_f2p(test_case, eval_status_map, f2f_success, f2f_failure)

        # Calculate not considered metrics
        for test_case in gold_results[PASS_TO_FAIL]:
            check_f2p(test_case, eval_status_map, p2f_success, p2f_failure)

    results.update(
        {
            FAIL_TO_FAIL: {
                "success": f2f_success,
                "failure": f2f_failure,
            },
            PASS_TO_FAIL: {
                "success": p2f_success,
                "failure": p2f_failure,
            },
        }
    )
    return results


def compute_fail_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute fail-to-pass metric. Accepts single report as argument.
    """
    total = len(report[FAIL_TO_PASS]["success"]) + len(report[FAIL_TO_PASS]["failure"])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]["success"]) / total


def compute_pass_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute pass-to-pass metric. Accepts single report as argument.
    """
    total = len(report[PASS_TO_PASS]["success"]) + len(report[PASS_TO_PASS]["failure"])
    if total == 0:
        # TODO: Don't factor in p2p metrics
        return 1
    return len(report[PASS_TO_PASS]["success"]) / total


def get_resolution_status(report: dict[str, dict[str, Any]]) -> str:
    """
    Determine resolved status of an evaluation instance

    Criteria:
        - If fail-to-pass (Resolution) = 1 and pass-to-pass (Maintenance) = 1 -> FULL
        - If (fail-to-pass (Resolution) < 1 and > 0) and pass-to-pass (Maintenance) = 1 -> PARTIAL
        - Otherwise -> NO
    """
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)

    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    elif f2p < 1 and f2p > 0 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    else:
        return ResolvedStatus.NO.value


def get_eval_report(
    test_spec: dict,
    patch: str,
    test_log_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    """
    Generate a report of model evaluation results from a prediction, task instance,
    and evaluation log.

    Args:
        test_spec (dict): test spec containing keys "instance_id", "FAIL_TO_PASS", and "PASS_TO_PASS"
        prediction (dict): prediction containing keys "instance_id", "model_name_or_path", and "model_patch"
        log_path (str): path to evaluation log
        include_tests_status (bool): whether to include the status of each test in the returned report
    Returns:
        report (dict): report of metrics
    """
    report_map = {}

    instance_id = test_spec[KEY_INSTANCE_ID]
    report_map[instance_id] = {
        "patch_is_None": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "resolved": False,
    }

    # Check if the model patch exists
    if patch is None or len( patch.strip() ) == 0:
        report_map[instance_id]["patch_is_None"] = True
        return report_map
    report_map[instance_id]["patch_exists"] = True

    # Get evaluation logs
    eval_status_map, found = get_logs_eval(test_spec, test_log_path)

    if not found:
        return report_map
    report_map[instance_id]["patch_successfully_applied"] = True

    f2p_ref = test_spec[FAIL_TO_PASS]

    if f2p_ref is None or f2p_ref == "":
        raise Exception("Expected FAIL_TO_PASS to exist")

    p2p_ref = test_spec[PASS_TO_PASS]
    if p2p_ref is None or p2p_ref == "":
        p2p_ref = ""


    # F2P/P2P may be a comma-separated string (raw dataset) or already a list
    # (pre-processed by run/extra/featbench.py). Handle both.
    def _to_list(value) -> list[str]:
        if isinstance(value, list):
            return [x.strip() for x in value if str(x).strip()]
        return [x.strip() for x in str(value).split(",") if x.strip()]

    f2p_ref = _to_list(f2p_ref)
    p2p_ref = _to_list(p2p_ref)

    eval_ref = {
        KEY_INSTANCE_ID: instance_id,
        FAIL_TO_PASS: f2p_ref,
        PASS_TO_PASS: p2p_ref,
    }

    eval_type = EvalType.FAIL_ONLY if test_spec['repo'] in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL

    report = get_eval_tests_report(
        eval_status_map, eval_ref, eval_type=eval_type
    )
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]["resolved"] = True

    if include_tests_status:
        report_map[instance_id]["tests_status"] = report  # type: ignore

    return report_map

