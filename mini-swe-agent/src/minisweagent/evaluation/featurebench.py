import os
from pathlib import Path
from minisweagent import Environment
from minisweagent.evaluation.python_constants import MAP_REPO_VERSION_TO_SPECS
from unidiff import PatchSet
from logging import Logger
import json
from typing import Optional, Dict, Any, Union, List

from minisweagent.run.extra.utils.exceptions import NoFailToPassError, NoTestPatchError, PatchApplyError, RepoSettingsError, TestRunError
from minisweagent.evaluation.parser_constants import MAP_REPO_TO_PARSER_PY, parse_log_pytest
from minisweagent.evaluation.python_constants import EvalType, TestStatus

CommandType = Union[str, List[str]]

def _normalize_patch_content(patch_content: str) -> str:
    # Some LLM outputs omit the final newline; `git apply` may report
    # `error: corrupt patch at line ...` when the patch file ends abruptly.
    if patch_content and not patch_content.endswith("\n"):
        return patch_content + "\n"
    return patch_content


def build_test_command(test_cmd: str, timeout_one: Optional[int] = None) -> str:
    """
    Build test command with timeout configuration.

    Args:
        test_cmd: Base test command
        timeout_one: Timeout per test case (seconds)

    Returns:
        Complete test command string
    """
    if timeout_one is not None and timeout_one > 0:
        return f"{test_cmd} --timeout={timeout_one}"
    return test_cmd

def _should_use_uv(specs: Optional[Dict[str, Any]]) -> bool:
    return bool(specs and specs.get("use_uv", False))

def apply_uv_run_prefix(command: CommandType, specs: Optional[Dict[str, Any]] = None) -> CommandType:
    """Optionally prefix a command with "uv run" when use_uv is enabled.

    Args:
        command: Command string or list of arguments.
        specs: Repo specs dict; reads `use_uv` boolean.

    Returns:
        The command with "uv run" prefixed if enabled and not already present.
    """
    if not command or not _should_use_uv(specs):
        return command

    if isinstance(command, list):
        if len(command) >= 2 and command[0] == "uv" and command[1] == "run":
            return command
        return ["uv", "run", *command]

    if isinstance(command, str):
        stripped = command.lstrip()
        if stripped == "uv run" or stripped.startswith("uv run "):
            return command
        return f"uv run {command}"

    return command

def get_test_config_from_repo_settings(repo_settings: dict) -> dict:
    """
    Extract test configuration from repo_settings.

    Args:
        repo_settings: Repository settings dictionary

    Returns:
        Dictionary with test_cmd, timeout_run, timeout_one
    """
    return {
        "test_cmd": repo_settings.get("test_cmd", "pytest -rA -p no:cacheprovider --color=no"),
        "timeout_run": repo_settings.get("timeout_run", 600),
        "timeout_one": repo_settings.get("timeout_one", 10),
        "use_uv": repo_settings.get("use_uv", False),
    }

def parse_repo_settings(instance: dict) -> dict:
    """
    Parse repo_settings from instance data.

    Args:
        instance: Instance data as dict

    Returns:
        Dictionary containing repo settings
    """
    repo_settings_str = instance.get("repo_settings", None)
    if not repo_settings_str or repo_settings_str is None:
        raise RepoSettingsError(f"No repo settings found for instance {instance['instance_id']}")

    return json.loads(repo_settings_str)

def parse_test_outputs(instance: dict, instance_dir: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    """
    Parse test output files and return parsed results (test_name -> status mapping).

    Args:
        log_dir: Directory containing test output files
        repo_name: Repository name for parser selection
        level: Evaluation level (1 or 2)

    Returns:
        Tuple of (f2p_status_map, p2p_status_map_list)
    """
    parser_fn = MAP_REPO_TO_PARSER_PY.get(instance["repo"], parse_log_pytest)

    f2p_test_output_path = instance_dir / "f2p_test_output.txt"
    f2p_status_map: dict[str, str] = {}
    if f2p_test_output_path.exists():
        with open(f2p_test_output_path, 'r', encoding='utf-8') as f:
            f2p_test_output = f.read()
        f2p_status_map = parser_fn(f2p_test_output)

    # Parse P2P test outputs
    p2p_status_map_list: list[dict[str, str]] = []
    for p2p_file in instance_dir.glob("test_output_p2p_*.txt"):
        with open(p2p_file, "r", encoding='utf-8', errors='replace') as f:
            p2p_output = f.read()
        p2p_status_map = parser_fn(p2p_output)
        p2p_status_map_list.append(p2p_status_map)

    return f2p_status_map, p2p_status_map_list



def test_passed(case: str, status_map: dict[str, str]) -> bool:
    """
    Check if a test case passed (PASS_AND_FAIL mode).

    A test is considered passed if:
    - It exists in the status map AND
    - Its status is PASSED or XFAIL

    Args:
        case: Test case name
        status_map: Dict mapping test names to status values

    Returns:
        bool: True if test passed
    """
    return case in status_map and status_map[case] in [
        TestStatus.PASSED.value,
        TestStatus.XFAIL.value,
    ]

def test_failed(case: str, status_map: dict[str, str]) -> bool:
    """
    Check if a test case failed (PASS_AND_FAIL mode).

    A test is considered failed if:
    - It's not in the status map OR
    - Its status is FAILED or ERROR

    Args:
        case: Test case name
        status_map: Dict mapping test names to status values

    Returns:
        bool: True if test failed
    """
    return case not in status_map or status_map[case] in [
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
    ]

def get_eval_report(
    eval_status_map: dict[str, str],
    expected_tests: list[str],
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
) -> dict[str, Any]:
    """
    Compute evaluation report based on test results.

    Args:
        eval_status_map: Dict mapping test names to status values (from parser)
        expected_tests: List of expected test names to check
        eval_type: Evaluation mode (PASS_AND_FAIL or FAIL_ONLY)

    Returns:
        dict: Report containing:
            - total: Total number of tests
            - success: Number of successful tests
            - failure: Number of failed tests
            - pass_rate: Pass rate (success / total)
            - success_tests: List of successful test names
            - failure_tests: List of failed test names
    """

    def check_pass_and_fail(
        test_case: str,
        status_map: dict[str, str],
        success: list[str],
        failed: list[str],
    ) -> None:
        """
        Check test case in PASS_AND_FAIL mode.
        - Test passes if status is PASSED or XFAIL.
        - Test fails if status is FAILED, ERROR, or not in map.
        """
        if test_passed(test_case, status_map):
            success.append(test_case)
        elif test_failed(test_case, status_map):
            failed.append(test_case)

    def check_fail_only(
        test_case: str,
        status_map: dict[str, str],
        success: list[str],
        failed: list[str],
    ) -> None:
        """
        Check test case in FAIL_ONLY mode.
        - Test fails only if explicitly marked as FAILED or ERROR.
        - Everything else (PASSED, XFAIL, SKIPPED, or not in map) silently passes.
        """
        if test_case in status_map and status_map[test_case] in [
            TestStatus.FAILED.value,
            TestStatus.ERROR.value,
        ]:
            failed.append(test_case)
        else:
            success.append(test_case)

    # Select check function based on eval type
    check_test_case = (
        check_pass_and_fail if eval_type == EvalType.PASS_AND_FAIL else check_fail_only
    )

    success_tests: list[str] = []
    failure_tests: list[str] = []

    for test_case in expected_tests:
        check_test_case(test_case, eval_status_map, success_tests, failure_tests)

    total = len(expected_tests)
    success = len(success_tests)
    failure = len(failure_tests)
    pass_rate = round(success / total, 4) if total > 0 else 0.0

    return {
        "total": total,
        "success": success,
        "failure": failure,
        "pass_rate": pass_rate,
        "success_tests": success_tests,
        "failure_tests": failure_tests,
    }

def build_test_status(
    f2p_status_map: dict[str, str],
    p2p_status_map_list: list[dict[str, str]],
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
) -> tuple[list, list, list, list]:
    """
    Build test status from parsed results using grading module.

    Args:
        f2p_status_map: F2P test status map (test_name -> status)
        p2p_status_map_list: List of P2P test status maps
        eval_type: Evaluation mode (PASS_AND_FAIL or FAIL_ONLY)

    Returns:
        Tuple of (f2p_success, f2p_failure, p2p_success, p2p_failure)
    """

    # Grade F2P tests
    f2p_tests = list(f2p_status_map.keys())
    f2p_report = get_eval_report(f2p_status_map, f2p_tests, eval_type)
    f2p_success = f2p_report["success_tests"]
    f2p_failure = f2p_report["failure_tests"]

    # Grade P2P tests
    p2p_success = []
    p2p_failure = []
    for p2p_status_map in p2p_status_map_list:
        p2p_tests = list(p2p_status_map.keys())
        p2p_report = get_eval_report(p2p_status_map, p2p_tests, eval_type)
        p2p_success.extend(p2p_report["success_tests"])
        p2p_failure.extend(p2p_report["failure_tests"])

    return f2p_success, f2p_failure, p2p_success, p2p_failure


def generate_instance_report(
    instance_id: str,
    patch_content: str | None,
    patch_applied: bool,
    f2p_success_list: list,
    f2p_failure_list: list,
    p2p_success_list: list,
    p2p_failure_list: list,
    eval_results: dict,
) -> dict:
    """
    Generate report for a single instance.

    Args:
        instance_id: Instance ID
        n_attempt: Attempt number
        patch_content: Patch content
        patch_applied: Whether patch was applied
        f2p_success_list: List of successful F2P tests
        f2p_failure_list: List of failed F2P tests
        p2p_success_list: List of successful P2P tests
        p2p_failure_list: List of failed P2P tests
        eval_results: Raw evaluation results

    Returns:
        Report dictionary
    """
    patch_is_none = patch_content is None
    patch_exists = bool(patch_content and patch_content.strip())

    # Determine if resolved
    resolved = eval_results.get("f2p_success", False) and eval_results.get("p2p_success", True)

    # Calculate F2P pass rate
    f2p_total = len(f2p_success_list) + len(f2p_failure_list)
    f2p_pass_rate = round(len(f2p_success_list) / f2p_total, 4) if f2p_total > 0 else 0.0

    report = {
        instance_id: {
            "patch_is_None": patch_is_none,
            "patch_exists": patch_exists,
            "patch_successfully_applied": patch_applied,
            "resolved": resolved,
            "pass_rate": f2p_pass_rate,
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": f2p_success_list,
                    "failure": f2p_failure_list
                },
                "PASS_TO_PASS": {
                    "success": p2p_success_list,
                    "failure": p2p_failure_list
                }
            }
        }
    }

    return report

def evaluation( instance: dict, instance_dir: Path, patch: str, env: Environment, test_config: dict, logger: Logger ) -> str:
    instance_id = instance["instance_id"]

    test_output_path = instance_dir / "test_output.txt"

    results = {
        "instance_id": instance_id,
        "patch_applied": False,
        "f2p_success": False,
        "p2p_success": False,
        "error": None,
    }


    # Delete F2P test files just in case Agent generated similar test files
    f2p_tests = instance["FAIL_TO_PASS"]
    if f2p_tests:
        f2p_tests = f2p_tests if isinstance(f2p_tests, list) else [f2p_tests]
        for f2p_test in f2p_tests:
            env.execute(f"rm -rf {f2p_test}")
    else:
        raise NoFailToPassError(f"No F2P test files to delete for instance {instance_id}")

    # Restore F2P test
    test_patch_content = instance.get('test_patch', None)

    if test_patch_content is None:
        raise NoTestPatchError(f"No test patch content found for instance {instance_id}")
    
    test_patch_content = _normalize_patch_content(test_patch_content)

    test_patch_file = instance_dir / "test_patch_reverse.diff"
    test_patch_file.write_text(test_patch_content)

    try:
        # Move to container
        env.copy_to_container(test_patch_file, "/testbed/test_patch_reverse.diff")

        # Reverse apply test_patch
        env.execute(f"git apply --reverse --whitespace=fix /testbed/test_patch_reverse.diff")

        # Remove from container
        env.execute(f"rm -rf /testbed/test_patch_reverse.diff")

        test_patch_file.unlink()
    except Exception as e:
        raise PatchApplyError() 

    # Run tests
    repo_settings = parse_repo_settings(instance)
    test_config = get_test_config_from_repo_settings(repo_settings)


    base_test_cmd = MAP_REPO_VERSION_TO_SPECS.get(instance["repo"], None)
    if base_test_cmd is None:
            base_test_cmd = test_config["test_cmd"]
    timeout_one = test_config["timeout_one"]
    
    test_cmd = build_test_command(base_test_cmd, timeout_one)
    test_cmd = apply_uv_run_prefix(test_cmd, test_config)

    effective_timeout = test_config.get("timeout_run", 1800)

    pass_to_pass = instance.get('PASS_TO_PASS', [])
    p2p_tests = pass_to_pass if isinstance(pass_to_pass, list) else []

    # Run F2P test
    f2p_test = f2p_tests[0]
    result = env.execute(f"cd /testbed && {test_cmd} {f2p_test}", timeout=effective_timeout)
    if result['returncode'] != 0:
        raise TestRunError(f"Failed to run F2P test for instance {instance_id}. Error: {result['output']}")

    test_output = result['output'].decode('utf-8', errors='replace')
    results["f2p_success"] = (result['returncode'] == 0)

    # Save F2P test output
    with open(test_output_path, 'w', encoding='utf-8') as f:
        f.write(test_output)

    # Run P2P tests
    p2p_results = []
    if p2p_tests is None or len(p2p_tests) == 0:
        results["p2p_success"] = True
    else:
        for p2p_test in p2p_tests:
            p2p_test_path = p2p_test
            if not p2p_test_path.startswith('/testbed/'):
                p2p_test_path = f"/testbed/{p2p_test_path}"

            test_cmd_full = f"{test_cmd} {p2p_test_path}"

            result = env.execute(test_cmd_full, timeout=effective_timeout)
            if result['returncode'] != 0:
                raise TestRunError(f"Failed to run P2P test for instance {instance_id}. Error: {result['output']}")

            p2p_output = result['output'].decode('utf-8', errors='replace')
            p2p_results.append(result['returncode'] == 0)

            # Save P2P test output
            test_file_name = os.path.basename(p2p_test_path).replace('.py', '')
            p2p_output_path = instance_dir / f"p2p_{test_file_name}.txt"
            with open(p2p_output_path, 'w', encoding='utf-8') as f:
                f.write(p2p_output)

            results["p2p_success"] = all(p2p_results)

    
    f2p_status_map, p2p_status_map_list = parse_test_outputs(instance, instance_dir)
    # Determine eval_type based on repo
    eval_type =  EvalType.PASS_AND_FAIL

    f2p_success, f2p_failure, p2p_success, p2p_failure = build_test_status(
        f2p_status_map, p2p_status_map_list, eval_type
    )

    report = generate_instance_report(
        instance_id=instance_id,
        patch_content=_normalize_patch_content(patch),
        patch_applied=results.get("patch_applied", False),
        f2p_success_list=f2p_success,
        f2p_failure_list=f2p_failure,
        p2p_success_list=p2p_success,
        p2p_failure_list=p2p_failure,
        eval_results=results,
    )

    report_path = instance_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=4))

    return "Resolved" if report[instance_id]["resolved"] else "Unresolved"
