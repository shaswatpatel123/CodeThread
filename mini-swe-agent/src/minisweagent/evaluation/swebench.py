from pathlib import Path, PurePosixPath
from minisweagent import Environment
from unidiff import PatchSet
from minisweagent.evaluation.python_constants import MAP_REPO_VERSION_TO_SPECS
import re
from minisweagent.run.extra.utils.exceptions import EvaluationTimeout
from minisweagent.evaluation.grading import get_eval_report
import json

def get_test_directives(instance: dict) -> list:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
    Returns:
        directives (list): List of test directives
    """
    # TODO: Put in constants
    NON_TEST_EXTS = [
        ".json",
        ".png",
        "csv",
        ".txt",
        ".md",
        ".jpg",
        ".jpeg",
        ".pkl",
        ".yml",
        ".yaml",
        ".toml",
    ]
    # For seq2seq code repos, testing command is fixed
    if instance["repo"] == "swe-bench/humaneval":
        return ["test.py"]

    # Get test directives from test patch and remove non-test files
    diff_pat = r"diff --git a/.* b/(.*)"
    test_patch = instance["test_patch"]
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if instance["repo"] == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed

    return directives

def get_modified_files(patch: str) -> list[str]:
    """
    Get the list of modified files in a patch
    """
    source_files = []
    for file in PatchSet(patch):
        if file.source_file != "/dev/null":
            source_files.append(file.source_file)
    return  [x[2:] for x in source_files if x.startswith("a/")]
    
def get_eval_script_list_py( instance: dict, specs ) -> list:
    HEREDOC_DELIMITER = "EOF_114329324912"
    
    test_patch = instance["test_patch"]
    test_files = get_modified_files(test_patch)

    base_commit = instance["base_commit"]
    # Reset the test files so if agent has changed them then they will get reverted
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )
    test_command = " ".join(
        [
            specs["test_cmd"],
            *get_test_directives(instance),
        ]
    )


    # We are doing in SingularityEnv.run() 
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed"
    ]
    #     f"cd {repo_directory}",
    eval_commands = []

    START_TEST_OUTPUT = ">>>>> Start Test Output"
    END_TEST_OUTPUT = ">>>>> End Test Output"

    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        "git config --global --add safe.directory /testbed",  # for nonroot user
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git -c core.fileMode=false diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed"
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
    ]
    return eval_commands

def get_eval_script_list( instance: dict, specs ) -> list:
    # TODO: for making it multilingual check swe-bench code
    return get_eval_script_list_py( instance, specs )

def get_eval_script(instance: dict) -> str:
    version = instance.get("version")
    repo = instance["repo"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    eval_script_list = get_eval_script_list( instance, specs )    
    
    return (
        "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_script_list)
        + "\n"
    )


def evaluation( instance: dict, instance_dir: Path, patch: str, env: Environment ):
    patch_file = instance_dir / "patch.diff"
    eval_file = instance_dir / "eval.sh"
    test_output_path = instance_dir / "test_output.txt"

    eval_file.parent.mkdir(parents=True, exist_ok=True)
    eval_script = get_eval_script( instance )

    # Write to container
    eval_file.write_text(eval_script)
    patch_file.write_text(patch)

    # Move to agent.copy_to_container()
    env.copy_to_container(eval_file, PurePosixPath("/testbed/eval.sh")) # ( src, target )

    # TODO: Move to config
    timeout = 1_800 # in seconds
    test_output, timed_out, total_runtime, returncode = env.exec_run_with_timeout("/bin/bash /testbed/eval.sh", timeout)

    with open(test_output_path, "w") as f:
        f.write(test_output)
        if timed_out:
            timeout = str( timeout )
            f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
            raise EvaluationTimeout()


def report( instance: dict, instance_dir: Path, patch: str, test_log_path: Path ) -> str:

    report_path = instance_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = get_eval_report(
                test_spec=instance,
                patch=patch,
                test_log_path=test_log_path,
                include_tests_status=True,
            )

    with open(report_path, "w") as f:
        f.write(json.dumps(report, indent=4))

    return "Resolved" if report[ instance["instance_id"] ]["resolved"] else "Unresolved"
