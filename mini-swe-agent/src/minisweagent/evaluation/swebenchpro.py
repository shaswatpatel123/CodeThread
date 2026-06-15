from pathlib import Path, PurePosixPath
from minisweagent.environments import Environment
import os
import re
import json
from minisweagent.run.extra.utils.exceptions import EvaluationTimeout
from unidiff import PatchSet
from .grading import get_eval_tests_report, get_resolution_status
from .python_constants import FAIL_TO_PASS, PASS_TO_PASS, KEY_INSTANCE_ID, ResolvedStatus



def load_base_docker(docker_scripts_dir, iid):
    with open(f"{docker_scripts_dir}/dockerfiles/base_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()


def instance_docker(docker_scripts_dir, iid):
    with open(f"{docker_scripts_dir}/dockerfiles/instance_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()


def load_local_script(scripts_dir, instance_id, script_name):
    """Load a script file from local scripts directory."""
    script_path = os.path.join(scripts_dir, instance_id, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    with open(script_path, 'r') as f:
        return f.read()


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch

    sections = re.split(r'(?=^diff --git )', patch, flags=re.MULTILINE)

    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r'^Binary files .* differ$', section, re.MULTILINE):
            continue
        if re.search(r'^GIT binary patch$', section, re.MULTILINE):
            continue
        kept.append(section)

    return "".join(kept)

def get_modified_files(patch: str) -> list[str]:
    """
    Get the list of modified files in a patch
    """
    source_files = []
    for file in PatchSet(patch):
        if file.source_file != "/dev/null":
            source_files.append(file.source_file)
    return  [x[2:] for x in source_files if x.startswith("a/")]


def create_entryscript(sample, docker_scripts_dir):
    before_repo_set_cmd = sample["before_repo_set_cmd"].strip().split("\n")[-1]
    selected_test_files_to_run = ",".join(eval(sample["selected_test_files_to_run"]))
    base_commit = sample["base_commit"]
    base_dockerfile = load_base_docker(docker_scripts_dir, sample["instance_id"])
    instance_dockerfile = instance_docker(docker_scripts_dir, sample["instance_id"])

    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                env_cmd = line.replace("ENV", "export", 1)
                env_cmds.append(env_cmd)

    env_cmds = "\n".join(env_cmds)

    test_patch = sample["test_patch"]
    test_files = get_modified_files(test_patch)
    reset_tests_command = (
        f"git checkout {base_commit} {' '.join(test_files)}"
        if test_files else ""
    )


    entry_script = f"""
{env_cmds}
# apply patch
cd /app
{reset_tests_command}
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
cat /workspace/stdout.log
cat /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    return entry_script


def evaluation(instance: dict, instance_dir: Path, patch: str, env: Environment, scripts_dir: str | None = None, docker_scripts_dir: str | None = None) -> str:
    """Evaluate a SWEBenchPro instance.

    Mirrors the interface of swebench_multilingual.evaluation but implements
    the SWEBenchPro-specific workflow: writes run_script.sh, parser.py and an
    entryscript to the container's /workspace/, executes them, then grades the
    resulting output.json against the instance's fail_to_pass / pass_to_pass
    test lists.

    Args:
        instance:     Dataset row for the instance being evaluated.
        instance_dir: Local directory used to persist artefacts (logs, patches…).
        patch:        Git-format patch string produced by the agent.
        env:          Running container environment.
        scripts_dir:  Path to the directory that contains per-instance run
                      scripts.  Falls back to instance["scripts_dir"] when
                      omitted.

    Returns:
        "Resolved" if all required tests pass, "Unresolved" otherwise.
    """
    uid = instance["instance_id"]

    if scripts_dir is None:
        scripts_dir = instance.get("scripts_dir")
    if scripts_dir is None:
        raise ValueError(
            f"scripts_dir must be provided as an argument or via instance['scripts_dir'] for {uid}"
        )

    # Strip binary hunks before doing anything with the patch
    cleaned_patch = strip_binary_hunks(patch)
    if cleaned_patch != patch:
        print(f"Stripped binary diff hunks from patch for {uid}")

    # Persist the patch locally
    patch_path = instance_dir / "patch.diff"
    patch_path.write_text(cleaned_patch)

    # Load per-instance run scripts from the scripts directory
    try:
        run_script = load_local_script(scripts_dir, uid, "run_script.sh")
        parser_script = load_local_script(scripts_dir, uid, "parser.py")
    except FileNotFoundError as e:
        print(f"Error loading scripts for {uid}: {e}")
        raise

    entryscript_content = create_entryscript(instance, docker_scripts_dir)

    # Write all workspace files locally so they can be copied to the container
    workspace_dir = instance_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    workspace_files = {
        "patch.diff": cleaned_patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript_content,
    }
    for rel_path, content in workspace_files.items():
        (workspace_dir / rel_path).write_text(content)

    # Save a top-level copy of the entryscript for easier inspection
    (instance_dir / "entryscript.sh").write_text(entryscript_content)

    # Copy workspace files into the container
    print( env.execute("mkdir -p /workspace") )
    for rel_path in workspace_files:
        env.copy_to_container(
            workspace_dir / rel_path,
            PurePosixPath(f"/workspace/{rel_path}"),
        )

    # Run the entryscript with a generous timeout (matches the oracle's 1-hour cap)
    timeout = 3_600
    test_output, timed_out, total_runtime, returncode = env.exec_run_with_timeout(
        "bash /workspace/entryscript.sh", timeout
    )

    print( test_output , "\nreturn code:", returncode)

    test_output_path = instance_dir / "test_output.txt"
    if timed_out:
        test_output_path.write_text(test_output + f"\n\nTimeout error: {timeout} seconds exceeded.")
        raise EvaluationTimeout()

    test_output_path.write_text(test_output)

    if returncode != 0:
        print(f"Entryscript failed for {uid} with return code: {returncode}")

    # Read output.json that the parser script deposited in the container
    cat_result = env.execute("cat /workspace/output.json")
    if cat_result["returncode"] != 0:
        print(f"Warning: output.json not found in container for {uid}.")
        return "Unresolved"

    try:
        output = json.loads(cat_result["output"])
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse output.json for {uid}: {e}")
        return "Unresolved"

    # Persist output.json locally
    (instance_dir / "output.json").write_text(json.dumps(output, indent=4))

    # Grade: all fail_to_pass and pass_to_pass tests must be in the passed set
    eval_status_map = {x["name"]: x["status"] for x in output.get("tests", [])}

    f2p = set(eval(instance.get("fail_to_pass", "[]")))
    p2p = set(eval(instance.get("pass_to_pass", "[]")))
    eval_ref = {
        KEY_INSTANCE_ID: uid,
        FAIL_TO_PASS: f2p,
        PASS_TO_PASS: p2p,
    }
    tests_report = get_eval_tests_report(eval_status_map, eval_ref)
    resolved = get_resolution_status(tests_report) == ResolvedStatus.FULL.value

    report = {
        uid: {
            "patch_is_None": False,
            "patch_exists": True,
            "patch_successfully_applied": True,
            "resolved": resolved,
            "tests_status": tests_report,
        }
    }
    (instance_dir / "report.json").write_text(json.dumps(report, indent=4))

    return "Resolved" if resolved else "Unresolved"
