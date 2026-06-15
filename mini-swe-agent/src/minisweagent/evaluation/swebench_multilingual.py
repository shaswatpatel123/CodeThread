from pathlib import Path, PurePosixPath
from minisweagent import Environment
from unidiff import PatchSet
from minisweagent.evaluation.python_constants import MAP_REPO_VERSION_TO_SPECS
import re
from minisweagent.run.extra.utils.exceptions import EvaluationTimeout
import json

from minisweagent.evaluation.test_spec.test_spec import make_test_spec
from minisweagent.evaluation.test_spec.grading import get_eval_report
from minisweagent.evaluation.constants import (
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
)

def evaluation(instance: dict, instance_dir: Path, patch: str, env: Environment):
    # Write the patch to the instance directory
    patch_path = instance_dir / "patch.diff"
    patch_path.write_text(patch)

    instance_test_spec = make_test_spec(instance)
    
    # Apply the patch to the environment
    # env.apply_patch(patch_path)
    
    eval_file = Path(instance_dir / "eval.sh")
    eval_file.write_text(instance_test_spec.eval_script)

    env.copy_to_container(eval_file, PurePosixPath("/testbed/eval.sh"))

    timeout = 1_800 # in seconds
    test_output, timed_out, total_runtime, returncode = env.exec_run_with_timeout("chmod 777 /testbed/eval.sh && /bin/bash /testbed/eval.sh", timeout)

    test_output_path = instance_dir / "test_output.txt"

    if timed_out:
        timeout = str( timeout )
        test_output_path.write_text(f"\n\nTimeout error: {timeout} seconds exceeded.")
        raise EvaluationTimeout()
    
    test_output_path.write_text(test_output)

    # Grade the test output
    report = get_eval_report(
        test_spec=instance_test_spec,
        prediction={ KEY_INSTANCE_ID: instance_test_spec.instance_id, KEY_PREDICTION:  patch },
        test_log_path=test_output_path,
        include_tests_status=True,
    )

    report_path = instance_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=4))
    
    return "Resolved" if report[instance["instance_id"]]["resolved"] else "Unresolved"
