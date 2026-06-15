"""TestableSWEAgent — agent that can trigger mid-loop test execution.

The agent recognises a special marker command (`echo RUN_TESTS_AND_REPORT`) in
command output and, instead of forwarding that output to the LM, runs the
benchmark test suite inside the container, formats which tests are still
failing, and returns that formatted report as the observation.

Usage::

    from minisweagent.agents.testable import TestableSWEAgent, get_evaluator

    evaluator = get_evaluator("swebench")   # "featbench" | "swebenchpro"
    agent = TestableSWEAgent(
        model, env,
        instance=instance,
        evaluator=evaluator,
        **config.get("agent", {}),
    )
    exit_status, result = agent.run(task)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from jinja2 import Template

from minisweagent import Environment
from minisweagent.agents.default import AgentConfig, DefaultAgent, ExecutionTimeoutError
from minisweagent.evaluation.grading import get_eval_tests_report
from minisweagent.evaluation.parser_constants import MAP_REPO_TO_PARSER
from minisweagent.evaluation.python_constants import (
    FAIL_TO_PASS,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_PASS,
)

# ---------------------------------------------------------------------------
# BenchmarkEvaluator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BenchmarkEvaluator(Protocol):
    def run_tests(self, instance: dict, env: Environment) -> dict[str, str]:
        """Run tests in the container. Returns {test_name: status} map."""
        ...

    def get_gold_results(self, instance: dict) -> dict:
        """Returns {FAIL_TO_PASS: [test_names], PASS_TO_PASS: [test_names]}."""
        ...


# ---------------------------------------------------------------------------
# Concrete evaluator: SWE-bench
# ---------------------------------------------------------------------------


class SWEBenchEvaluator:
    def get_gold_results(self, instance: dict) -> dict:
        f2p = json.loads(instance[FAIL_TO_PASS])
        p2p = json.loads(instance[PASS_TO_PASS])
        return {FAIL_TO_PASS: f2p, PASS_TO_PASS: p2p}

    def run_tests(self, instance: dict, env: Environment) -> dict[str, str]:
        repo = instance["repo"]
        version = instance["version"]

        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]

        gold = self.get_gold_results(instance)
        all_tests = gold[FAIL_TO_PASS] + gold[PASS_TO_PASS]
        full_cmd = f"{test_cmd} {' '.join(all_tests)}"

        output = env.execute(full_cmd)
        log_parser = MAP_REPO_TO_PARSER[repo]
        return log_parser(output.get("output", ""), instance)


# ---------------------------------------------------------------------------
# Concrete evaluator: FeatBench
# ---------------------------------------------------------------------------


class FeatBenchEvaluator:
    def get_gold_results(self, instance: dict) -> dict:
        def _to_list(value) -> list[str]:
            if isinstance(value, list):
                return [x.strip() for x in value if str(x).strip()]
            return [x.strip() for x in str(value).split(",") if x.strip()]

        f2p = _to_list(instance.get(FAIL_TO_PASS, ""))
        p2p = _to_list(instance.get(PASS_TO_PASS, ""))
        return {FAIL_TO_PASS: f2p, PASS_TO_PASS: p2p}

    def run_tests(self, instance: dict, env: Environment) -> dict[str, str]:
        from minisweagent.evaluation.featbench import get_test_directives, get_p2p_test_files
        from minisweagent.evaluation.parser_constants import parse_log_pytest_combined

        base_cmd = "python3 -m pytest -q -rA --tb=no --timeout=240 --continue-on-collection-errors"

        f2p_directives = get_test_directives(instance)
        p2p_files = get_p2p_test_files(instance)
        all_directives = f2p_directives + [f for f in p2p_files if f not in f2p_directives]
        test_cmd = " ".join([base_cmd] + all_directives)

        output = env.execute(test_cmd)
        return parse_log_pytest_combined(output.get("output", ""), instance)


# ---------------------------------------------------------------------------
# Concrete evaluator: SWE-bench Multilingual
# ---------------------------------------------------------------------------


class SWEBenchMultilingualEvaluator:
    """Evaluator for SWE-bench Multilingual instances (Java, Go, C, JS, …).

    Mid-loop test flow:
    1. Reset test files to the clean HEAD state (reset_repo() already rewrote
       git history so HEAD == the clean benchmark snapshot).
    2. Apply the instance's ``test_patch`` so the oracle test cases exist.
    3. Run the per-repo test commands (from ``evaluation.constants``
       MAP_REPO_VERSION_TO_SPECS — the multilingual version that covers all 66
       repos) wrapped in START/END_TEST_OUTPUT sentinels.
    4. Reset test files back to HEAD.
    5. Parse the captured output with the multilingual log parser
       (``evaluation.log_parsers.MAP_REPO_TO_PARSER``).
    """

    def get_gold_results(self, instance: dict) -> dict:
        from minisweagent.evaluation.test_spec.test_spec import make_test_spec

        test_spec = make_test_spec(instance)
        return {FAIL_TO_PASS: test_spec.FAIL_TO_PASS, PASS_TO_PASS: test_spec.PASS_TO_PASS}

    def run_tests(self, instance: dict, env: Environment) -> dict[str, str]:
        import tempfile
        from pathlib import Path, PurePosixPath

        from minisweagent.evaluation.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
        from minisweagent.evaluation.log_parsers import MAP_REPO_TO_PARSER as MULTI_PARSER
        from minisweagent.evaluation.test_spec.test_spec import make_test_spec
        from minisweagent.evaluation.test_spec.utils import get_modified_files, get_test_cmds

        test_spec = make_test_spec(instance)
        repo_directory = "/testbed"
        test_patch = instance["test_patch"]

        # Files touched by the oracle test patch
        test_files = get_modified_files(test_patch)
        HEREDOC = "EOF_114329324912"

        # reset_repo() replaces .git with a fresh repo; HEAD is the clean snapshot.
        # Use that to restore test files before/after applying the oracle patch.
        if test_files:
            reset_cmd = f"git checkout HEAD -- {' '.join(test_files)}"
        else:
            reset_cmd = 'echo "No test files to reset"'

        apply_cmd = (
            f"git apply --verbose --reject - <<'{HEREDOC}'\n{test_patch}\n{HEREDOC}"
        )
        test_commands = get_test_cmds(instance)

        script_lines = [
            "#!/bin/bash",
            "set -uxo pipefail",
            f"cd {repo_directory}",
            reset_cmd,                       # ensure test files are at clean HEAD
            apply_cmd,                       # apply oracle test patch
            f": '{START_TEST_OUTPUT}'",
            *test_commands,
            f": '{END_TEST_OUTPUT}'",
            reset_cmd,                       # restore test files
        ]
        script = "\n".join(script_lines) + "\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            local_path = Path(f.name)

        try:
            env.copy_to_container(local_path, PurePosixPath("/tmp/mid_loop_test.sh"))
            test_output, timed_out, _, _ = env.exec_run_with_timeout(
                "bash /tmp/mid_loop_test.sh", timeout=600
            )
        finally:
            local_path.unlink(missing_ok=True)

        if timed_out:
            return {}

        # The multilingual log parsers accept (log: str, test_spec: TestSpec).
        # Pass the full output for robustness (some parsers scan the entire log).
        log_parser = MULTI_PARSER[instance["repo"]]
        return log_parser(test_output, test_spec)


# ---------------------------------------------------------------------------
# Concrete evaluator: SWE-bench Pro
# ---------------------------------------------------------------------------


class SWEBenchProEvaluator:
    def get_gold_results(self, instance: dict) -> dict:
        f2p = list(eval(instance.get("fail_to_pass", "[]")))
        p2p = list(eval(instance.get("pass_to_pass", "[]")))
        return {FAIL_TO_PASS: f2p, PASS_TO_PASS: p2p}

    def run_tests(self, instance: dict, env: Environment) -> dict[str, str]:
        selected = ",".join(eval(instance["selected_test_files_to_run"]))
        # Use the run_script.sh approach: simplified inline execution that runs
        # the selected test files and reads output.json produced by parser.py.
        # For mid-loop use we just run pytest over the selected files directly.
        test_cmd = f"python3 -m pytest -q -rA --tb=no --timeout=240 {selected.replace(',', ' ')}"
        output = env.execute(test_cmd)

        # Try to read structured output.json if available (written by parser.py)
        cat_result = env.execute("cat /workspace/output.json 2>/dev/null || echo '{}'")
        try:
            parsed = json.loads(cat_result.get("output", "{}") or "{}")
            tests = parsed.get("tests", [])
            if tests:
                return {x["name"]: x["status"] for x in tests}
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: parse pytest output directly
        from minisweagent.evaluation.parser_constants import parse_log_pytest_combined
        return parse_log_pytest_combined(output.get("output", ""), instance)


# ---------------------------------------------------------------------------
# Evaluator registry / factory
# ---------------------------------------------------------------------------

_EVALUATOR_REGISTRY: dict[str, type] = {
    "swebench": SWEBenchEvaluator,
    "multilingual": SWEBenchMultilingualEvaluator,
    "featbench": FeatBenchEvaluator,
    "swebenchpro": SWEBenchProEvaluator,
}


def get_evaluator(name: str) -> BenchmarkEvaluator:
    """Instantiate a BenchmarkEvaluator by name."""
    cls = _EVALUATOR_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(_EVALUATOR_REGISTRY)
        raise ValueError(f"Unknown evaluator {name!r}. Available: {available}")
    return cls()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_TEST_RESULT_TEMPLATE = """\
Test Results:
{% if f2p_failed %}FAIL_TO_PASS still failing ({{ f2p_failed|length }}):
{% for t in f2p_failed[:max_failures_shown] %}  - {{ t }}
{% endfor %}{% endif %}\
{% if p2p_failed %}PASS_TO_PASS regressions ({{ p2p_failed|length }}):
{% for t in p2p_failed[:max_failures_shown] %}  - {{ t }}
{% endfor %}{% endif %}\
{% if not f2p_failed and not p2p_failed %}All tracked tests passing!{% endif %}"""


@dataclass
class TestableSWEAgentConfig(AgentConfig):
    test_trigger_marker: str = "RUN_TESTS_AND_REPORT"
    report_fail_to_pass: bool = True
    report_pass_to_pass: bool = True
    max_failures_shown: int = 20
    test_result_template: str = field(default=_DEFAULT_TEST_RESULT_TEMPLATE)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TestableSWEAgent(DefaultAgent):
    """DefaultAgent extended with mid-loop test execution.

    When the agent issues ``echo RUN_TESTS_AND_REPORT`` the command output
    starts with the marker.  Instead of forwarding that output to the LM this
    agent runs the benchmark test suite, formats the failing tests, and returns
    that report as the observation so the agent can continue working.
    """

    def __init__(
        self,
        model,
        env: Environment,
        *,
        instance: dict,
        evaluator: BenchmarkEvaluator,
        config_class=TestableSWEAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.instance = instance
        self.evaluator = evaluator

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            raw = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=raw)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output="")
            )

        # Check if the first non-empty output line is the test trigger marker
        first_line = next(
            (ln.strip() for ln in output.get("output", "").splitlines() if ln.strip()),
            "",
        )
        if first_line == self.config.test_trigger_marker:
            return self._run_tests()

        self.has_finished(output)
        return output

    def _run_tests(self) -> dict:
        """Run tests, format results, return observation dict."""
        try:
            status_map = self.evaluator.run_tests(self.instance, self.env)
        except Exception as e:
            return {"output": f"Test run unavailable for this instance: {e}"}

        try:
            gold = self.evaluator.get_gold_results(self.instance)
            report = get_eval_tests_report(status_map, gold)
        except Exception as e:
            return {"output": f"Could not compute test report: {e}"}

        f2p_failed: list[str] = report[FAIL_TO_PASS]["failure"] if self.config.report_fail_to_pass else []
        p2p_failed: list[str] = report[PASS_TO_PASS]["failure"] if self.config.report_pass_to_pass else []

        rendered = Template(self.config.test_result_template).render(
            f2p_failed=f2p_failed,
            p2p_failed=p2p_failed,
            max_failures_shown=self.config.max_failures_shown,
        )
        return {"output": rendered}
