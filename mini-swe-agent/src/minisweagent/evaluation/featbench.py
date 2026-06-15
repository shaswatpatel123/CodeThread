from pathlib import Path, PurePosixPath
from minisweagent.environments import Environment
from unidiff import PatchSet
from minisweagent.run.extra.utils.exceptions import EvaluationTimeout
from minisweagent.evaluation.grading_featbench import get_eval_report
import json
import re


def get_test_directives(instance: dict) -> list:
    """
    Get test directives from the test_patch of a task instance.
    Used to reset test files to base commit before applying the oracle patch.
    """
    NON_TEST_EXTS = [
        ".json", ".png", "csv", ".txt", ".md",
        ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml",
    ]
    if instance["repo"] == "swe-bench/humaneval":
        return ["test.py"]

    diff_pat = r"diff --git a/.* b/(.*)"
    test_patch = instance["test_patch"]
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]

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
    """Return source files touched by a unified diff patch."""
    source_files = []
    for file in PatchSet(patch):
        if file.source_file != "/dev/null":
            source_files.append(file.source_file)
    return [x[2:] for x in source_files if x.startswith("a/")]


def get_oracle_test_ids(instance: dict) -> tuple[list[str], list[str]]:
    """
    Return (FAIL_TO_PASS, PASS_TO_PASS) as lists of fully-qualified pytest test IDs
    (e.g. 'tests/test_foo.py::TestBar::test_baz'). Handles both raw CSV strings and
    pre-split lists. Mirrors what TsinghuaISE/FeatBench feeds to pytest directly,
    instead of widening to whole test files (which can be 5-10x larger).
    """
    def _split(v):
        if not v:
            return []
        if isinstance(v, list):
            return [t.strip() for t in v if t and t.strip()]
        return [t.strip() for t in v.split(",") if t.strip()]

    return _split(instance.get("FAIL_TO_PASS")), _split(instance.get("PASS_TO_PASS"))


def get_p2p_test_files(instance: dict) -> list[str]:
    """
    Extract unique test file paths from the PASS_TO_PASS oracle.
    Handles both a comma-separated string (raw dataset) and a pre-split list
    (pre-processed by run/extra/featbench.py).
    """
    p2p = instance.get("PASS_TO_PASS") or ""
    entries: list[str] = p2p if isinstance(p2p, list) else p2p.split(",")

    seen: set[str] = set()
    files: list[str] = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        file_path = entry.split("::")[0]
        if file_path and file_path not in seen:
            seen.add(file_path)
            files.append(file_path)
    return files


def get_repo_setup_commands(repo: str) -> list[str]:
    """Repo-specific environment setup commands mirroring TsinghuaISE/FeatBench PR #9.

    Each block is sentinel-gated so re-running on the same container is a no-op.
    """
    repo_name = repo.split("/")[-1] if repo else ""
    name = repo_name.lower()

    if repo_name == "conan":
        # Conan's test/conftest.py hardcodes Linux CMake paths per version
        # (e.g. /usr/share/cmake-3.15.7/bin, /usr/share/cmake-3.16.9/bin, ...).
        # Symlink the system cmake into every 'Linux': '<path>' entry it names,
        # and also make sure a real 3.15 is available for the tests that need it.
        # Mirrors harbor-framework/harbor PR #1218 conan block.
        sentinel = "/usr/share/cmake-3.15.7/.conan_setup_done"
        cmd = r"""if [ ! -f SENTINEL ]; then
cmake_bin=$(command -v cmake 2>/dev/null)
if [ -n "$cmake_bin" ]; then
python3 -c "
import re
c = open('test/conftest.py').read()
for p in re.findall(r\"'Linux': ['\\\"]([^'\\\"]+)['\\\"]\", c):
    print(p)
" | while read -r path; do
    [ -z "$path" ] && continue
    # Skip the dir where cmake already lives (ln -sf would clobber the real
    # binary with a self-referencing symlink) and the 3.15.7 dir (handled
    # separately by the cmake-version branch below).
    [ "$path" = "$(dirname "$cmake_bin")" ] && continue
    [ "$path" = "/usr/share/cmake-3.15.7/bin" ] && continue
    mkdir -p "$path" && ln -sf "$cmake_bin" "$path/cmake"
done
cmake_ver=$(cmake --version 2>/dev/null | head -1)
if ! echo "$cmake_ver" | grep -q 'cmake version 3\.15'; then
    curl -fsSL https://cmake.org/files/v3.15/cmake-3.15.7-Linux-x86_64.sh -o /tmp/cmake315.sh && \
    chmod +x /tmp/cmake315.sh && \
    /tmp/cmake315.sh --prefix=/usr/local --skip-license && \
    mkdir -p /usr/share/cmake-3.15.7/bin && \
    ln -sf /usr/local/bin/cmake /usr/share/cmake-3.15.7/bin/cmake
else
    mkdir -p /usr/share/cmake-3.15.7/bin && \
    ln -sf "$cmake_bin" /usr/share/cmake-3.15.7/bin/cmake
fi
fi
touch SENTINEL
fi || true"""
        return [cmd.replace("SENTINEL", sentinel)]
    if repo_name == "tox":
        sentinel = "/tmp/.tox_env_setup_done"
        return [f"[ -f {sentinel} ] || (pip install -e . && touch {sentinel})"]
    if name == "pybamm":
        sentinel = "/tmp/.pybamm_env_setup_done"
        return [f"[ -f {sentinel} ] || (pip install -e '.[all]' && touch {sentinel})"]
    if repo_name == "jupyter-ai":
        sentinel = "/tmp/.jupyter_ai_env_setup_done"
        return [
            f"[ -f {sentinel} ] || ("
            "npm install -g n && n 14 && hash -r && "
            "pip install -e \"packages/jupyter-ai-magics[test]\" "
            "-e \"packages/jupyter-ai-test[test]\" "
            "-e \"packages/jupyter-ai[test]\" && "
            f"touch {sentinel})"
        ]
    return []


def get_eval_script_list_py(instance: dict, specs, cwd: str) -> list:
    HEREDOC_DELIMITER = "EOF_114329324912"

    test_patch = instance["test_patch"]
    test_files = get_modified_files(test_patch)

    base_commit = instance["base_commit"]

    # Reset oracle test files so any agent edits to them are discarded
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )

    # Run the *exact* F2P + P2P test IDs (matching TsinghuaISE/FeatBench), not whole
    # files. P2P sets can be 1k-4k tests in 50+ files; running whole files inflates
    # runtime by 5-10x and lets unrelated breakage in those files cascade into hangs.
    f2p_ids, p2p_ids = get_oracle_test_ids(instance)
    seen = set(f2p_ids)
    all_test_ids = list(f2p_ids) + [t for t in p2p_ids if t not in seen and not seen.add(t)]

    # Bash ARG_MAX is ~128KB; chunk to keep each invocation well under that and
    # combine with `;` so a hang/crash in one chunk doesn't lose the others.
    MAX_CMD_BYTES = 100_000
    base_cmd = specs["test_cmd"]
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_len = len(base_cmd) + 1
    for tid in all_test_ids:
        add = len(tid) + 1
        if cur and cur_len + add > MAX_CMD_BYTES:
            chunks.append(cur)
            cur, cur_len = [], len(base_cmd) + 1
        cur.append(tid)
        cur_len += add
    if cur:
        chunks.append(cur)
    if not chunks:
        chunks = [[]]  # still emit one pytest call so START/END markers are produced

    test_command = " ; ".join(" ".join([base_cmd, *chunk]) for chunk in chunks)

    START_TEST_OUTPUT = ">>>>> Start Test Output"
    END_TEST_OUTPUT = ">>>>> End Test Output"

    eval_commands = []
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {cwd}",
        "git status",
        "git show",
        f"git -c core.fileMode=false diff {base_commit}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])

    # Install pytest-xdist and pytest-timeout (per-test timeout to prevent hangs)
    eval_commands.append("pip install pytest-xdist pytest-timeout")

    # Repo-specific env setup (conan cmake, tox/pybamm/jupyter-ai editable installs).
    # Mirrors TsinghuaISE/FeatBench PR #9. Sentinel-gated, safe to re-run.
    eval_commands += get_repo_setup_commands(instance.get("repo", ""))

    # Redirect TMPDIR into the writable workdir so temp files created by tests
    # (e.g. conan's "path with spaces" fixtures) don't hit Apptainer fakeroot
    # permission issues under /tmp.
    # Redirect TMPDIR into the writable workdir so temp files created by tests
    # (e.g. conan's "path with spaces" fixtures) don't hit Apptainer fakeroot
    # permission issues under /tmp.  Add _tmp/ to .gitignore so the
    # maintainability metrics step (git ls-files --others) doesn't pick them up.
    eval_commands += [
        f"mkdir -p {cwd}/_tmp",
        f"export TMPDIR={cwd}/_tmp",
        f"echo '_tmp/' >> {cwd}/.gitignore",
        # Prevent git from walking upward out of tempdirs into the outer repo.
        # Conan's TestClient creates packages under TMPDIR that are inside cwd;
        # without this, `revision_mode='scm'` tests that expect "no git" instead
        # resolve the outer .git and succeed, inverting the expected failure.
        f"export GIT_CEILING_DIRECTORIES={cwd}",
        reset_tests_command,
        apply_test_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
    ]
    return eval_commands


def get_eval_script_list(instance: dict, specs, cwd: str) -> list:
    return get_eval_script_list_py(instance, specs, cwd)


def evaluation(instance: dict, instance_dir: Path, patch: str, env: Environment):
    patch_file = instance_dir / "patch.diff"
    patch_file.write_text(patch)

    base_cmd_template = "python3 -m pytest -q -rA --tb=no -n auto --timeout=10 --timeout-method=thread --continue-on-collection-errors"

    # base_cmd_template = "python3 -m pytest -q -rA --tb=no --continue-on-collection-errors"
    instance["test_cmd"] = base_cmd_template

    eval_script_list = get_eval_script_list(instance, instance, env.config.cwd)
    eval_script = "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_script_list) + "\n"

    eval_file = instance_dir / "eval.sh"
    eval_file.write_text(eval_script)
    env.copy_to_container(eval_file, PurePosixPath(f"{env.config.cwd}/eval.sh"))

    timeout = 14_400  # seconds
    test_output, timed_out, total_runtime, returncode = env.exec_run_with_timeout(
        f"/bin/bash {env.config.cwd}/eval.sh", timeout
    )
    test_output_path = instance_dir / "test_output.txt"

    with open(test_output_path, "w") as f:
        if isinstance(test_output, bytes):
            test_output = test_output.decode("utf-8", errors="replace")
            
        f.write(test_output)
        if timed_out:
            f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
            raise EvaluationTimeout()


def report(instance: dict, instance_dir: Path, patch: str, test_output_path: Path):
    report_path = instance_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    result = get_eval_report(
        test_spec=instance,
        patch=patch,
        test_log_path=test_output_path,
        include_tests_status=True,
    )

    with open(report_path, "w") as f:
        f.write(json.dumps(result, indent=4))

    done_path = instance_dir / "done.txt"
    with open(done_path, "w") as f:
        f.write("done")

    return "Resolved" if result[instance["instance_id"]]["resolved"] else "Unresolved"

