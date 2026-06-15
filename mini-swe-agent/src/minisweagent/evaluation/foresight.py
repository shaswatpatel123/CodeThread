#!/usr/bin/env python3
"""Evaluator for the foresight anticipation-gap benchmark.

Instead of the SWE-bench eval stack (conda + per-repo specs + text-log parsers),
this drives the eval primitive baked into every foresight image:

    run_state.sh <checkout_sha> <overlay_sha|-> <junit_out> <test_file...>

which checks out a commit, optionally overlays test files from another commit,
builds a fresh uv venv, installs the package, and runs pytest --junitxml. Django
images expose run_state_django.sh with identical CLI that converts runtests.py
output to the SAME (classname, name)-keyed JUnit XML.

Grading flow (the agent's patch has already been applied to /testbed by the runner):
  1. Commit the agent's tree -> AGENT_SHA.
  2. PR1 score: run_state.sh AGENT_SHA <pr1_head_sha> pr1.xml <pr1_test_files>
     (overlay pulls PR1's canonical tests onto the agent's solution).
  3. Anticipation score: run_state.sh AGENT_SHA <pr2_head_sha> ant.xml
     <anticipation_test_files> (overlay pulls the future edge-case tests).
  4. gap = pass%(FAIL_TO_PASS) - pass%(ANTICIPATION_FAIL_TO_PASS).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

_JUNIT_MARKER = "<<<FORESIGHT_JUNIT>>>"
_PYTEST_HELPER = "run_state.sh"
_DJANGO_HELPER = "run_state_django.sh"


def parse_junit(xml_text: str) -> dict[str, str]:
    """Parse a JUnit XML string into {"classname::name": status}.

    status is one of PASSED / FAILED / ERROR / SKIPPED. Matches the
    (classname, name) keying that both run_state.sh and run_state_django.sh emit.
    """
    results: dict[str, str] = {}
    if not xml_text or not xml_text.strip():
        return results
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results
    for tc in root.iter("testcase"):
        classname = tc.attrib.get("classname", "")
        name = tc.attrib.get("name", "")
        tid = f"{classname}::{name}" if classname else name
        if tc.find("failure") is not None:
            status = "FAILED"
        elif tc.find("error") is not None:
            status = "ERROR"
        elif tc.find("skipped") is not None:
            status = "SKIPPED"
        else:
            status = "PASSED"
        results[tid] = status
    return results


def _lookup(results: dict[str, str], test_id: str) -> str:
    """Find a test's status, tolerant of parametrized-id mismatches.

    Tries: exact id, then id with any ``[param]`` suffix stripped on both sides,
    then a name-only (post-``::``) fallback.
    """
    if test_id in results:
        return results[test_id]
    base = test_id.split("[", 1)[0]
    for k, v in results.items():
        if k.split("[", 1)[0] == base:
            return v
    # name-only fallback (last ::-segment)
    want = test_id.split("::")[-1].split("[", 1)[0]
    for k, v in results.items():
        if k.split("::")[-1].split("[", 1)[0] == want:
            return v
    return "MISSING"


def _score(results: dict[str, str], test_ids: list[str]) -> dict:
    per_test = {tid: _lookup(results, tid) for tid in test_ids}
    passed = sum(1 for s in per_test.values() if s == "PASSED")
    total = len(test_ids)
    pct = (passed / total * 100.0) if total else 0.0
    return {"total": total, "passed": passed, "pct": round(pct, 2), "per_test": per_test}


def _run_overlay(env, checkout_sha: str, overlay_sha: str, test_files: list[str],
                 out_xml: str, helper: str, timeout: int = 1800):
    """Invoke the baked eval primitive and retrieve the JUnit XML via stdout.

    NOTE: out_xml MUST be unique per instance+side. Apptainer's --contain shares
    /tmp across concurrent containers, so a fixed path (e.g. /tmp/pr1.xml) causes
    workers to clobber each other's JUnit file -> cross-contaminated results.
    """
    files = " ".join(test_files)
    ov = overlay_sha if overlay_sha else "-"
    cmd = (
        f"cd {env.config.cwd} && rm -f {out_xml}; {helper} {checkout_sha} {ov} {out_xml} {files}; "
        f"echo '{_JUNIT_MARKER}'; cat {out_xml} 2>/dev/null || true"
    )
    out, timed_out, _, rc = env.exec_run_with_timeout(cmd, timeout=timeout)
    xml_text = out.split(_JUNIT_MARKER, 1)[1].strip() if _JUNIT_MARKER in out else ""
    return xml_text, out, timed_out


def evaluate_foresight(instance: dict, instance_dir: Path, env) -> str:
    """Run PR1 + anticipation test sets against the agent's solution; write report.

    Returns a short exit-status string. Assumes the agent's patch is already
    applied to /testbed (runner did env.apply_patch) and .git is intact
    (grade phase uses skip_git_reset=True).
    """
    instance_dir.mkdir(parents=True, exist_ok=True)
    instance_id = instance["instance_id"]
    runner = instance.get("runner", "pytest")
    helper = _DJANGO_HELPER if runner == "django" else _PYTEST_HELPER

    # 1. Commit the agent's tree to get a stable SHA the helper can check out.
    commit_cmd = (
        f"cd {env.config.cwd} && git config user.email 'foresight@eval' "
        f"&& git config user.name 'foresight' && git add -A "
        f"&& git commit -m foresight-agent-solution --no-verify >/dev/null 2>&1; "
        f"git rev-parse HEAD"
    )
    out, timed_out, _, rc = env.exec_run_with_timeout(commit_cmd, timeout=300)
    agent_sha = out.strip().splitlines()[-1].strip() if out.strip() else ""
    if rc != 0 or len(agent_sha) < 7:
        report = {instance_id: {
            "instance_id": instance_id, "repo": instance.get("repo"), "runner": runner,
            "error": "commit_failed", "commit_output": out, "resolved": False,
        }}
        (instance_dir / "foresight_report.json").write_text(json.dumps(report, indent=2))
        return "commit_failed"

    # In-container JUnit paths MUST be unique per instance: apptainer --contain
    # shares /tmp across concurrent workers, so a fixed name cross-contaminates.
    # /testbed is the repo dir (private per sandbox); run_state.sh git-cleans it,
    # so write the junit just above it.
    safe = instance_id.replace("/", "_")
    pr1_out = f"/foresight_{safe}_pr1.xml"
    ant_out = f"/foresight_{safe}_ant.xml"

    # 2. PR1 canonical tests (overlay from pr1_head).
    pr1_xml, pr1_log, pr1_to = _run_overlay(
        env, agent_sha, instance["pr1_head_sha"], instance["pr1_test_files"],
        pr1_out, helper)
    # 3. Anticipation tests (overlay from pr2_head).
    ant_xml, ant_log, ant_to = _run_overlay(
        env, agent_sha, instance["pr2_head_sha"], instance["anticipation_test_files"],
        ant_out, helper)

    (instance_dir / "pr1_junit.xml").write_text(pr1_xml)
    (instance_dir / "ant_junit.xml").write_text(ant_xml)
    (instance_dir / "test_output.txt").write_text(
        f"=== PR1 ({helper}) ===\n{pr1_log}\n\n=== ANTICIPATION ===\n{ant_log}")

    pr1_results = parse_junit(pr1_xml)
    pr1 = _score(pr1_results, instance["FAIL_TO_PASS"])
    # P2P (regression) tests live in the SAME pr1_test_files, so they're already in
    # pr1_xml — score them from the same JUnit, no extra container run.
    p2p = _score(pr1_results, instance.get("PASS_TO_PASS", []))
    ant = _score(parse_junit(ant_xml), instance["ANTICIPATION_FAIL_TO_PASS"])
    gap = round(pr1["pct"] - ant["pct"], 2)

    # SWE-bench-style resolution: PR1 fail->pass AND no regressions. Empty P2P set
    # is vacuously satisfied (p2p.total == 0).
    p2p_ok = p2p["total"] == 0 or p2p["pct"] == 100.0
    pr1_resolved = pr1["pct"] == 100.0
    resolved = pr1_resolved and p2p_ok

    report = {instance_id: {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "runner": runner,
        "agent_sha": agent_sha,
        "pr1": pr1,
        "p2p": p2p,
        "anticipation": ant,
        "gap": gap,
        "pr1_resolved": pr1_resolved,        # F2P only
        "no_regressions": p2p_ok,            # P2P all pass (or none)
        "anticipated": ant["pct"] == 100.0,
        "resolved": resolved,                # SWE-bench bar: F2P==100 AND P2P==100
        "timed_out": bool(pr1_to or ant_to),
    }}
    (instance_dir / "foresight_report.json").write_text(json.dumps(report, indent=2))
    return f"PR1={pr1['pct']}% ANT={ant['pct']}% gap={gap}%"


def make_foresight_report(predictions: dict, full_dataset: list, output_path: Path) -> Path:
    """Aggregate per-instance foresight_report.json into the gap headline metric."""
    per = []
    repo_acc: dict[str, list] = {}
    micro_pr1_pass = micro_pr1_tot = micro_ant_pass = micro_ant_tot = 0
    micro_p2p_pass = micro_p2p_tot = 0

    for inst in full_dataset:
        iid = inst["instance_id"]
        if iid not in predictions:
            continue
        rf = output_path / iid / "foresight_report.json"
        if not rf.exists():
            continue
        r = json.loads(rf.read_text()).get(iid, {})
        if "pr1" not in r:
            continue
        r.setdefault("p2p", {"total": 0, "passed": 0, "pct": 0.0})  # back-compat
        r.setdefault("no_regressions", r["p2p"]["total"] == 0 or r["p2p"]["pct"] == 100.0)
        per.append(r)
        repo_acc.setdefault(r.get("repo", "?"), []).append(r)
        micro_pr1_pass += r["pr1"]["passed"]; micro_pr1_tot += r["pr1"]["total"]
        micro_ant_pass += r["anticipation"]["passed"]; micro_ant_tot += r["anticipation"]["total"]
        micro_p2p_pass += r["p2p"]["passed"]; micro_p2p_tot += r["p2p"]["total"]

    n = len(per)

    def _mean(key_path):
        if not n:
            return 0.0
        vals = []
        for r in per:
            v = r
            for k in key_path:
                v = v[k]
            vals.append(v)
        return round(sum(vals) / n, 2)

    def _pct(a, b):
        return round(a / b * 100.0, 2) if b else 0.0

    by_repo = {}
    for repo, rs in sorted(repo_acc.items()):
        m = len(rs)
        by_repo[repo] = {
            "n": m,
            "mean_pr1_f2p": round(sum(x["pr1"]["pct"] for x in rs) / m, 2),
            "mean_p2p": round(sum(x["p2p"]["pct"] for x in rs) / m, 2),
            "mean_anticipation_f2p": round(sum(x["anticipation"]["pct"] for x in rs) / m, 2),
            "mean_gap": round(sum(x["gap"] for x in rs) / m, 2),
            "pr1_resolved": sum(1 for x in rs if x["pr1_resolved"]),
            "resolved": sum(1 for x in rs if x.get("resolved")),
            "anticipated": sum(1 for x in rs if x["anticipated"]),
        }

    report = {
        "graded_instances": n,
        "mean_pr1_f2p": _mean(["pr1", "pct"]),
        "mean_p2p": _mean(["p2p", "pct"]),
        "mean_anticipation_f2p": _mean(["anticipation", "pct"]),
        "mean_gap": _mean(["gap"]),
        "micro_pr1_f2p": _pct(micro_pr1_pass, micro_pr1_tot),
        "micro_p2p": _pct(micro_p2p_pass, micro_p2p_tot),
        "micro_anticipation_f2p": _pct(micro_ant_pass, micro_ant_tot),
        "micro_gap": round(_pct(micro_pr1_pass, micro_pr1_tot) - _pct(micro_ant_pass, micro_ant_tot), 2),
        "pr1_resolved_instances": sum(1 for r in per if r["pr1_resolved"]),
        "resolved_instances": sum(1 for r in per if r.get("resolved")),
        "anticipated_instances": sum(1 for r in per if r["anticipated"]),
        "by_repo": by_repo,
        "schema_version": 2,
    }
    print(f"Graded: {n}")
    print(f"Mean PR1 F2P:           {report['mean_pr1_f2p']}%")
    print(f"Mean P2P (regression):  {report['mean_p2p']}%")
    print(f"Mean Anticipation F2P:  {report['mean_anticipation_f2p']}%")
    print(f"Mean anticipation gap:  {report['mean_gap']}%")
    print(f"PR1 resolved (F2P):     {report['pr1_resolved_instances']}/{n}")
    print(f"Resolved (F2P & P2P):   {report['resolved_instances']}/{n}")
    print(f"Anticipated edge case:  {report['anticipated_instances']}/{n}")

    out = output_path / "final_foresight_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Report written to {out}")
    return out
