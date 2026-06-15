#!/usr/bin/env python3

import sys
from importlib import import_module

from rich.console import Console

subcommands = [
    ("minisweagent.run.extra.config", ["config"], "Manage the global config file"),
    ("minisweagent.run.inspector", ["inspect", "i", "inspector"], "Run inspector (browse trajectories)"),
    ("minisweagent.run.github_issue", ["github-issue", "gh"], "Run on a GitHub issue"),
    ("minisweagent.run.extra.swebench", ["swebench"], "Evaluate on SWE-bench (batch mode)"),
    ("minisweagent.run.extra.swebench_single", ["swebench-single"], "Evaluate on SWE-bench (single instance)"),
    ("minisweagent.run.extra.maintainability", ["maintainability"], "Evaluate maintability (batch mode)"),
    ("minisweagent.run.extra.maintainability_fast", ["maintainability-fast"], "Evaluate maintainability (batch mode)"),
    ("minisweagent.run.extra.patch_generation", ["patch-generation"], "Generate patch (batch mode)"),
    ("minisweagent.run.extra.featurebench", ["featurebench"], "Evaluate on featurebench (batch mode)"),
    ("minisweagent.run.extra.featbench", ["featbench"], "Evaluate on featbench (batch mode)"),
    ("minisweagent.run.extra.swebenchpro", ["swebenchpro"], "Evaluate on SWEBenchPro (batch mode)"),
    ("minisweagent.run.extra.swebench_testable", ["swebench-testable"], "Evaluate on SWE-bench with mid-loop test feedback (TestableSWEAgent)"),
    ("minisweagent.run.extra.maintainability_fast_swebenchpro", ["maintainability-fast-swebenchpro"], "Evaluate maintainability for swebenchpro"),
    ("minisweagent.run.extra.maintainability_fast_swebench_multi", ["maintainability-fast-swebench-multi"], "Evaluate maintainability for swebench multilingual"),
    ("minisweagent.run.extra.maintainability_fast_featbench", ["maintainability-fast-featbench"], "Evaluate maintainability for featbench"),
    ("minisweagent.run.extra.swebench_old", ["swebench-old"], "SWE-Bench old code"),
]


def get_docstring() -> str:
    lines = [
        "This is the [yellow]central entry point for all extra commands[/yellow] from mini-swe-agent.",
        "",
        "Available sub-commands:",
        "",
    ]
    for _, aliases, description in subcommands:
        alias_text = " or ".join(f"[bold green]{alias}[/bold green]" for alias in aliases)
        lines.append(f"  {alias_text}: {description}")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]

    if len(args) == 0 or len(args) == 1 and args[0] in ["-h", "--help"]:
        return Console().print(get_docstring())

    for module_path, aliases, _ in subcommands:
        if args[0] in aliases:
            return import_module(module_path).app(args[1:], prog_name=f"mini-extra {aliases[0]}")

    return Console().print(get_docstring())


if __name__ == "__main__":
    main()
