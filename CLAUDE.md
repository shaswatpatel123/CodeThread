# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**CodeThread** is the research artifact for the paper *"Is Agent Code Less Maintainable Than Human Code?"* It is a framework that turns repository-level coding benchmarks (SWE-bench Verified, SWE-bench Multilingual, SWE-bench Pro, FeatBench) into controlled *maintenance* experiments. For each task it builds a two-PR chain on the same repo and compares resolving PR 2 on top of **agent-written** PR 1 vs **human-written (gold)** PR 1. The diff in PR 2 resolve rate isolates how maintainable the underlying code is.

This is an experiment/data pipeline, not a deployable application. Most work happens by running CLI entry points and editing the wrapper scripts that drive them — not by serving anything.

## Repository layout (the big picture)

- `mini-swe-agent/` — a **vendored fork** of [mini-swe-agent](https://github.com/SWE-agent/mini-SWE-agent) (MIT). This is where the agent harness *and* all CodeThread-specific run/eval logic live. Edit here for harness behavior. Installed editable.
- `dataset/scripts/` — two-stage dataset generation pipeline that stubs out gold-patch function bodies and generates problem statements (PR0 → PR1 task construction). See `dataset/scripts/README.md`.
- `scripts/` — ready-to-edit Bash wrappers, **one per benchmark**, that string the full chain end-to-end (resolve PR1 → prep chain → resolve PR2 → grade both arms). These are HPC/Slurm-flavored with hardcoded author paths — read and edit the variables at the top before running.
- `scripts/utils/` — chain-wiring helpers: `synthetic_chains.py` → `get_instance_ids.py` → `generate_2ndPRJson.py` (produces `secondPRMapper.json`, the `--init-patch-map` input that stacks PR2 on PR1).
- `scripts/maintability/`, `scripts/gold/` — metric recompute wrappers and the human-base (gold) baseline runner.
- `env/setup_env.sh` — installs the agent + the multi-language maintainability metrics toolchain. See `env/README.md`.

## CodeThread-specific code inside the vendored harness

The fork adds files beyond upstream mini-swe-agent — focus here for CodeThread logic:

- `src/minisweagent/run/extra/` — benchmark drivers reachable as `mini-extra <name>` subcommands: `swebench.py`, `swebenchpro.py`, `featbench.py`, plus `maintainability*.py` variants and chain helpers. Entry point: `minisweagent.run.mini_extra:main`.
- `src/minisweagent/evaluation/` — grading and metrics: `grading.py`, `reporting.py` (aggregates `--generate-final-report`), and `maintainability_metrics.py` (writes `maintainability_metrics.json` + `modified_functions.json` per instance). Language-specific metric backends are dispatched here.
- `src/minisweagent/config/extra/*.yaml` — model/agent config overrides used by the wrapper scripts (passed via `--config`).

Core upstream structure (`agents/`, `models/`, `environments/`, `config/`) is unchanged in spirit; each has its own `README.md`.

## Common commands

All commands below run from inside `mini-swe-agent/` unless noted.

```bash
# Install (editable). From repo root:
cd mini-swe-agent && pip install -e .          # core agent only
cd env && ./setup_env.sh                        # agent + metrics toolchain (Rust/Go CLIs)
./setup_env.sh --python-only                     # skip Rust/Go CLIs (Python-only metrics)

# Tests (pytest, asyncio auto-mode)
python -m pytest                                 # full suite
python -m pytest tests/run/test_swebench.py      # single file
python -m pytest tests/run/test_swebench.py::test_name   # single test
python -m pytest -k "not slow"                   # skip slow tests (slow marker defined in pyproject)

# Lint / format (ruff, line-length 120, target py310)
ruff check .
ruff format .
```

Patch generation + evaluation are driven by the `mini-extra` entry point (e.g. `mini-extra swebench`, `mini-extra swebenchpro`, `mini-extra featbench`). Don't invoke these ad-hoc — start from the matching wrapper in `scripts/` and edit the inline variables.

## Key `mini-extra` flags

- `--output <dir>` — where predictions/results are written.
- `--config <yaml>` — agent/model config (often from `config/extra/`).
- `--run-only-eval` — apply existing patches in the sandbox, run tests, grade (no generation).
- `--generate-final-report` — aggregate per-instance results via `evaluation/reporting.py`.
- `--gold` — evaluate human gold patches (the human-base arm / upper bound).
- `--init-patch-map <file>` — apply PR1 before the agent starts (stacks PR2 on PR1); typically `secondPRMapper.json`.
- `--filter-ids "id1|id2|..."` — restrict to specific instances.
- `--path-local-images <dir>` — use pre-pulled container images.
- `--multilingual` — multilingual benchmark variant.
- `--environment-class singularity` — sandbox backend (Singularity/Apptainer is the default; Docker can be added following the Singularity env as a template under `environments/`).

## Environment / model notes

- Model API keys via env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.), used as needed per agent.
- Self-hosted / open-weight models are served via vLLM through litellm's `hosted_vllm` provider and registered in `scripts/model_registry.json` (pricing + max_tokens per model).
- Sandboxing uses Singularity/Apptainer; runs expect benchmark container images locally (`--path-local-images`).
- The maintainability metric pipeline locates three non-Python CLIs via env vars whose **defaults are absolute paths from the authors' machine** — export them or non-Python languages silently fall back to `lizard` (cyclomatic complexity only):
  ```bash
  export RCA_CLI="$HOME/.cargo/bin/rust-code-analysis-cli"   # Rust/C/Java/JS/TS
  export GO_HALSTEAD_CLI="$HOME/.go/bin/go-halstead"          # Go: Halstead/MI/CC
  export GOCOGNIT_CLI="$HOME/.go/bin/gocognit"                # Go: cognitive complexity
  ```

## Working in this repo

- The `scripts/*.sh` wrappers contain hardcoded HPC paths (`/scratch/spp9399/...`, Apptainer cache dirs, Slurm `module purge`). Treat them as templates: copy/edit the top variables for your environment rather than running as-is.
- Not a git repository — there is no commit history or branch workflow here.
- Two parquet-passing dataset stages: `data_gen_patches.py` (CPU, stubs functions) → `data_gen_problems.py` (GPU/vLLM, writes `PR1_Problem_Statement`). Run via `dataset/scripts/run_problem.sh`.
