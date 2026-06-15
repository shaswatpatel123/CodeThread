# Environment Setup Guide

CodeThread runs on top of [mini-swe-agent](../mini-swe-agent), plus a small extra
stack for computing maintainability metrics across languages. There are two
pieces to install:

1. **The agent harness** — `mini-swe-agent` (drives patch generation & evaluation).
2. **The metrics toolchain** — Python analysis libraries + multi-language metric
   CLIs (used by the *Maintainability metrics* stage).

The `setup_env.sh` script in this folder installs both.

## Quick start

```bash
# Activate your conda/venv FIRST (the script installs into the active env)
cd env
./setup_env.sh                 # Python stack + mini-swe-agent + Rust/Go metric CLIs
# or
./setup_env.sh --python-only   # skip the Rust/Go CLIs (Python-only metrics)
```

## What `setup_env.sh` installs

| Step | Contents |
|------|----------|
| 1. Python packages | Analysis stack (`pandas`, `numpy`, `pyarrow`, `radon`, `complexipy`, `unidiff`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`, `datasets`, …) **and** `mini-swe-agent` installed editable from [`../mini-swe-agent`](../mini-swe-agent). |
| 2. External metric CLIs | `rust-code-analysis-cli` (via `cargo`) → CC/CogC/Halstead/MI for Rust, C, Java, JS, TS; `gocognit` (via `go install`) → Go cognitive complexity. |
| 3. `go-halstead` | Our custom stdlib-only Go tool, built from [`../mini-swe-agent/tools/go-halstead`](../mini-swe-agent/tools/go-halstead). |

`--python-only` stops after step 1. In that mode, non-Python languages fall back
to `lizard` (cyclomatic complexity only); Python metrics remain fully available.

## Prerequisites (must be on PATH before running)

- `python3` + `pip` — activate your conda/venv first (on an HPC cluster you may
  have a `source run_env.sh` step of your own).
- `cargo` (Rust toolchain) — only needed for `rust-code-analysis-cli` ([rustup.rs](https://rustup.rs)).
- `go` (≥ 1.25) — only needed for `gocognit` and `go-halstead`.

## After install

Make sure the Go/Rust binaries are on PATH (add to `~/.bashrc` if needed):

```bash
export PATH="$HOME/.cargo/bin:$HOME/.go/bin:$PATH"
```

The metrics pipeline locates its three non-Python CLIs via these env vars
(`calculate_metrics_multilingual.py`). Their built-in defaults are absolute paths
from the authors' machine, so **export these to point at your own binaries** —
otherwise non-Python languages silently fall back to `lizard` (cyclomatic
complexity only):

```bash
export RCA_CLI="$HOME/.cargo/bin/rust-code-analysis-cli"   # Rust / C / Java / JS / TS
export GO_HALSTEAD_CLI="$HOME/.go/bin/go-halstead"         # Go: Halstead + MI + CC
export GOCOGNIT_CLI="$HOME/.go/bin/gocognit"               # Go: cognitive complexity
```

`setup_env.sh` installs these binaries to exactly the paths shown above.

## Core agent only

If you only need to run the agent (patch generation / evaluation) and not the
metrics, install `mini-swe-agent` directly per its own docs:

```bash
cd ../mini-swe-agent && pip install -e .
```

See [`../mini-swe-agent/README.md`](../mini-swe-agent/README.md) for full
mini-swe-agent documentation, model configuration, and CLI usage.
