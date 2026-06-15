#!/usr/bin/env bash
#
# setup_env.sh — set up the MaintainableCoder analysis + metrics environment.
#
# Installs three things:
#   1. Python packages (analysis stack + mini-swe-agent, editable)
#   2. External metric CLIs for non-Python languages (rust-code-analysis-cli, gocognit)
#   3. Our custom Go tool (go-halstead) — built from tools/go-halstead/
#
# Usage:
#   ./setup_env.sh                 # install everything (Python + Rust + Go tools)
#   ./setup_env.sh --python-only   # skip the Rust/Go CLIs (Python-only metrics)
#
# Prerequisites you must have on PATH yourself:
#   - python3 + pip  (activate your conda/venv FIRST; on NYU HPC: `source run_env.sh`)
#   - cargo  (Rust toolchain)   -> only for rust-code-analysis-cli
#   - go     (>= 1.25)          -> only for gocognit + go-halstead
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ONLY=0
[[ "${1:-}" == "--python-only" ]] && PYTHON_ONLY=1

# Install dirs for the Go/Rust binaries (override by exporting before running).
GOBIN="${GOBIN:-$HOME/.go/bin}"
CARGO_BIN="${CARGO_HOME:-$HOME/.cargo}/bin"

echo "==> Project root: $PROJECT_ROOT"

# ----------------------------------------------------------------------------
# 1. Python packages
# ----------------------------------------------------------------------------
echo ""
echo "==> [1/3] Installing Python packages with pip ..."
echo "    (make sure you have already activated your conda/venv)"

python3 -m pip install --upgrade pip

# Analysis + maintainability-metrics stack (beyond mini-swe-agent's own deps).
python3 -m pip install \
  "pandas" "numpy" "pyarrow" \
  "radon" "complexipy" "unidiff" \
  "scipy" "statsmodels" "scikit-learn" \
  "matplotlib" "seaborn" \
  "google-genai" "tiktoken" "datasets"

# mini-swe-agent fork (editable). Pulls in its own deps: litellm, openai,
# rich, typer, textual, lizard, pyyaml, requests, jinja2, etc.
echo "==> Installing mini-swe-agent (editable) ..."
python3 -m pip install -e "$PROJECT_ROOT/../mini-swe-agent"

if [[ "$PYTHON_ONLY" == "1" ]]; then
  echo ""
  echo "==> --python-only set: skipping Rust/Go metric CLIs."
  echo "    NOTE: non-Python languages (Rust/C/Java/JS/TS/Go) will fall back to"
  echo "    lizard (cyclomatic complexity only). Python metrics are fully available."
  echo "==> Done."
  exit 0
fi

# ----------------------------------------------------------------------------
# 2. External metric CLIs (multi-language backends)
# ----------------------------------------------------------------------------
echo ""
echo "==> [2/3] Installing external metric CLIs ..."

# rust-code-analysis-cli -> Rust, C, Java, JS, TS (CC / CogC / Halstead / MI)
if command -v cargo >/dev/null 2>&1; then
  echo "==> Installing rust-code-analysis-cli via cargo ..."
  cargo install rust-code-analysis-cli
else
  echo "!! cargo not found — skipping rust-code-analysis-cli."
  echo "   Rust/C/Java/JS/TS will fall back to lizard (CC only)."
  echo "   Install Rust from https://rustup.rs then re-run."
fi

# gocognit -> Go cognitive complexity
if command -v go >/dev/null 2>&1; then
  echo "==> Installing gocognit via go install (-> $GOBIN) ..."
  GOBIN="$GOBIN" go install github.com/uudashr/gocognit/cmd/gocognit@latest
else
  echo "!! go not found — skipping gocognit AND go-halstead."
  echo "   Go metrics will be unavailable. Install Go >= 1.25 and re-run."
  exit 0
fi

# ----------------------------------------------------------------------------
# 3. go-halstead (our custom tool) — stdlib-only, just `go build`
# ----------------------------------------------------------------------------
echo ""
echo "==> [3/3] Building go-halstead (custom, no external deps) ..."
mkdir -p "$GOBIN"
( cd "$PROJECT_ROOT/../mini-swe-agent/tools/go-halstead" && go build -o "$GOBIN/go-halstead" . )
echo "==> go-halstead built -> $GOBIN/go-halstead"

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "==> Done. Make sure these are on PATH (add to ~/.bashrc if needed):"
echo "      export PATH=\"$CARGO_BIN:$GOBIN:\$PATH\""
echo ""
echo "    The metrics pipeline reads these env vars (defaults shown). Export them"
echo "    to point at the binaries installed above, or the code falls back to the"
echo "    authors' hard-coded /scratch paths:"
echo "      export RCA_CLI=$CARGO_BIN/rust-code-analysis-cli"
echo "      export GO_HALSTEAD_CLI=$GOBIN/go-halstead"
echo "      export GOCOGNIT_CLI=$GOBIN/gocognit"

