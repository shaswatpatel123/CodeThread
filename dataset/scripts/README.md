# CodeThread — Dataset Generation Pipeline

This directory builds the **CodeThread** dataset: starting from existing
repository-level coding benchmarks, it produces *stubbed* tasks where a function
body has been emptied out, together with a natural-language problem statement
describing what that function must do. These tasks are the building blocks
CodeThread uses to construct controlled human-vs-agent maintenance experiments
(the **PR0 → PR1 → PR2** chain).

The pipeline runs in **two stages**:

| Stage | Script | Compute | Input | Output |
|-------|--------|---------|-------|--------|
| 1. Patch processing | `data_gen_patches.py` | CPU, no LLM | benchmark dataset(s) | intermediate parquet |
| 2. Problem statements | `data_gen_problems.py` | GPU (vLLM) | intermediate parquet | final parquet |

Both stages are wired together in [`run_problem.sh`](run_problem.sh) — edit the
config variables at the top and run `bash run_problem.sh`.

---

## Stage 1 — `data_gen_patches.py` (CPU)

Turns each benchmark instance's **gold patch** into a *stubbed starting point*.
No LLM or GPU is required.

For every instance it:

1. **Loads & normalizes the dataset** into a common schema (see
   [Dataset schema](#dataset-schema)). Multiple datasets can be loaded and
   concatenated in a single run.
2. **Clones the repository** and checks out `base_commit` (skippable with
   `--skip_clone` if the repos already exist locally; falls back to
   reconstructing context from the patch when a repo is unavailable).
3. **Analyzes the gold patch** to find every function/method the patch touches.
   Language is detected from the file extension; parsing uses tree-sitter where
   available with regex/brace/indent fallbacks, so it works across Python, Java,
   JS/TS, Go, Rust, C/C++, Ruby, and more.
4. **Generates `PR0_Patch`** — a patch that replaces each modified function's
   **body** with a language-appropriate stub (e.g. `# Write Your Code Here\n
   pass` for Python, `throw new UnsupportedOperationException(...)` for Java).
   Signatures, class structure, imports, and the rest of the file are preserved.
5. **Captures `original_functions`** — the real source of each modified function,
   kept as the reference the model (Stage 2) and downstream solvers must
   reproduce.

**Columns added to the parquet:**

| Column | Description |
|--------|-------------|
| `functions_modified` | List of function/method names the gold patch changes |
| `file_to_function_mapping` | `{file_path: [function names]}` |
| `PR0_Patch` | Diff that stubs out the modified function bodies (the task's starting point) |
| `original_functions` | `{function: source code}` — the ground-truth implementation |

### Usage

```bash
python data_gen_patches.py \
    --datasets all \
    --output ./data/patches.parquet \
    --repo_base_path cloned_repos \
    --clone_workers 4
```

| Flag | Default | Notes |
|------|---------|-------|
| `--output` | *(required)* | Intermediate parquet path (input to Stage 2) |
| `--datasets` | `multilingual` | One or more keys, a custom HF name, or `all` |
| `--repo_base_path` | `cloned_repos` | Where repos are cloned |
| `--clone_workers` | `4` | Parallel clone workers |
| `--skip_clone` | off | Reuse repos already present at `--repo_base_path` |
| `--num_instances` | all | Process only the first *N* instances per dataset (handy for smoke tests) |
| `--split` | per-dataset | Override the dataset split |

Built-in dataset keys (`KNOWN_DATASETS`): `verified`, `multilingual`, `pro`,
`featbench`.

---

## Stage 2 — `data_gen_problems.py` (GPU / vLLM)

Reads the intermediate parquet and, for each instance, generates a detailed
**problem statement** describing the stubbed functions — *without revealing the
original code*. This is the task description a human or agent sees when resolving
the first PR (PR1).

For each instance it builds a prompt from `original_functions` and
`file_to_function_mapping` (see `PROBLEM_STATEMENT_PROMPT` in the script),
asking the model for a structured spec: task overview, per-function summary,
args/returns/raises, implementation steps, and edge cases. Generation is batched
through a local vLLM engine.

**Column added to the parquet:**

| Column | Description |
|--------|-------------|
| `PR1_Problem_Statement` | Natural-language spec for re-implementing the stubbed functions |

### Usage

```bash
python data_gen_problems.py \
    --input ./data/patches.parquet \
    --output ./data/dataset_with_problem_statements.parquet \
    --model_name zai-org/GLM-4.7-Flash \
    --num_gpus 1 \
    --batch_size 4
```

| Flag | Default | Notes |
|------|---------|-------|
| `--input` | *(required)* | Intermediate parquet from Stage 1 |
| `--output` | *(required)* | Final parquet with problem statements |
| `--model_name` | script default | HF name or local path to the generator model |
| `--num_gpus` | script default | `tensor_parallel_size` |
| `--batch_size` | script default | vLLM generation batch size |

> Requires `pip install vllm` and a GPU. The model can be any vLLM-servable
> instruction model; the defaults are set at the top of the script.

---

## Dataset schema

After loading, every instance is normalized to these common columns (missing
ones are filled with empty values):

| Column | Used for |
|--------|----------|
| `source` | Provenance label (which benchmark the row came from) |
| `repo` | `org/name` — used to clone the repository |
| `instance_id` | Unique task id |
| `base_commit` | Commit checked out before applying the patch |
| `patch` | **Gold patch** — the source of truth for modified functions and the PR0 stub |
| `test_patch` | Tests that validate a solution |
| `problem_statement` | Original benchmark problem statement (if any) |
| `FAIL_TO_PASS` / `PASS_TO_PASS` | Test outcome sets used for grading |

`patch`, `repo`, and `base_commit` are the load-bearing fields: Stage 1 cannot
build a stub without a parseable gold patch.

---

## Extension to other benchmarks

CodeThread is benchmark-agnostic. To run the pipeline over a new benchmark,
pick the case that matches your data source.

### 1. A HuggingFace dataset already in SWE-bench format

If your dataset is on the Hub and exposes `instance_id`, `repo`, `base_commit`,
`patch`, `test_patch` (and ideally `FAIL_TO_PASS` / `PASS_TO_PASS`), no code
change is needed — just pass the Hub name directly:

```bash
python data_gen_patches.py --datasets your-org/Your-Bench --split test --output ./data/patches.parquet
```

Unknown keys are treated as custom HF dataset names automatically. For a
permanent, named entry, add it to `KNOWN_DATASETS` in `data_gen_patches.py`:

```python
KNOWN_DATASETS = {
    ...
    "yourbench": {
        "hf_name": "your-org/Your-Bench",
        "split": "test",
        "source_label": "yourbench",
    },
}
```

If column names differ (e.g. lowercase `fail_to_pass`), extend the normalization
in `load_and_normalize_dataset` so they map onto the common schema.

### 2. A non-HuggingFace source (JSON / URL / local file)

Follow the **FeatBench** example, which is loaded from a raw JSON URL. Add an
entry with a `json_url` (instead of `hf_name`) and let `load_json_dataset`
handle it:

```python
"yourbench": {
    "json_url": "https://.../your_bench.json",
    "source_label": "yourbench",
},
```

If your patches are stored as a list of per-file dicts rather than a single
unified diff, reuse / adapt `_patch_files_to_unified_diff` so the `patch` column
ends up as a standard unified diff (this is exactly what FeatBench needs).

For a fully custom loader, write a function that returns a `pandas.DataFrame`
with the [common schema](#dataset-schema) columns and set `df["source"]`, then
branch to it in `main()` alongside the existing `json_url` / `hf_name` cases.

### 3. A new programming language

Language handling in Stage 1 is fully table-driven at the top of
`data_gen_patches.py`. To support a new language, add an entry to each map:

- `EXTENSION_TO_LANGUAGE` — file extension → language name
- `LANGUAGE_STUB_COMMENT` — the stub body that replaces a function (your
  "Write Your Code Here" placeholder)
- `FUNCTION_DEF_PATTERNS` — regex(es) to recognize function/method definitions
- `FUNCTION_NODE_TYPES` + `TREESITTER_MODULES` — for tree-sitter parsing (install
  the matching `tree_sitter_<lang>` package)
- add the language to `BRACE_LANGUAGES` or `INDENT_LANGUAGES` so the body-extraction
  fallback knows how to find function boundaries


Run Stage 1 with `--num_instances 5` first to confirm `PR0_Patch` and
`original_functions` come out non-empty before scaling up.
