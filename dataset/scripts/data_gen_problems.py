"""
data_gen_problems.py - GPU-based Problem Statement Generation Pipeline

Reads the intermediate parquet produced by data_gen_patches.py and generates
problem statements using vLLM.

Usage:
    python data_gen_problems.py --input output/patches.parquet --output output/final.parquet

Prerequisites:
    - Run data_gen_patches.py first to produce the intermediate parquet
    - Requires GPU and vLLM installed: pip install vllm
"""

import os
import json
import logging

import pandas as pd
from rich.console import Console
from rich.progress import Progress

# ============================================================================
# GLOBAL CONFIG - Fill in your model and GPU settings
# ============================================================================

VLLM_MODEL_NAME = "zai-org/GLM-4.7-Flash"  # Change to your model
VLLM_NUM_GPUS = 1  # tensor_parallel_size
VLLM_BATCH_SIZE = 2  # prompts per batch
VLLM_MAX_TOKENS = 32784
VLLM_TEMPERATURE = 1.0

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

PROBLEM_STATEMENT_PROMPT = """You are creating a GitHub issue / problem statement for a Lead Software Engineer hiring test. The candidate must implement the function bodies shown below (which have been emptied out with stub placeholders).

**Repository**: {repo}
**Language(s)**: {languages}
**Files Modified**: {files_modified}

**Original Function Implementations (the correct code the candidate must reproduce)**:

{function_sources}

**Functions to Implement**:
{functions_list}

Using the original function implementations above, create a detailed problem statement that describes what each function should do so that a developer can re-implement them correctly. Do NOT include the actual code — only describe the behavior, logic, and requirements.

Use this EXACT format:

## Task Overview
[1-2 sentence summary of the overall task]

## Functions to Implement

For each function:

### `function_name` in `file_path`

**Summary**: [1-2 sentence description of the function's purpose]

**Args**:
- `param_name` (type): [Detailed explanation of the parameter]

**Returns**:
- [Exact structure and type]

**Raises**:
- `ExceptionType`: [Conditions when this exception is raised]

**Implementation Steps**:
1. [First step with details]
2. [Second step with details]
3. [Continue...]

**Edge Cases**:
- [Edge case 1 and how to handle it]
- [Edge case 2 and how to handle it]

**CRITICAL**: Include enough detail that the functions can be implemented without seeing the original code. Focus on:
- Exact return value structures
- Key algorithmic steps in order
- Conditional logic and when it applies
- Error handling and validation checks
- How functions interact with each other (if multiple)

Return ONLY the markdown content starting with "## Task Overview"."""


# ============================================================================
# VLLM ENGINE
# ============================================================================

def init_vllm_engine():
    """Initialize the vLLM engine with the configured model."""
    try:
        from vllm import LLM
    except ImportError:
        raise ImportError(
            "vLLM is required for problem statement generation. "
            "Install with: pip install vllm"
        )

    logger.info(f"Initializing vLLM engine with model={VLLM_MODEL_NAME}, GPUs={VLLM_NUM_GPUS}")
    llm = LLM(
        model=VLLM_MODEL_NAME,
        tensor_parallel_size=VLLM_NUM_GPUS,
        trust_remote_code=True,
        max_model_len=VLLM_MAX_TOKENS,
    )
    return llm


# ============================================================================
# PROMPT PREPARATION
# ============================================================================

def prepare_prompts(df: pd.DataFrame) -> list[dict]:
    """Prepare all prompts for problem statement generation from the intermediate CSV.

    Expects the DataFrame to have columns:
        - instance_id, repo, file_to_function_mapping, original_functions

    Returns:
        List of dicts with 'instance_id' and 'messages' keys
    """
    prompts = []
    for _, row in df.iterrows():
        iid = row["instance_id"]

        # Parse file_to_function_mapping from JSON string
        try:
            file_to_functions = json.loads(row.get("file_to_function_mapping", "{}"))
        except (json.JSONDecodeError, TypeError):
            file_to_functions = {}

        if not file_to_functions:
            continue

        # Parse original_functions from JSON string
        try:
            original_functions = json.loads(row.get("original_functions", "{}"))
        except (json.JSONDecodeError, TypeError):
            original_functions = {}

        repo = row.get("repo", "unknown")

        # Build function list and source blocks
        functions_text = ""
        function_sources = ""
        languages = set()
        files_modified = []
        for file_path, funcs in file_to_functions.items():
            if not funcs:
                continue
            ext = os.path.splitext(file_path)[1].lower()
            lang_map = {
                ".py": "python", ".java": "java", ".js": "javascript",
                ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
                ".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp",
                ".cc": "cpp", ".h": "c", ".hpp": "cpp", ".rb": "ruby",
                ".kt": "kotlin", ".scala": "scala", ".swift": "swift",
                ".php": "php",
            }
            lang = lang_map.get(ext, "unknown")
            languages.add(lang)
            files_modified.append(file_path)
            for func in funcs:
                functions_text += f"- `{func}` in `{file_path}` ({lang})\n"
                # Look up original function source
                key = f"{file_path}::{func}"
                source = original_functions.get(key, "")
                if source:
                    function_sources += f"### `{func}` in `{file_path}`\n```{lang}\n{source}\n```\n\n"

        if not functions_text:
            continue

        # If no original sources were captured, fall back to the gold patch
        if not function_sources.strip():
            patch = str(row.get("patch", ""))
            function_sources = f"(Original function sources not available. Gold patch for reference):\n```diff\n{patch[:6000]}\n```"

        prompt_text = PROBLEM_STATEMENT_PROMPT.format(
            repo=repo,
            languages=", ".join(sorted(languages)),
            files_modified=", ".join(files_modified),
            function_sources=function_sources,
            functions_list=functions_text,
        )

        prompts.append({
            "instance_id": iid,
            "messages": [{"role": "user", "content": prompt_text}],
        })

    return prompts


# ============================================================================
# BATCHED GENERATION
# ============================================================================

def generate_problem_statements_batched(
    llm,
    prompts: list[dict],
    batch_size: int = None,
) -> dict[str, str]:
    """Generate problem statements using vLLM with optimal batching.

    Returns:
        Dict of instance_id -> generated problem statement
    """
    from vllm import SamplingParams

    if batch_size is None:
        batch_size = VLLM_BATCH_SIZE

    sampling_params = SamplingParams(
        temperature=VLLM_TEMPERATURE,
        max_tokens=VLLM_MAX_TOKENS,
    )

    results = {}
    total = len(prompts)

    with Progress() as progress:
        task = progress.add_task("[cyan]Generating problem statements...", total=total)

        for i in range(0, total, batch_size):
            batch = prompts[i : i + batch_size]

            conversations = [p["messages"] for p in batch]

            try:
                outputs = llm.chat(
                    messages=conversations,
                    sampling_params=sampling_params,
                )

                for prompt_info, output in zip(batch, outputs):
                    results[prompt_info["instance_id"]] = output.outputs[0].text
            except Exception as e:
                logger.error(f"vLLM batch generation failed for batch {i//batch_size}: {e}")
                for prompt_info in batch:
                    results[prompt_info["instance_id"]] = ""

            progress.update(task, advance=len(batch))

    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # Override globals from args
    global VLLM_MODEL_NAME, VLLM_NUM_GPUS, VLLM_BATCH_SIZE

    import argparse

    parser = argparse.ArgumentParser(
        description="GPU-based problem statement generation for SWE-bench Multilingual"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to intermediate parquet from data_gen_patches.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the final parquet with problem statements",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=f"vLLM model name (default: {VLLM_MODEL_NAME})",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help=f"Number of GPUs for tensor parallelism (default: {VLLM_NUM_GPUS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=f"Batch size for vLLM generation (default: {VLLM_BATCH_SIZE})",
    )

    args = parser.parse_args()

    if args.model_name:
        VLLM_MODEL_NAME = args.model_name
    if args.num_gpus:
        VLLM_NUM_GPUS = args.num_gpus
    if args.batch_size:
        VLLM_BATCH_SIZE = args.batch_size

    # Step 1: Load intermediate parquet
    console.print(f"[bold]Loading intermediate parquet: {args.input}[/bold]")
    if not os.path.exists(args.input):
        console.print(f"[bold red]Error: {args.input} not found. Run data_gen_patches.py first.[/bold red]")
        return

    df = pd.read_parquet(args.input)
    console.print(f"Loaded {len(df)} instances")

    # Step 2: Prepare prompts
    console.print("[bold]Step 1/3: Preparing prompts...[/bold]")
    prompts = prepare_prompts(df)
    console.print(f"Prepared {len(prompts)} prompts for generation")

    if not prompts:
        console.print("[yellow]No prompts to generate. Check that input CSV has file_to_function_mapping data.[/yellow]")
        df["PR1_Problem_Statement"] = ""
        df.to_parquet(args.output, index=False)
        return

    # Step 3: Initialize vLLM and generate
    console.print("[bold]Step 2/3: Generating problem statements via vLLM...[/bold]")
    llm = init_vllm_engine()
    problem_statements = generate_problem_statements_batched(llm, prompts)

    # Step 4: Merge results and save
    console.print("[bold]Step 3/3: Assembling output...[/bold]")
    df["PR1_Problem_Statement"] = df["instance_id"].map(
        lambda iid: problem_statements.get(iid, "")
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_parquet(args.output, index=False)

    # Summary
    total_with_ps = sum(1 for v in problem_statements.values() if v)
    console.print(f"\n[bold green]Generation complete![/bold green]")
    console.print(f"  Instances processed: {len(df)}")
    console.print(f"  Problem statements generated: {total_with_ps}")
    console.print(f"  Output saved to: {args.output}")


if __name__ == "__main__":
    main()
