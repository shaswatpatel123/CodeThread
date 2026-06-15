"""
data_gen.py - SWE-bench Multilingual Data Generation Pipeline

Processes SWE-bench Multilingual dataset to:
1. Extract modified functions/files from gold patches
2. Generate PR0_Patch (stubs replacing function bodies)
3. Generate problem statements via vLLM
4. Output enriched CSV
"""

import os
import re
import json
import logging
import difflib
import subprocess
import shutil
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from datasets import load_dataset
from unidiff import PatchSet
from rich.console import Console
from rich.progress import Progress

# ============================================================================
# GLOBAL CONFIG - Fill in your model and GPU settings
# ============================================================================

VLLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Change to your model
VLLM_NUM_GPUS = 1  # tensor_parallel_size
VLLM_BATCH_SIZE = 32  # prompts per batch
VLLM_MAX_TOKENS = 4096
VLLM_TEMPERATURE = 0.0

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# LANGUAGE DETECTION AND STUB MAPPING
# ============================================================================

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
}

# Stub body per language - replaces function body
LANGUAGE_STUB_COMMENT = {
    "python": '    # Write Your Code Here\n    pass',
    "java": '    // Write Your Code Here\n    throw new UnsupportedOperationException("Not implemented");',
    "javascript": '    // Write Your Code Here\n    throw new Error("Not implemented");',
    "typescript": '    // Write Your Code Here\n    throw new Error("Not implemented");',
    "go": '    // Write Your Code Here\n    panic("not implemented")',
    "rust": '    // Write Your Code Here\n    todo!()',
    "c": '    /* Write Your Code Here */\n    return;',
    "cpp": '    /* Write Your Code Here */\n    return;',
    "ruby": '    # Write Your Code Here\n    raise NotImplementedError',
    "kotlin": '    // Write Your Code Here\n    TODO("Not implemented")',
    "scala": '    // Write Your Code Here\n    ???',
    "swift": '    // Write Your Code Here\n    fatalError("Not implemented")',
}

# Regex patterns to extract function/method names from diff context or lines
FUNCTION_DEF_PATTERNS = {
    "python": [
        r'(?:async\s+)?def\s+(\w+)\s*\(',
        r'class\s+(\w+)\s*[\(:]',
    ],
    "java": [
        r'(?:public|private|protected|static|final|abstract|synchronized|native|\s)*\s*(?:<[^>]*>\s+)?(?:\w+(?:\[\])*)\s+(\w+)\s*\(',
        r'(?:public|private|protected|static|\s)*\s*class\s+(\w+)',
    ],
    "javascript": [
        r'(?:async\s+)?function\s+(\w+)\s*\(',
        r'(\w+)\s*(?:=|:)\s*(?:async\s+)?function',
        r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{',
        r'class\s+(\w+)',
    ],
    "typescript": [
        r'(?:async\s+)?function\s+(\w+)\s*[\(<]',
        r'(\w+)\s*(?:=|:)\s*(?:async\s+)?function',
        r'(?:async\s+)?(\w+)\s*[\(<][^)]*\)\s*(?::\s*\w+)?\s*\{',
        r'class\s+(\w+)',
    ],
    "go": [
        r'func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(',
    ],
    "rust": [
        r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)',
        r'impl(?:<[^>]*>)?\s+(\w+)',
    ],
    "c": [
        r'(?:\w+[\s*]+)+(\w+)\s*\([^)]*\)\s*\{',
    ],
    "cpp": [
        r'(?:\w+[\s*:]+)*(\w+)\s*\([^)]*\)\s*(?:const)?\s*(?:override)?\s*\{',
        r'class\s+(\w+)',
    ],
    "ruby": [
        r'def\s+(\w+)',
        r'class\s+(\w+)',
    ],
    "kotlin": [
        r'fun\s+(\w+)\s*[\(<]',
        r'class\s+(\w+)',
    ],
    "scala": [
        r'def\s+(\w+)',
        r'class\s+(\w+)',
    ],
    "swift": [
        r'func\s+(\w+)',
        r'class\s+(\w+)',
    ],
}

# Tree-sitter node types per language for function definitions
FUNCTION_NODE_TYPES = {
    "python": ["function_definition"],
    "java": ["method_declaration", "constructor_declaration"],
    "javascript": ["function_declaration", "method_definition", "function"],
    "typescript": ["function_declaration", "method_definition", "function"],
    "go": ["function_declaration", "method_declaration"],
    "rust": ["function_item"],
    "c": ["function_definition"],
    "cpp": ["function_definition"],
}

# Tree-sitter language module mapping
TREESITTER_MODULES = {
    "python": "tree_sitter_python",
    "java": "tree_sitter_java",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
}

# Brace-delimited languages (for fallback parsing)
BRACE_LANGUAGES = {"java", "javascript", "typescript", "go", "rust", "c", "cpp", "kotlin", "scala", "swift"}
INDENT_LANGUAGES = {"python", "ruby"}


def detect_language(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    return EXTENSION_TO_LANGUAGE.get(ext.lower(), "unknown")


# ============================================================================
# REPOSITORY CLONING AND CHECKOUT
# ============================================================================

def clone_repo(repo_name: str, clone_dir: str) -> Optional[str]:
    """Clone a GitHub repository if not already cloned.

    Args:
        repo_name: GitHub repo in 'owner/name' format
        clone_dir: Base directory to clone into

    Returns:
        Path to the cloned repo directory, or None on failure
    """
    # Use repo name (without owner) as folder name to match function_modifier.py pattern
    repo_folder = repo_name.replace("/", "_")
    repo_path = os.path.join(clone_dir, repo_folder)

    if os.path.exists(repo_path):
        logger.info(f"Repo already exists: {repo_path}")
        return repo_path

    clone_url = f"https://github.com/{repo_name}.git"
    try:
        result = subprocess.run(
            ["git", "clone", clone_url, repo_path],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.error(f"Failed to clone {repo_name}: {result.stderr}")
            return None
        logger.info(f"Cloned {repo_name} to {repo_path}")
        return repo_path
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout cloning {repo_name}")
        return None
    except Exception as e:
        logger.error(f"Error cloning {repo_name}: {e}")
        return None


def checkout_commit(repo_path: str, commit_hash: str) -> bool:
    """Reset and checkout a specific commit in a repo.

    Args:
        repo_path: Path to the git repository
        commit_hash: The commit SHA to checkout

    Returns:
        True on success, False on failure
    """
    try:
        subprocess.run(
            ["git", "reset", "--hard"],
            cwd=repo_path, capture_output=True, text=True, check=True,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=repo_path, capture_output=True, text=True, check=True,
        )
        subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=repo_path, capture_output=True, text=True, check=True,
        )
        return True
    except Exception as e:
        logger.error(f"Error checking out {commit_hash} in {repo_path}: {e}")
        return False


def clone_all_repos(df: pd.DataFrame, clone_dir: str, max_workers: int = 4) -> dict[str, str]:
    """Clone all unique repositories from the dataset in parallel.

    Args:
        df: DataFrame with 'repo' column
        clone_dir: Base directory to clone repos into
        max_workers: Number of parallel clone workers

    Returns:
        Dict of repo_name -> repo_local_path
    """
    os.makedirs(clone_dir, exist_ok=True)
    unique_repos = df["repo"].unique().tolist()
    console.print(f"[bold]Cloning {len(unique_repos)} unique repositories to {clone_dir}...[/bold]")

    repo_paths = {}

    with Progress() as progress:
        task = progress.add_task("[cyan]Cloning repositories...", total=len(unique_repos))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_repo = {
                executor.submit(clone_repo, repo_name, clone_dir): repo_name
                for repo_name in unique_repos
            }

            for future in as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    path = future.result()
                    if path:
                        repo_paths[repo_name] = path
                    else:
                        logger.error(f"Failed to clone {repo_name}")
                except Exception as e:
                    logger.error(f"Exception cloning {repo_name}: {e}")
                progress.update(task, advance=1)

    console.print(f"Successfully cloned {len(repo_paths)}/{len(unique_repos)} repositories")
    return repo_paths


def read_file_at_commit(repo_path: str, file_path: str) -> Optional[str]:
    """Read a file from the currently checked-out commit.

    Args:
        repo_path: Path to the git repository
        file_path: Relative path to the file within the repo

    Returns:
        File contents as string, or None if file not found
    """
    full_path = os.path.join(repo_path, file_path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"File not found: {full_path}")
        return None
    except Exception as e:
        logger.warning(f"Error reading {full_path}: {e}")
        return None


# ============================================================================
# TREE-SITTER FUNCTION DETECTION
# ============================================================================

_parser_cache = {}


def get_treesitter_parser(language: str):
    """Get a tree-sitter parser for the given language. Returns (parser, ts_language) or (None, None)."""
    if language in _parser_cache:
        return _parser_cache[language]

    try:
        from tree_sitter import Language, Parser
    except ImportError:
        logger.warning("tree-sitter not installed. Install with: pip install tree-sitter")
        _parser_cache[language] = (None, None)
        return None, None

    module_name = TREESITTER_MODULES.get(language)
    if not module_name:
        _parser_cache[language] = (None, None)
        return None, None

    try:
        mod = __import__(module_name)
        ts_language = Language(mod.language())
        parser = Parser(ts_language)
        _parser_cache[language] = (parser, ts_language)
        return parser, ts_language
    except Exception as e:
        logger.warning(f"Could not load tree-sitter grammar for {language}: {e}")
        _parser_cache[language] = (None, None)
        return None, None


@dataclass
class FunctionInfo:
    name: str
    start_line: int      # 0-indexed
    end_line: int        # 0-indexed, inclusive
    signature_line: str  # the def/function line text
    body_start_line: int
    body_end_line: int


def _extract_function_name(node, source_bytes: bytes, language: str) -> Optional[str]:
    """Extract function name from a tree-sitter node."""
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    # For some languages, the name is nested deeper
    for child in node.children:
        if child.type == "declarator":
            return _extract_function_name(child, source_bytes, language)
    return None


def _find_body_node(node, language: str):
    """Find the body/block node of a function definition."""
    body_types = {
        "python": ("block",),
        "java": ("block", "constructor_body"),
        "javascript": ("statement_block",),
        "typescript": ("statement_block",),
        "go": ("block",),
        "rust": ("block",),
        "c": ("compound_statement",),
        "cpp": ("compound_statement",),
    }
    targets = body_types.get(language, ("block", "statement_block", "compound_statement"))
    for child in node.children:
        if child.type in targets:
            return child
    return None


def find_functions_in_source(source_code: str, language: str) -> list[FunctionInfo]:
    """Use tree-sitter to find all function definitions in source code."""
    parser, ts_language = get_treesitter_parser(language)
    if parser is None:
        return []

    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)

    node_types = FUNCTION_NODE_TYPES.get(language, [])
    functions = []

    def walk(node):
        if node.type in node_types:
            name = _extract_function_name(node, source_bytes, language)
            if name:
                body = _find_body_node(node, language)
                body_start = body.start_point[0] if body else node.start_point[0] + 1
                body_end = body.end_point[0] if body else node.end_point[0]
                lines = source_code.split("\n")
                sig_line = lines[node.start_point[0]] if node.start_point[0] < len(lines) else ""
                functions.append(FunctionInfo(
                    name=name,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    signature_line=sig_line,
                    body_start_line=body_start,
                    body_end_line=body_end,
                ))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return functions


def find_function_by_name(source_code: str, language: str, func_name: str) -> Optional[FunctionInfo]:
    """Find a specific function by name using tree-sitter."""
    functions = find_functions_in_source(source_code, language)
    for f in functions:
        if f.name == func_name:
            return f
    return None


# ============================================================================
# PATCH PARSING
# ============================================================================

def analyze_patch_multilingual(patch_text: str) -> dict:
    """Analyze a multilingual patch to extract modified files, functions, and metadata.

    Returns:
        {
            'edited_files': list of file paths,
            'file_to_functions': dict of file_path -> list of function names,
            'language_map': dict of file_path -> language string,
            'files_added': int,
            'files_deleted': int,
            'lines_added': int,
            'lines_deleted': int,
        }
    """
    if not patch_text or not patch_text.strip():
        return {
            "edited_files": [],
            "file_to_functions": {},
            "language_map": {},
            "lines_added": 0,
            "lines_deleted": 0,
        }

    # Normalize escaped newlines
    if "\\n" in patch_text and "\n" not in patch_text:
        patch_text = patch_text.replace("\\n", "\n")

    edited_files = []
    file_to_functions = {}
    language_map = {}
    lines_added = 0
    lines_deleted = 0

    try:
        patchset = PatchSet(StringIO(patch_text))
    except Exception as e:
        logger.warning(f"Failed to parse patch with unidiff: {e}. Falling back to manual parsing.")
        return _analyze_patch_manual(patch_text)

    for patched_file in patchset:
        file_path = patched_file.path
        # unidiff sometimes gives 'b/path' or just 'path'
        if file_path.startswith("b/"):
            file_path = file_path[2:]

        lang = detect_language(file_path)
        edited_files.append(file_path)
        language_map[file_path] = lang
        file_functions = set()

        current_function = None

        for hunk in patched_file:
            lines_added += hunk.added
            lines_deleted += hunk.removed

            # Extract function from hunk header section_header
            section_header = hunk.section_header or ""
            if section_header.strip():
                func_name = _extract_function_from_context(section_header, lang)
                if func_name:
                    current_function = func_name

            for line in hunk:
                line_str = str(line.value).rstrip("\n")

                # Check if this line defines a new function
                func_in_line = _extract_function_from_line(line_str, lang)
                if func_in_line:
                    current_function = func_in_line

                # If this is a changed line (added or removed), record the current function
                if line.is_added or line.is_removed:
                    stripped = line_str.strip()
                    if stripped and current_function:
                        file_functions.add(current_function)

        file_to_functions[file_path] = list(file_functions)

    return {
        "edited_files": edited_files,
        "file_to_functions": file_to_functions,
        "language_map": language_map,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
    }


def _extract_function_from_context(context: str, language: str) -> Optional[str]:
    """Extract function name from @@ hunk header context string."""
    patterns = FUNCTION_DEF_PATTERNS.get(language, [])
    for pattern in patterns:
        match = re.search(pattern, context)
        if match:
            # Return the first non-None group
            for g in match.groups():
                if g:
                    return g
    return None


def _extract_function_from_line(line: str, language: str) -> Optional[str]:
    """Extract function name from a diff line (may have +/- prefix)."""
    # Strip diff prefix
    clean = line.lstrip("+-").lstrip()
    patterns = FUNCTION_DEF_PATTERNS.get(language, [])
    for pattern in patterns:
        match = re.match(pattern, clean)
        if match:
            for g in match.groups():
                if g:
                    return g
    return None


def _analyze_patch_manual(patch_text: str) -> dict:
    """Fallback manual patch parser when unidiff fails."""
    edited_files = []
    file_to_functions = {}
    language_map = {}
    lines_added = 0
    lines_deleted = 0

    current_file = None
    current_function = None
    current_lang = "unknown"

    for line in patch_text.split("\n"):
        if line.startswith("diff --git"):
            # Extract file path from diff header
            match = re.search(r"b/(.+)$", line)
            if match:
                current_file = match.group(1)
                current_lang = detect_language(current_file)
                if current_file not in edited_files:
                    edited_files.append(current_file)
                    language_map[current_file] = current_lang
                    file_to_functions[current_file] = []
                current_function = None

        elif line.startswith("@@"):
            match = re.search(r"@@.*?@@\s*(.+)", line)
            if match:
                context = match.group(1).strip()
                func = _extract_function_from_context(context, current_lang)
                if func:
                    current_function = func

        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
            func = _extract_function_from_line(line, current_lang)
            if func:
                current_function = func
            if current_function and current_file:
                if current_function not in file_to_functions.get(current_file, []):
                    file_to_functions.setdefault(current_file, []).append(current_function)

        elif line.startswith("-") and not line.startswith("---"):
            lines_deleted += 1
            func = _extract_function_from_line(line, current_lang)
            if func:
                current_function = func
            if current_function and current_file:
                if current_function not in file_to_functions.get(current_file, []):
                    file_to_functions.setdefault(current_file, []).append(current_function)

    return {
        "edited_files": edited_files,
        "file_to_functions": file_to_functions,
        "language_map": language_map,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
    }


# ============================================================================
# PR0 PATCH GENERATION
# ============================================================================

def reconstruct_before_after(patch_text: str) -> dict[str, tuple[list[str], list[str], list[tuple[int, int]]]]:
    """Reconstruct 'before' and 'after' file content from a unified diff.

    Returns:
        dict of file_path -> (before_lines, after_lines, hunk_ranges)
        hunk_ranges: list of (start_line_in_before, end_line_in_before) for each hunk
    """
    if not patch_text or not patch_text.strip():
        return {}

    if "\\n" in patch_text and "\n" not in patch_text:
        patch_text = patch_text.replace("\\n", "\n")

    result = {}

    try:
        patchset = PatchSet(StringIO(patch_text))
    except Exception as e:
        logger.warning(f"Failed to parse patch for reconstruction: {e}")
        return {}

    for patched_file in patchset:
        file_path = patched_file.path
        if file_path.startswith("b/"):
            file_path = file_path[2:]

        before_lines = []
        after_lines = []
        hunk_ranges = []

        for hunk in patched_file:
            hunk_before_start = len(before_lines)
            for line in hunk:
                if line.is_context:
                    before_lines.append(str(line.value).rstrip("\n"))
                    after_lines.append(str(line.value).rstrip("\n"))
                elif line.is_removed:
                    before_lines.append(str(line.value).rstrip("\n"))
                elif line.is_added:
                    after_lines.append(str(line.value).rstrip("\n"))
            hunk_before_end = len(before_lines)
            hunk_ranges.append((hunk_before_start, hunk_before_end))

        result[file_path] = (before_lines, after_lines, hunk_ranges)

    return result


def _get_stub_body(language: str, indent: str = "    ") -> str:
    """Get stub body for a language with the given indentation."""
    base_stub = LANGUAGE_STUB_COMMENT.get(language, "    // Write Your Code Here")
    # Re-indent
    lines = base_stub.split("\n")
    return "\n".join(indent + line.strip() for line in lines)


def _find_function_boundary_brace(lines: list[str], func_start_idx: int) -> Optional[int]:
    """Find end of a brace-delimited function using brace counting.

    Args:
        lines: source lines
        func_start_idx: index of the line with the function definition

    Returns:
        Index of the closing brace line, or None if not found
    """
    brace_count = 0
    started = False
    for i in range(func_start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                brace_count += 1
                started = True
            elif ch == "}":
                brace_count -= 1
        if started and brace_count == 0:
            return i
    return None


def _find_function_boundary_indent(lines: list[str], func_start_idx: int) -> Optional[int]:
    """Find end of an indent-delimited function (Python/Ruby).

    Returns index of last line of the function body.
    """
    if func_start_idx >= len(lines):
        return None

    # Get indentation of the def line
    def_line = lines[func_start_idx]
    def_indent = len(def_line) - len(def_line.lstrip())

    # The body starts after the def line (possibly multi-line signature)
    body_start = func_start_idx + 1

    # Handle multi-line function signatures
    # Check if the def line has a colon at the end
    combined = def_line.rstrip()
    i = func_start_idx
    while not combined.endswith(":") and i + 1 < len(lines):
        i += 1
        combined = lines[i].rstrip()
        body_start = i + 1

    if body_start >= len(lines):
        return func_start_idx

    last_body_line = body_start
    for i in range(body_start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue  # skip blank lines
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent <= def_indent:
            break
        last_body_line = i

    return last_body_line


def generate_stub_replacement(
    before_lines: list[str],
    func_name: str,
    language: str,
    func_start_idx: int,
) -> Optional[tuple[int, int, list[str]]]:
    """Generate stub replacement for a function in the before content.

    First tries tree-sitter, then falls back to brace/indent counting.

    Returns:
        (body_start_line_idx, body_end_line_idx, stub_lines) or None
    """
    source = "\n".join(before_lines)

    # Try tree-sitter first
    func_info = find_function_by_name(source, language, func_name)
    if func_info:
        body_start = func_info.body_start_line
        body_end = func_info.body_end_line

        # Determine indentation from body
        if body_start < len(before_lines):
            sample = before_lines[body_start]
            indent = " " * (len(sample) - len(sample.lstrip())) if sample.strip() else "    "
        else:
            indent = "    "

        if language in BRACE_LANGUAGES:
            # For brace languages, keep the opening { on signature line and closing }
            stub_body = _get_stub_body(language, indent)
            # The body node includes braces, so we replace the content between them
            stub_lines = [indent[:-4] + "{" if indent else "{"]
            stub_lines.extend(stub_body.split("\n"))
            stub_lines.append(indent[:-4] + "}" if indent else "}")
            return body_start, body_end, stub_lines
        else:
            # Python/Ruby: replace body content
            stub_body = _get_stub_body(language, indent)
            stub_lines = stub_body.split("\n")
            return body_start, body_end, stub_lines

    # Fallback: heuristic boundary detection
    if language in BRACE_LANGUAGES:
        end_idx = _find_function_boundary_brace(before_lines, func_start_idx)
        if end_idx is not None:
            # Find the opening brace
            brace_line = None
            for i in range(func_start_idx, end_idx + 1):
                if "{" in before_lines[i]:
                    brace_line = i
                    break
            if brace_line is not None:
                # Determine indent from first line after brace
                if brace_line + 1 < len(before_lines):
                    sample = before_lines[brace_line + 1]
                    indent = " " * (len(sample) - len(sample.lstrip())) if sample.strip() else "    "
                else:
                    indent = "    "
                stub_body = _get_stub_body(language, indent)
                # Keep the signature up to and including {, then stub, then }
                sig_end = brace_line
                stub_lines = [before_lines[sig_end]]  # line with {
                stub_lines.extend(stub_body.split("\n"))
                closing_indent = before_lines[end_idx].rstrip().replace("}", "")
                stub_lines.append(closing_indent + "}")
                return sig_end, end_idx, stub_lines

    elif language in INDENT_LANGUAGES:
        end_idx = _find_function_boundary_indent(before_lines, func_start_idx)
        if end_idx is not None:
            # Determine indent from body
            body_start = func_start_idx + 1
            if body_start < len(before_lines):
                sample = before_lines[body_start]
                indent = " " * (len(sample) - len(sample.lstrip())) if sample.strip() else "    "
            else:
                indent = "    "
            stub_body = _get_stub_body(language, indent)
            stub_lines = stub_body.split("\n")
            return body_start, end_idx, stub_lines

    return None


def generate_pr0_patch(gold_patch: str, repo_path: str = None) -> tuple[str, dict]:
    """Generate PR0 patch that stubs out all modified functions.

    Uses the actual repo files (if repo_path provided) for accurate function
    boundary detection. Falls back to reconstructing from the patch if repo
    is not available.

    Args:
        gold_patch: The gold solution patch from the dataset
        repo_path: Path to the cloned repo (checked out at base_commit).
                   If None, falls back to patch-based reconstruction.

    Returns:
        (pr0_patch_text, file_to_function_mapping)
    """
    analysis = analyze_patch_multilingual(gold_patch)
    file_to_functions = analysis["file_to_functions"]
    language_map = analysis["language_map"]

    if not file_to_functions:
        return "", {}

    # If we have repo access, read actual files; otherwise reconstruct from patch
    reconstructed = reconstruct_before_after(gold_patch) if repo_path is None else None

    pr0_patch_parts = []
    actual_mapping = {}

    for file_path, func_names in file_to_functions.items():
        if not func_names:
            continue

        language = language_map.get(file_path, "unknown")

        # Get the "before" content (original file at base_commit)
        if repo_path:
            file_content = read_file_at_commit(repo_path, file_path)
            if file_content is None:
                logger.warning(f"Could not read {file_path} from repo, falling back to patch reconstruction")
                if reconstructed is None:
                    reconstructed = reconstruct_before_after(gold_patch)
                if file_path in reconstructed:
                    before_lines = reconstructed[file_path][0]
                else:
                    logger.warning(f"Could not get content for {file_path}")
                    continue
            else:
                before_lines = file_content.split("\n")
        else:
            if file_path not in reconstructed:
                logger.warning(f"Could not reconstruct content for {file_path}")
                continue
            before_lines = reconstructed[file_path][0]

        # Work on a copy for modifications
        modified_lines = list(before_lines)
        functions_stubbed = []

        # Sort functions by their position in the source (process from bottom to top
        # to avoid line number shifts)
        func_positions = []
        for func_name in func_names:
            pos = _find_func_start_in_lines(before_lines, func_name, language)
            if pos is not None:
                func_positions.append((pos, func_name))

        # Sort by position descending (process from bottom to top)
        func_positions.sort(key=lambda x: x[0], reverse=True)

        for func_start, func_name in func_positions:
            replacement = generate_stub_replacement(
                modified_lines, func_name, language, func_start
            )
            if replacement:
                body_start, body_end, stub_lines = replacement
                modified_lines[body_start:body_end + 1] = stub_lines
                functions_stubbed.append(func_name)
                logger.info(f"Stubbed function {func_name} in {file_path}")
            else:
                logger.warning(f"Could not generate stub for {func_name} in {file_path}")

        if functions_stubbed:
            actual_mapping[file_path] = functions_stubbed
            # Generate unified diff
            diff = difflib.unified_diff(
                before_lines,
                modified_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm="",
            )
            pr0_patch_parts.append("\n".join(diff))

    pr0_patch = "\n".join(pr0_patch_parts)
    return pr0_patch, actual_mapping


def _find_func_start_in_lines(lines: list[str], func_name: str, language: str) -> Optional[int]:
    """Find the line index where a function definition starts."""
    # Try tree-sitter first
    source = "\n".join(lines)
    func_info = find_function_by_name(source, language, func_name)
    if func_info:
        return func_info.start_line

    # Fallback: regex search
    patterns = FUNCTION_DEF_PATTERNS.get(language, [])
    for i, line in enumerate(lines):
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                for g in match.groups():
                    if g == func_name:
                        return i
    return None


# ============================================================================
# VLLM PROBLEM STATEMENT GENERATION
# ============================================================================

PROBLEM_STATEMENT_PROMPT = """You are creating a GitHub issue section for a Lead Software Engineer hiring test. This describes functions to implement.

**Repository**: {repo}
**Language(s)**: {languages}
**Files Modified**: {files_modified}

**Gold Patch (the correct implementation)**:
```diff
{patch}
```

**Functions to Implement**:
{functions_list}

Create a detailed problem statement using this EXACT format:

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
        max_model_len=8192,
    )
    return llm


def prepare_prompts(
    instances: list[dict],
    patch_analyses: dict[str, dict],
) -> list[dict]:
    """Prepare all prompts for problem statement generation.

    Args:
        instances: List of dataset instance dicts
        patch_analyses: Dict of instance_id -> analyze result with file_to_functions etc.

    Returns:
        List of dicts with 'instance_id' and 'messages' keys
    """
    prompts = []
    for instance in instances:
        iid = instance["instance_id"]
        analysis = patch_analyses.get(iid)
        if not analysis or not analysis.get("file_to_functions"):
            continue

        repo = instance.get("repo", "unknown")
        patch = instance.get("patch", "")
        file_to_functions = analysis["file_to_functions"]
        language_map = analysis.get("language_map", {})

        # Build functions list text
        functions_text = ""
        languages = set()
        files_modified = []
        for file_path, funcs in file_to_functions.items():
            if not funcs:
                continue
            lang = language_map.get(file_path, "unknown")
            languages.add(lang)
            files_modified.append(file_path)
            for func in funcs:
                functions_text += f"- `{func}` in `{file_path}` ({lang})\n"

        if not functions_text:
            continue

        prompt_text = PROBLEM_STATEMENT_PROMPT.format(
            repo=repo,
            languages=", ".join(sorted(languages)),
            files_modified=", ".join(files_modified),
            patch=patch[:6000],  # Truncate very long patches
            functions_list=functions_text,
        )

        prompts.append({
            "instance_id": iid,
            "messages": [{"role": "user", "content": prompt_text}],
        })

    return prompts


def generate_problem_statements_batched(
    llm,
    prompts: list[dict],
    batch_size: int = None,
) -> dict[str, str]:
    """Generate problem statements using vLLM with optimal batching.

    Uses chat template formatting via llm.chat().
    Processes prompts in batches for memory efficiency.

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
# INSTANCE PROCESSING
# ============================================================================

def process_instance(instance: dict, repo_paths: dict[str, str] = None) -> dict:
    """Process a single instance: parse patch, checkout repo, generate PR0 patch.

    Args:
        instance: Dataset instance dict with 'patch', 'repo', 'base_commit' etc.
        repo_paths: Dict of repo_name -> local repo path (from clone_all_repos)

    Returns dict with:
        - functions_modified: list of function names
        - file_to_function_mapping: dict
        - PR0_Patch: str (unified diff)
        - analysis: full patch analysis
    """
    patch = instance.get("patch", "")

    if not patch or not patch.strip():
        return {
            "functions_modified": [],
            "file_to_function_mapping": {},
            "PR0_Patch": "",
            "analysis": {},
        }

    # Analyze the gold patch
    analysis = analyze_patch_multilingual(patch)

    # Checkout to base_commit and read actual files if repo is available
    repo_path = None
    if repo_paths:
        repo_name = instance.get("repo", "")
        base_commit = instance.get("base_commit", "")
        repo_path = repo_paths.get(repo_name)

        if repo_path and base_commit:
            if not checkout_commit(repo_path, base_commit):
                logger.warning(f"Could not checkout {base_commit} for {repo_name}, falling back to patch reconstruction")
                repo_path = None

    # Generate PR0 patch (uses actual files if repo_path available, else patch reconstruction)
    pr0_patch, actual_mapping = generate_pr0_patch(patch, repo_path=repo_path)

    # Collect all function names
    all_functions = []
    for funcs in analysis["file_to_functions"].values():
        all_functions.extend(funcs)

    return {
        "functions_modified": all_functions,
        "file_to_function_mapping": actual_mapping if actual_mapping else analysis["file_to_functions"],
        "PR0_Patch": pr0_patch,
        "analysis": analysis,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate data from SWE-bench Multilingual")
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the output CSV file",
    )
    parser.add_argument(
        "--repo_base_path",
        type=str,
        default="cloned_repos",
        help="Base directory to clone repositories into (default: cloned_repos)",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=None,
        help="Number of instances to process (default: all)",
    )
    parser.add_argument(
        "--clone_workers",
        type=int,
        default=4,
        help="Number of parallel workers for cloning repos (default: 4)",
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
    parser.add_argument(
        "--skip_problem_gen",
        action="store_true",
        help="Skip problem statement generation (useful for testing patch parsing)",
    )
    parser.add_argument(
        "--skip_clone",
        action="store_true",
        help="Skip cloning repos (assumes they already exist at --repo_base_path)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SWE-bench/SWE-bench_Multilingual",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)",
    )

    args = parser.parse_args()

    # Override globals from args
    global VLLM_MODEL_NAME, VLLM_NUM_GPUS, VLLM_BATCH_SIZE
    if args.model_name:
        VLLM_MODEL_NAME = args.model_name
    if args.num_gpus:
        VLLM_NUM_GPUS = args.num_gpus
    if args.batch_size:
        VLLM_BATCH_SIZE = args.batch_size

    # Step 1: Load dataset
    console.print(f"[bold]Loading dataset: {args.dataset_name} (split={args.split})[/bold]")
    dataset = load_dataset(args.dataset_name, split=args.split)
    df = pd.DataFrame(dataset)
    console.print(f"Loaded {len(df)} instances")

    if args.num_instances:
        df = df.head(args.num_instances)
        console.print(f"Processing first {args.num_instances} instances")

    # Step 2: Clone all repositories
    repo_paths = {}
    if not args.skip_clone:
        console.print("[bold]Step 1/4: Cloning repositories...[/bold]")
        repo_paths = clone_all_repos(df, args.repo_base_path, max_workers=args.clone_workers)
    else:
        console.print("[yellow]Skipping clone (--skip_clone). Scanning existing repos...[/yellow]")
        # Build repo_paths from existing directories
        for repo_name in df["repo"].unique():
            repo_folder = repo_name.replace("/", "_")
            repo_path = os.path.join(args.repo_base_path, repo_folder)
            if os.path.exists(repo_path):
                repo_paths[repo_name] = repo_path
            else:
                logger.warning(f"Repo not found at {repo_path}, will use patch reconstruction for {repo_name}")
        console.print(f"Found {len(repo_paths)}/{len(df['repo'].unique())} repos locally")

    # Step 3: Process each instance (patch parsing + checkout + PR0 generation)
    console.print("[bold]Step 2/4: Analyzing patches and generating PR0 patches...[/bold]")
    patch_results = {}

    with Progress() as progress:
        task = progress.add_task("[green]Processing instances...", total=len(df))
        for idx, row in df.iterrows():
            instance = row.to_dict()
            result = process_instance(instance, repo_paths=repo_paths)
            patch_results[instance["instance_id"]] = result
            progress.update(task, advance=1)

    # Step 4: Generate problem statements with vLLM
    problem_statements = {}
    if not args.skip_problem_gen:
        console.print("[bold]Step 3/4: Generating problem statements via vLLM...[/bold]")
        llm = init_vllm_engine()
        all_prompts = prepare_prompts(df.to_dict("records"), {
            iid: r["analysis"] for iid, r in patch_results.items()
        })
        console.print(f"Prepared {len(all_prompts)} prompts for generation")
        problem_statements = generate_problem_statements_batched(llm, all_prompts)
    else:
        console.print("[yellow]Skipping problem statement generation (--skip_problem_gen)[/yellow]")
        for iid in df["instance_id"]:
            problem_statements[iid] = ""

    # Step 5: Assemble output DataFrame
    console.print("[bold]Step 4/4: Assembling output...[/bold]")

    df["functions_modified"] = df["instance_id"].map(
        lambda iid: json.dumps(patch_results[iid]["functions_modified"])
    )
    df["file_to_function_mapping"] = df["instance_id"].map(
        lambda iid: json.dumps(patch_results[iid]["file_to_function_mapping"])
    )
    df["PR0_Patch"] = df["instance_id"].map(
        lambda iid: patch_results[iid]["PR0_Patch"]
    )
    df["PR1_Problem_Statement"] = df["instance_id"].map(
        lambda iid: problem_statements.get(iid, "")
    )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    # Summary stats
    total_functions = sum(len(r["functions_modified"]) for r in patch_results.values())
    total_with_pr0 = sum(1 for r in patch_results.values() if r["PR0_Patch"])
    total_with_ps = sum(1 for v in problem_statements.values() if v)
    languages = set()
    for r in patch_results.values():
        if r["analysis"]:
            languages.update(r["analysis"].get("language_map", {}).values())

    console.print(f"\n[bold green]Processing complete![/bold green]")
    console.print(f"  Instances processed: {len(df)}")
    console.print(f"  Total functions found: {total_functions}")
    console.print(f"  Instances with PR0_Patch: {total_with_pr0}")
    console.print(f"  Instances with PR1_Problem_Statement: {total_with_ps}")
    console.print(f"  Languages detected: {', '.join(sorted(languages))}")
    console.print(f"  Repos cloned to: {os.path.abspath(args.repo_base_path)}")
    console.print(f"  Output saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
