"""
data_gen_patches.py - CPU-based Patch Processing Pipeline

Processes SWE-bench style datasets to:
1. Clone repositories
2. Extract modified functions/files from gold patches
3. Generate PR0_Patch (stubs replacing function bodies)
4. Output intermediate parquet (no LLM required)

Supports multiple datasets:
- SWE-bench/SWE-bench_Multilingual
- ScaleAI/SWE-bench_Pro
- LiberCoders/FeatureBench
- PGCodeLLM/FeatBench_v1.0

Usage:
    # Single dataset
    python data_gen_patches.py --output output/patches.parquet

    # All supported datasets concatenated
    python data_gen_patches.py --output output/patches.parquet --datasets all

The output parquet can then be fed into data_gen_problems.py for LLM-based
problem statement generation.
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
    ".php": "php",
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
    "php": '    // Write Your Code Here\n    throw new \\RuntimeException("Not implemented");',
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
    "php": [
        r'(?:public|private|protected|static|\s)*\s*function\s+(\w+)\s*\(',
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
BRACE_LANGUAGES = {"java", "javascript", "typescript", "go", "rust", "c", "cpp", "kotlin", "scala", "swift", "php"}
INDENT_LANGUAGES = {"python", "ruby"}


def detect_language(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    return EXTENSION_TO_LANGUAGE.get(ext.lower(), "unknown")


# ============================================================================
# REPOSITORY CLONING AND CHECKOUT
# ============================================================================

def clone_repo(repo_name: str, clone_dir: str) -> Optional[str]:
    """Clone a GitHub repository if not already cloned."""
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
    """Reset and checkout a specific commit in a repo."""
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
    """Clone all unique repositories from the dataset in parallel."""
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
    """Read a file from the currently checked-out commit."""
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
    """Analyze a multilingual patch to extract modified files, functions, and metadata."""
    if not patch_text or not patch_text.strip():
        return {
            "edited_files": [],
            "file_to_functions": {},
            "language_map": {},
            "lines_added": 0,
            "lines_deleted": 0,
        }

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

            section_header = hunk.section_header or ""
            if section_header.strip():
                func_name = _extract_function_from_context(section_header, lang)
                if func_name:
                    current_function = func_name

            for line in hunk:
                line_str = str(line.value).rstrip("\n")

                func_in_line = _extract_function_from_line(line_str, lang)
                if func_in_line:
                    current_function = func_in_line

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
            for g in match.groups():
                if g:
                    return g
    return None


def _extract_function_from_line(line: str, language: str) -> Optional[str]:
    """Extract function name from a diff line (may have +/- prefix)."""
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
    """Reconstruct 'before' and 'after' file content from a unified diff."""
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
    lines = base_stub.split("\n")
    return "\n".join(indent + line.strip() for line in lines)


def _find_function_boundary_brace(lines: list[str], func_start_idx: int) -> Optional[int]:
    """Find end of a brace-delimited function using brace counting."""
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
    """Find end of an indent-delimited function (Python/Ruby)."""
    if func_start_idx >= len(lines):
        return None

    def_line = lines[func_start_idx]
    def_indent = len(def_line) - len(def_line.lstrip())

    body_start = func_start_idx + 1

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
            continue
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
    """Generate stub replacement for a function in the before content."""
    source = "\n".join(before_lines)

    # Try tree-sitter first
    func_info = find_function_by_name(source, language, func_name)
    if func_info:
        body_start = func_info.body_start_line
        body_end = func_info.body_end_line

        if body_start < len(before_lines):
            sample = before_lines[body_start]
            indent = " " * (len(sample) - len(sample.lstrip())) if sample.strip() else "    "
        else:
            indent = "    "

        if language in BRACE_LANGUAGES:
            stub_body = _get_stub_body(language, indent)
            stub_lines = [indent[:-4] + "{" if indent else "{"]
            stub_lines.extend(stub_body.split("\n"))
            stub_lines.append(indent[:-4] + "}" if indent else "}")
            return body_start, body_end, stub_lines
        else:
            stub_body = _get_stub_body(language, indent)
            stub_lines = stub_body.split("\n")
            return body_start, body_end, stub_lines

    # Fallback: heuristic boundary detection
    if language in BRACE_LANGUAGES:
        end_idx = _find_function_boundary_brace(before_lines, func_start_idx)
        if end_idx is not None:
            brace_line = None
            for i in range(func_start_idx, end_idx + 1):
                if "{" in before_lines[i]:
                    brace_line = i
                    break
            if brace_line is not None:
                if brace_line + 1 < len(before_lines):
                    sample = before_lines[brace_line + 1]
                    indent = " " * (len(sample) - len(sample.lstrip())) if sample.strip() else "    "
                else:
                    indent = "    "
                stub_body = _get_stub_body(language, indent)
                sig_end = brace_line
                stub_lines = [before_lines[sig_end]]
                stub_lines.extend(stub_body.split("\n"))
                closing_indent = before_lines[end_idx].rstrip().replace("}", "")
                stub_lines.append(closing_indent + "}")
                return sig_end, end_idx, stub_lines

    elif language in INDENT_LANGUAGES:
        end_idx = _find_function_boundary_indent(before_lines, func_start_idx)
        if end_idx is not None:
            # Skip past multi-line signature: scan forward until the line ending with ':'
            body_start = func_start_idx + 1
            combined = before_lines[func_start_idx].rstrip()
            i = func_start_idx
            while not combined.endswith(":") and i + 1 < len(before_lines):
                i += 1
                combined = before_lines[i].rstrip()
                body_start = i + 1
            if body_start < len(before_lines):
                sample = before_lines[body_start]
                indent = " " * (len(sample) - len(sample.lstrip())) if sample.strip() else "    "
            else:
                indent = "    "
            stub_body = _get_stub_body(language, indent)
            stub_lines = stub_body.split("\n")
            return body_start, end_idx, stub_lines

    return None


def generate_pr0_patch(gold_patch: str, repo_path: str = None) -> tuple[str, dict, dict]:
    """Generate PR0 patch that stubs out all modified functions.

    Returns:
        (pr0_patch, actual_mapping, original_functions)
        original_functions: dict mapping "file_path::func_name" -> full original function source
    """
    analysis = analyze_patch_multilingual(gold_patch)
    file_to_functions = analysis["file_to_functions"]
    language_map = analysis["language_map"]

    if not file_to_functions:
        return "", {}, {}

    reconstructed = reconstruct_before_after(gold_patch) if repo_path is None else None

    pr0_patch_parts = []
    actual_mapping = {}
    original_functions = {}

    for file_path, func_names in file_to_functions.items():
        if not func_names:
            continue

        language = language_map.get(file_path, "unknown")

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

        modified_lines = list(before_lines)
        functions_stubbed = []

        func_positions = []
        for func_name in func_names:
            pos = _find_func_start_in_lines(before_lines, func_name, language)
            if pos is not None:
                func_positions.append((pos, func_name))

        func_positions.sort(key=lambda x: x[0], reverse=True)

        for func_start, func_name in func_positions:
            # Capture original function source BEFORE stubbing
            source = "\n".join(before_lines)
            func_info = find_function_by_name(source, language, func_name)
            if func_info:
                orig_lines = before_lines[func_info.start_line:func_info.end_line + 1]
                original_functions[f"{file_path}::{func_name}"] = "\n".join(orig_lines)
            else:
                # Fallback: use heuristic boundaries
                if language in BRACE_LANGUAGES:
                    end_idx = _find_function_boundary_brace(before_lines, func_start)
                elif language in INDENT_LANGUAGES:
                    end_idx = _find_function_boundary_indent(before_lines, func_start)
                else:
                    end_idx = None
                if end_idx is not None:
                    orig_lines = before_lines[func_start:end_idx + 1]
                    original_functions[f"{file_path}::{func_name}"] = "\n".join(orig_lines)

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
            diff = difflib.unified_diff(
                before_lines,
                modified_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm="",
            )
            pr0_patch_parts.append("\n".join(diff))

    pr0_patch = "\n".join(pr0_patch_parts)
    return pr0_patch, actual_mapping, original_functions


def _find_func_start_in_lines(lines: list[str], func_name: str, language: str) -> Optional[int]:
    """Find the line index where a function definition starts."""
    source = "\n".join(lines)
    func_info = find_function_by_name(source, language, func_name)
    if func_info:
        return func_info.start_line

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
# INSTANCE PROCESSING
# ============================================================================

def process_instance(instance: dict, repo_paths: dict[str, str] = None) -> dict:
    """Process a single instance: parse patch, checkout repo, generate PR0 patch."""
    patch = instance.get("patch", "")

    if not patch or not patch.strip():
        return {
            "functions_modified": [],
            "file_to_function_mapping": {},
            "PR0_Patch": "",
            "original_functions": {},
            "analysis": {},
        }

    analysis = analyze_patch_multilingual(patch)

    repo_path = None
    if repo_paths:
        repo_name = instance.get("repo", "")
        base_commit = instance.get("base_commit", "")
        repo_path = repo_paths.get(repo_name)

        if repo_path and base_commit:
            if not checkout_commit(repo_path, base_commit):
                logger.warning(f"Could not checkout {base_commit} for {repo_name}, falling back to patch reconstruction")
                repo_path = None

    pr0_patch, actual_mapping, original_functions = generate_pr0_patch(patch, repo_path=repo_path)

    all_functions = []
    for funcs in analysis["file_to_functions"].values():
        all_functions.extend(funcs)

    return {
        "functions_modified": all_functions,
        "file_to_function_mapping": actual_mapping if actual_mapping else analysis["file_to_functions"],
        "PR0_Patch": pr0_patch,
        "original_functions": original_functions,
        "analysis": analysis,
    }


# ============================================================================
# DATASET LOADING AND NORMALIZATION
# ============================================================================

KNOWN_DATASETS = {
    "verified": {
        "hf_name": "SWE-bench/SWE-bench_Verified",
        "split": "test",
        "source_label": "swebench_verified",
    },
    "multilingual": {
        "hf_name": "SWE-bench/SWE-bench_Multilingual",
        "split": "test",
        "source_label": "swebench_multilingual",
    },
    "pro": {
        "hf_name": "ScaleAI/SWE-bench_Pro",
        "split": "test",
        "source_label": "swebench_pro",
    },
    "featbench": {
        "json_url": "https://raw.githubusercontent.com/TsinghuaISE/FeatBench/main/dataset/featbench_v1_0.json",
        "source_label": "featbench",
    },
}

# Common columns to keep in the final concatenated output
COMMON_COLUMNS = [
    "source", "repo", "instance_id", "base_commit", "patch", "test_patch",
    "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS",
]


def _patch_files_to_unified_diff(patch_files: list[dict]) -> str:
    """Convert FeatBench-style patch_files (list of per-file dicts) to a unified diff string."""
    parts = []
    for pf in patch_files:
        filename = pf.get("filename", "")
        patch_content = pf.get("patch", "")
        if not filename or not patch_content:
            continue
        header = f"diff --git a/{filename} b/{filename}\n--- a/{filename}\n+++ b/{filename}\n"
        parts.append(header + patch_content)
    return "\n".join(parts)


def load_json_dataset(json_url: str, source_label: str) -> pd.DataFrame:
    """Load a dataset from a JSON URL (e.g. FeatBench from GitHub)."""
    import urllib.request

    console.print(f"[bold]Loading {source_label} from {json_url}...[/bold]")
    with urllib.request.urlopen(json_url) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    df = pd.DataFrame(data)

    # FeatBench stores patch/test_patch as list of per-file dicts; convert to unified diff strings
    for col in ["patch", "test_patch"]:
        if col in df.columns and len(df) > 0 and isinstance(df[col].iloc[0], list):
            df[col] = df[col].apply(_patch_files_to_unified_diff)

    # Convert remaining list/dict columns to JSON strings
    for col in df.columns:
        if len(df) > 0 and isinstance(df[col].iloc[0], (list, dict)):
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

    # Ensure all common columns exist
    for col in COMMON_COLUMNS:
        if col not in df.columns and col != "source":
            df[col] = ""

    df["source"] = source_label
    console.print(f"  Loaded {len(df)} instances from {source_label}")
    return df


def load_and_normalize_dataset(hf_name: str, split: str, source_label: str) -> pd.DataFrame:
    """Load a HuggingFace dataset and normalize columns to a common schema."""
    console.print(f"[bold]Loading {hf_name} (split={split})...[/bold]")
    dataset = load_dataset(hf_name, split=split)
    df = pd.DataFrame(dataset)

    # Normalize fail/pass column names (SWE-bench_Pro uses lowercase)
    if "fail_to_pass" in df.columns and "FAIL_TO_PASS" not in df.columns:
        df["FAIL_TO_PASS"] = df["fail_to_pass"]
    if "pass_to_pass" in df.columns and "PASS_TO_PASS" not in df.columns:
        df["PASS_TO_PASS"] = df["pass_to_pass"]

    # Convert list/dict columns to JSON strings for parquet compatibility
    for col in ["FAIL_TO_PASS", "PASS_TO_PASS", "patch_files", "test_patch_files", "test_files"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else (x if isinstance(x, str) else "[]")
            )

    # Ensure all common columns exist
    for col in COMMON_COLUMNS:
        if col not in df.columns and col != "source":
            df[col] = ""

    df["source"] = source_label
    console.print(f"  Loaded {len(df)} instances from {hf_name}")
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CPU-based patch processing for SWE-bench style datasets")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the intermediate parquet file (input for data_gen_problems.py)",
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
        help="Number of instances to process per dataset (default: all)",
    )
    parser.add_argument(
        "--clone_workers",
        type=int,
        default=4,
        help="Number of parallel workers for cloning repos (default: 4)",
    )
    parser.add_argument(
        "--skip_clone",
        action="store_true",
        help="Skip cloning repos (assumes they already exist at --repo_base_path)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["multilingual"],
        help=(
            "Datasets to process. Options: multilingual, pro, featurebench, featbench, all. "
            "Or provide custom HuggingFace dataset name(s). (default: multilingual)"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split override (default: auto per dataset)",
    )

    args = parser.parse_args()

    # Resolve dataset list
    if args.datasets == ["all"]:
        dataset_configs = list(KNOWN_DATASETS.values())
    else:
        dataset_configs = []
        for ds_key in args.datasets:
            if ds_key in KNOWN_DATASETS:
                dataset_configs.append(KNOWN_DATASETS[ds_key])
            else:
                # Treat as custom HuggingFace dataset name
                dataset_configs.append({
                    "hf_name": ds_key,
                    "split": args.split or "test",
                    "source_label": ds_key.replace("/", "_").lower(),
                })

    # Apply split override if provided
    if args.split:
        for cfg in dataset_configs:
            cfg["split"] = args.split

    # Step 1: Load and concatenate all datasets
    console.print(f"[bold]Step 1: Loading {len(dataset_configs)} dataset(s)...[/bold]")
    dfs = []
    for cfg in dataset_configs:
        if "json_url" in cfg:
            df_part = load_json_dataset(cfg["json_url"], cfg["source_label"])
        else:
            df_part = load_and_normalize_dataset(cfg["hf_name"], cfg["split"], cfg["source_label"])
        if args.num_instances:
            df_part = df_part.head(args.num_instances)
            console.print(f"  Trimmed to first {args.num_instances} instances")
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)
    console.print(f"[bold]Total instances across all datasets: {len(df)}[/bold]")

    # Step 2: Clone all repositories
    repo_paths = {}
    if not args.skip_clone:
        console.print("[bold]Step 2: Cloning repositories...[/bold]")
        repo_paths = clone_all_repos(df, args.repo_base_path, max_workers=args.clone_workers)
    else:
        console.print("[yellow]Skipping clone (--skip_clone). Scanning existing repos...[/yellow]")
        for repo_name in df["repo"].unique():
            repo_folder = repo_name.replace("/", "_")
            repo_path = os.path.join(args.repo_base_path, repo_folder)
            if os.path.exists(repo_path):
                repo_paths[repo_name] = repo_path
            else:
                logger.warning(f"Repo not found at {repo_path}, will use patch reconstruction for {repo_name}")
        console.print(f"Found {len(repo_paths)}/{len(df['repo'].unique())} repos locally")

    # Step 3: Process each instance
    console.print("[bold]Step 3: Analyzing patches and generating PR0 patches...[/bold]")
    patch_results = {}

    with Progress() as progress:
        task = progress.add_task("[green]Processing instances...", total=len(df))
        for idx, row in df.iterrows():
            instance = row.to_dict()
            result = process_instance(instance, repo_paths=repo_paths)
            patch_results[instance["instance_id"]] = result
            progress.update(task, advance=1)

    # Step 4: Assemble output DataFrame
    console.print("[bold]Step 4: Assembling output...[/bold]")

    df["functions_modified"] = df["instance_id"].map(
        lambda iid: json.dumps(patch_results[iid]["functions_modified"])
    )
    df["file_to_function_mapping"] = df["instance_id"].map(
        lambda iid: json.dumps(patch_results[iid]["file_to_function_mapping"])
    )
    df["PR0_Patch"] = df["instance_id"].map(
        lambda iid: patch_results[iid]["PR0_Patch"]
    )
    df["original_functions"] = df["instance_id"].map(
        lambda iid: json.dumps(patch_results[iid]["original_functions"])
    )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_parquet(args.output, index=False)

    # Summary stats
    total_functions = sum(len(r["functions_modified"]) for r in patch_results.values())
    total_with_pr0 = sum(1 for r in patch_results.values() if r["PR0_Patch"])
    languages = set()
    for r in patch_results.values():
        if r["analysis"]:
            languages.update(r["analysis"].get("language_map", {}).values())

    console.print(f"\n[bold green]Processing complete![/bold green]")
    console.print(f"  Instances processed: {len(df)}")
    for cfg in dataset_configs:
        src = cfg["source_label"]
        count = len(df[df["source"] == src])
        pr0_count = sum(
            1 for _, row in df[df["source"] == src].iterrows()
            if patch_results[row["instance_id"]]["PR0_Patch"]
        )
        console.print(f"    {src}: {count} instances, {pr0_count} with PR0_Patch")
    console.print(f"  Total functions found: {total_functions}")
    console.print(f"  Instances with PR0_Patch: {total_with_pr0}")
    console.print(f"  Languages detected: {', '.join(sorted(languages))}")
    console.print(f"  Repos cloned to: {os.path.abspath(args.repo_base_path)}")
    console.print(f"  Output saved to: {args.output}")
    console.print(f"\n[bold]Next step:[/bold] Run data_gen_problems.py --input {args.output} --output final.parquet")


if __name__ == "__main__":
    main()
