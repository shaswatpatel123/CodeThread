"""
Function extraction utilities for tracking code modifications.
Reuses AST parsing patterns from function_modifier.py.
"""

import ast
import logging

import lizard

logger = logging.getLogger(__name__)

# Files/patterns to ignore (from some analysis/main.py)
IGNORE_EXTENSIONS = {'.md', '.diff', '.sh', '.txt', '.rst',
                    '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg'}
IGNORE_PATTERNS = ['test', 'Test', 'TEST']
IGNORE_DIRS = {'test', 'tests', '__tests__', 'examples', 'demo', 'docs'}

LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".ts", ".tsx"],  # JS repos often contain TS files
    "typescript": [".ts", ".tsx", ".js", ".jsx"],
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
    "c": [".c", ".h"],
    "ruby": [".rb"],
}

# Language-specific test file patterns
_TEST_FILENAME_PATTERNS = {
    "go": ["_test.go"],
}
_TEST_DIR_PATTERNS = {
    "java": {"src/test"},
}


_LANG_ALIASES = {"js": "javascript", "ts": "typescript"}


def filter_source_files(files: list[str], repo: str, language: str = "python") -> list[str]:
    """Filter to source files of the given language, excluding tests."""
    language = _LANG_ALIASES.get(language, language)
    extensions = LANGUAGE_EXTENSIONS.get(language, LANGUAGE_EXTENSIONS["python"])
    test_suffixes = _TEST_FILENAME_PATTERNS.get(language, [])
    extra_test_dirs = _TEST_DIR_PATTERNS.get(language, set())

    filtered = []
    for filepath in files:
        if not any(filepath.endswith(ext) for ext in extensions):
            continue
        if any(filepath.endswith(ext) for ext in IGNORE_EXTENSIONS):
            continue
        # Language-specific test file suffixes (e.g. Go _test.go)
        if any(filepath.endswith(s) for s in test_suffixes):
            continue
        filename = filepath.split('/')[-1]
        if any(pattern in filename for pattern in IGNORE_PATTERNS):
            continue
        path_parts = filepath.split('/')
        if any(dir_name in IGNORE_DIRS for dir_name in path_parts):
            continue
        # Extra test dirs (e.g. Java src/test/)
        if extra_test_dirs and any(td in filepath for td in extra_test_dirs):
            continue
        filtered.append(filepath)
    return filtered


def filter_python_files(files: list[str], repo: str) -> list[str]:
    """Filter out non-Python files and test files."""
    return filter_source_files(files, repo, language="python")


def extract_all_functions(file_content: str) -> dict[str, ast.AST]:
    """
    Extract all functions from file content with full paths.
    
    Handles:
    - Top-level functions: "function_name"
    - Async functions: "async_function_name"
    - Class methods: "ClassName.method_name"
    - Nested functions: "outer.inner"
    
    Args:
        file_content: Source code content
        
    Returns:
        Dictionary mapping function paths to AST nodes
        Example: {"func": node, "Class.method": node, "outer.inner": node}
        
    Raises:
        SyntaxError: If file cannot be parsed
    """
    tree = ast.parse(file_content)
    functions = {}
    
    def walk_functions(nodes, prefix=""):
        """
        Recursively walk AST nodes to find all functions.
        
        Args:
            nodes: List of AST nodes to walk
            prefix: Current path prefix (e.g., "ClassName" or "outer_func")
        """
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Build the full path name
                if prefix:
                    full_name = f"{prefix}.{node.name}"
                else:
                    full_name = node.name
                
                functions[full_name] = node
                
                # Recursively check for nested functions
                walk_functions(node.body, full_name)
            
            elif isinstance(node, ast.ClassDef):
                # For classes, add class name to prefix and walk methods
                class_prefix = f"{prefix}.{node.name}" if prefix else node.name
                walk_functions(node.body, class_prefix)
    
    # Start walking from the top-level module body
    walk_functions(tree.body)
    
    return functions


def compare_function_asts(node1: ast.AST, node2: ast.AST) -> bool:
    """
    Compare two AST nodes to check if they're identical.
    
    Args:
        node1: First AST node
        node2: Second AST node
        
    Returns:
        True if identical, False if different
    """
    # Use ast.dump() to get string representation and compare
    # This captures signature and body changes
    return ast.dump(node1) == ast.dump(node2)


def detect_crud_operations(old_funcs: dict[str, ast.AST], new_funcs: dict[str, ast.AST]) -> list[dict]:
    """
    Detect CRUD operations by comparing old and new function sets.
    
    Args:
        old_funcs: Functions from old version (name -> AST node)
        new_funcs: Functions from new version (name -> AST node)
        
    Returns:
        List of dicts with CRUD information:
        [{"name": "func_name", "crud": "ADDED/MODIFIED/DELETED", "info": "SUCCESS"}]
    """
    results = []

    old_names = set(old_funcs.keys())
    new_names = set(new_funcs.keys())

    # ADDED: Functions in new but not in old
    added = new_names - old_names
    for func_name in sorted(added):
        results.append({
            "name": func_name,
            "crud": "ADDED",
            "info": "SUCCESS"
        })

    # DELETED: Functions in old but not in new
    deleted = old_names - new_names
    for func_name in sorted(deleted):
        results.append({
            "name": func_name,
            "crud": "DELETED",
            "info": "SUCCESS"
        })

    # MODIFIED: Functions in both, compare AST to check if changed
    common = old_names & new_names
    for func_name in sorted(common):
        old_node = old_funcs[func_name]
        new_node = new_funcs[func_name]

        # If AST dumps differ, function was modified
        if not compare_function_asts(old_node, new_node):
            results.append({
                "name": func_name,
                "crud": "MODIFIED",
                "info": "SUCCESS"
            })

    return results


# ─── Multilingual support (lizard-based) ─────────────────────────────────────


def extract_all_functions_multilingual(content: str, file_path: str) -> dict[str, tuple[int, int]]:
    """Extract function names and line ranges using lizard (works for any language).

    Returns:
        {"ClassName::method": (start_line, end_line), "func": (start, end), ...}
    """
    fname = file_path or "tmp.txt"
    # lizard needs a filename with the right extension to pick the language
    analysis = lizard.analyze_file.analyze_source_code(fname, content)

    functions: dict[str, tuple[int, int]] = {}
    for func in analysis.function_list:
        # long_name includes class prefix separated by "::" (e.g. "MyClass::method")
        # Normalise to dot notation for consistency with the Python path
        name = func.long_name.replace("::", ".")
        functions[name] = (func.start_line, func.end_line)
    return functions


def detect_crud_operations_multilingual(
    old_funcs: dict[str, tuple[int, int]],
    new_funcs: dict[str, tuple[int, int]],
    old_content: str,
    new_content: str,
) -> list[dict]:
    """Name-based CRUD detection for non-Python languages.

    Compares extracted line ranges to determine if a function body changed.
    Returns same schema as ``detect_crud_operations``.
    """
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    def _extract_body(lines, start, end):
        return "\n".join(lines[start - 1 : end])

    results = []
    old_names = set(old_funcs.keys())
    new_names = set(new_funcs.keys())

    for func_name in sorted(new_names - old_names):
        results.append({"name": func_name, "crud": "ADDED", "info": "SUCCESS"})

    for func_name in sorted(old_names - new_names):
        results.append({"name": func_name, "crud": "DELETED", "info": "SUCCESS"})

    for func_name in sorted(old_names & new_names):
        old_start, old_end = old_funcs[func_name]
        new_start, new_end = new_funcs[func_name]
        old_body = _extract_body(old_lines, old_start, old_end)
        new_body = _extract_body(new_lines, new_start, new_end)
        if old_body != new_body:
            results.append({"name": func_name, "crud": "MODIFIED", "info": "SUCCESS"})

    return results

