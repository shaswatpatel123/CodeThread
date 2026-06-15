"""Multi-language maintainability metrics using rust-code-analysis, Go tools, and lizard."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import lizard

logger = logging.getLogger(__name__)

# Tool paths (configurable via env vars)
RCA_CLI = os.getenv("RCA_CLI", "/scratch/spp9399/.cargo/bin/rust-code-analysis-cli")
GO_HALSTEAD_CLI = os.getenv("GO_HALSTEAD_CLI", "/scratch/spp9399/.go/bin/go-halstead")
GOCOGNIT = os.getenv("GOCOGNIT_CLI", "/scratch/spp9399/.go/bin/gocognit")

_LANG_ALIASES = {"js": "javascript", "ts": "typescript"}
_RCA_LANGUAGES = {"rust", "c", "java", "javascript", "typescript"}
_EXT_MAP = {
    "rust": ".rs", "c": ".c", "java": ".java",
    "javascript": ".js", "typescript": ".ts",
    "go": ".go", "ruby": ".rb",
}


def _rnd(x, nd=3):
    try:
        return round(float(x), nd)
    except Exception:
        return None


def _empty_schema() -> Dict[str, Any]:
    return {
        "maintainability_index": {"file_level": {"mi_no_comments": None, "mi_comments": None}, "per_function": []},
        "cyclomatic_complexity": {"file_level": {"total": None, "average": None, "max": None, "function_count": 0}, "per_function": []},
        "cognitive_complexity": {"file_level": {"total": None}, "per_function": []},
        "halstead": {"file_level": {}, "per_function": []},
        "raw_metrics": {"file_level": {"sloc": None, "lloc": None, "cloc": None, "blank": None}, "per_function": []},
    }


# ─── rust-code-analysis backend ──────────────────────────────────────────────


def _parse_rca_spaces(spaces: dict, per_func: list[dict], per_cc: list[dict],
                      per_cog: list[dict], per_h: list[dict], per_mi: list[dict]):
    """Recursively walk RCA spaces to collect per-function metrics."""
    kind = spaces.get("kind", "")
    name = spaces.get("name", "")
    metrics = spaces.get("metrics", {})

    if kind == "function":
        cc_val = metrics.get("cyclomatic", {}).get("sum", 0)
        cog_val = metrics.get("cognitive", {}).get("sum", 0)
        hal = metrics.get("halstead", {})
        mi_data = metrics.get("mi", {})
        loc = metrics.get("loc", {})

        per_cc.append({"name": name, "classname": None, "complexity": cc_val})
        per_cog.append({"name": name, "classname": None, "complexity": cog_val})
        h_entry = {"name": name, "classname": None}
        for k in ("h1", "h2", "N1", "N2", "vocabulary", "length",
                   "calculated_length", "volume", "difficulty", "effort", "time", "bugs"):
            h_entry[k] = _rnd(hal.get(k))
        per_h.append(h_entry)
        mi_val = _rnd(mi_data.get("mi_visual_studio"))
        per_mi.append({"name": name, "classname": None, "mi_no_comments": mi_val, "mi_comments": mi_val})

        per_func.append({"name": name, "sloc": loc.get("sloc", 0), "lloc": loc.get("lloc", 0)})

    for sub in spaces.get("spaces", []):
        _parse_rca_spaces(sub, per_func, per_cc, per_cog, per_h, per_mi)


def _calc_rca(content: str, file_path: str | None, language: str) -> Dict[str, Any]:
    # Prefer actual file extension so RCA picks the right parser (e.g. .ts vs .js)
    if file_path:
        ext = Path(file_path).suffix or _EXT_MAP.get(language, ".c")
    else:
        ext = _EXT_MAP.get(language, ".c")
    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [RCA_CLI, "-m", "-p", tmp_path, "-O", "json"],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        logger.warning("rust-code-analysis-cli not found, falling back to lizard")
        return _calc_lizard(content, file_path, language)
    except subprocess.TimeoutExpired:
        logger.warning("rust-code-analysis-cli timed out")
        return _calc_lizard(content, file_path, language)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        logger.warning("rust-code-analysis-cli failed: %s", result.stderr[:200])
        return _calc_lizard(content, file_path, language)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("rust-code-analysis-cli returned invalid JSON")
        return _calc_lizard(content, file_path, language)

    m = _empty_schema()

    # Top-level spaces entry
    spaces = data if isinstance(data, dict) else (data[0] if data else {})
    top_metrics = spaces.get("metrics", {})

    # File-level metrics from top-level
    cc_top = top_metrics.get("cyclomatic", {})
    cog_top = top_metrics.get("cognitive", {})
    hal_top = top_metrics.get("halstead", {})
    mi_top = top_metrics.get("mi", {})
    loc_top = top_metrics.get("loc", {})

    m["cyclomatic_complexity"]["file_level"]["total"] = cc_top.get("sum", 0)
    m["cognitive_complexity"]["file_level"]["total"] = cog_top.get("sum", 0)

    for k in ("h1", "h2", "N1", "N2", "vocabulary", "length",
              "calculated_length", "volume", "difficulty", "effort", "time", "bugs"):
        m["halstead"]["file_level"][k] = _rnd(hal_top.get(k))

    mi_val = _rnd(mi_top.get("mi_visual_studio"))
    m["maintainability_index"]["file_level"]["mi_no_comments"] = mi_val
    m["maintainability_index"]["file_level"]["mi_comments"] = mi_val

    m["raw_metrics"]["file_level"]["sloc"] = loc_top.get("sloc", 0)
    m["raw_metrics"]["file_level"]["lloc"] = loc_top.get("lloc", 0)
    m["raw_metrics"]["file_level"]["cloc"] = loc_top.get("cloc", 0)
    m["raw_metrics"]["file_level"]["blank"] = loc_top.get("blank", 0)

    # Per-function from nested spaces
    per_func, per_cc, per_cog, per_h, per_mi = [], [], [], [], []
    for sub in spaces.get("spaces", []):
        _parse_rca_spaces(sub, per_func, per_cc, per_cog, per_h, per_mi)

    m["cyclomatic_complexity"]["per_function"] = per_cc
    m["cognitive_complexity"]["per_function"] = per_cog
    m["halstead"]["per_function"] = per_h
    m["maintainability_index"]["per_function"] = per_mi
    m["raw_metrics"]["per_function"] = per_func

    # Compute averages
    cc_vals = [f["complexity"] for f in per_cc]
    if cc_vals:
        m["cyclomatic_complexity"]["file_level"]["average"] = _rnd(sum(cc_vals) / len(cc_vals))
        m["cyclomatic_complexity"]["file_level"]["max"] = max(cc_vals)
        m["cyclomatic_complexity"]["file_level"]["function_count"] = len(cc_vals)

    return m


# ─── Go backend ──────────────────────────────────────────────────────────────


def _calc_go(content: str, file_path: str | None, language: str) -> Dict[str, Any]:
    """Go metrics backend.
    - CC:      lizard  (AST-level, no import resolution needed, consistent with Ruby)
    - Halstead + MI: go-halstead (stdlib-only Go tool in tools/go-halstead/, no imports needed)
    - CogC:    gocognit (AST-level, unchanged)
    - LOC:     lizard
    """
    m = _empty_schema()
    fname = file_path or "main.go"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    try:
        # CC + LOC via lizard (no import resolution, consistent across languages)
        analysis = lizard.analyze_file.analyze_source_code(fname, content)
        for func in analysis.function_list:
            name = func.long_name.replace("::", ".")
            m["cyclomatic_complexity"]["per_function"].append(
                {"name": name, "classname": None, "complexity": func.cyclomatic_complexity})
        m["raw_metrics"]["file_level"]["sloc"] = analysis.nloc
        m["raw_metrics"]["file_level"]["lloc"] = analysis.nloc

        # Halstead + MI via go-halstead (tools/go-halstead/main.go, stdlib only)
        try:
            result = subprocess.run(
                [GO_HALSTEAD_CLI, tmp_path],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                for fn in data.get("functions", []):
                    name = fn["name"]
                    m["halstead"]["per_function"].append({
                        "name": name, "classname": None,
                        "h1": fn.get("n1"), "h2": fn.get("n2"),
                        "N1": fn.get("N1"), "N2": fn.get("N2"),
                        "vocabulary": fn.get("vocabulary"),
                        "length": fn.get("length"),
                        "volume": fn.get("volume"),
                        "difficulty": fn.get("difficulty"),
                        "effort": fn.get("effort"),
                        "calculated_length": None, "time": None, "bugs": None,
                    })
                    mi_val = fn.get("mi")
                    m["maintainability_index"]["per_function"].append(
                        {"name": name, "classname": None, "mi_no_comments": mi_val, "mi_comments": mi_val})
            elif result.stderr:
                logger.warning("go-halstead stderr: %s", result.stderr[:200])
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.warning("go-halstead failed: %s", e)

        # CogC via gocognit (unchanged — works without import resolution)
        try:
            result = subprocess.run(
                [GOCOGNIT, "-json", "-over", "-1", tmp_path],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                cog_data = json.loads(result.stdout)
                for entry in cog_data:
                    name = entry.get("FuncName", "")
                    complexity = entry.get("Complexity", 0)
                    m["cognitive_complexity"]["per_function"].append(
                        {"name": name, "classname": None, "complexity": complexity})
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.warning("gocognit failed: %s", e)

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # File-level aggregation from per-function data
    cc_vals = [f["complexity"] for f in m["cyclomatic_complexity"]["per_function"] if f["complexity"] is not None]
    if cc_vals:
        m["cyclomatic_complexity"]["file_level"]["total"] = sum(cc_vals)
        m["cyclomatic_complexity"]["file_level"]["average"] = _rnd(sum(cc_vals) / len(cc_vals))
        m["cyclomatic_complexity"]["file_level"]["max"] = max(cc_vals)
        m["cyclomatic_complexity"]["file_level"]["function_count"] = len(cc_vals)

    cog_vals = [f["complexity"] for f in m["cognitive_complexity"]["per_function"] if f["complexity"] is not None]
    if cog_vals:
        m["cognitive_complexity"]["file_level"]["total"] = sum(cog_vals)

    hal_vols = [f["volume"] for f in m["halstead"]["per_function"] if f.get("volume") is not None]
    if hal_vols:
        m["halstead"]["file_level"]["volume"] = _rnd(sum(hal_vols))

    mi_vals = [f["mi_no_comments"] for f in m["maintainability_index"]["per_function"] if f["mi_no_comments"] is not None]
    if mi_vals:
        m["maintainability_index"]["file_level"]["mi_no_comments"] = _rnd(sum(mi_vals) / len(mi_vals))
        m["maintainability_index"]["file_level"]["mi_comments"] = m["maintainability_index"]["file_level"]["mi_no_comments"]

    return m


# ─── lizard backend (Ruby, fallback) ────────────────────────────────────────


def _add_lizard_loc(content: str, file_path: str, m: Dict[str, Any]):
    """Use lizard to fill in LOC/raw_metrics."""
    fname = file_path or "tmp.txt"
    analysis = lizard.analyze_file.analyze_source_code(fname, content)

    m["raw_metrics"]["file_level"]["sloc"] = analysis.nloc
    m["raw_metrics"]["file_level"]["lloc"] = analysis.nloc  # lizard doesn't distinguish lloc
    m["raw_metrics"]["file_level"]["cloc"] = None
    m["raw_metrics"]["file_level"]["blank"] = None


def _calc_lizard(content: str, file_path: str | None, language: str) -> Dict[str, Any]:
    """Lizard-only backend: CC per function, no cognitive/halstead/MI."""
    m = _empty_schema()
    ext = _EXT_MAP.get(language, ".rb")
    fname = file_path or f"tmp{ext}"

    analysis = lizard.analyze_file.analyze_source_code(fname, content)

    per_cc = []
    for func in analysis.function_list:
        name = func.long_name.replace("::", ".")
        per_cc.append({"name": name, "classname": None, "complexity": func.cyclomatic_complexity})

    cc_vals = [f["complexity"] for f in per_cc]
    m["cyclomatic_complexity"]["per_function"] = per_cc
    if cc_vals:
        m["cyclomatic_complexity"]["file_level"]["total"] = sum(cc_vals)
        m["cyclomatic_complexity"]["file_level"]["average"] = _rnd(sum(cc_vals) / len(cc_vals))
        m["cyclomatic_complexity"]["file_level"]["max"] = max(cc_vals)
        m["cyclomatic_complexity"]["file_level"]["function_count"] = len(cc_vals)

    m["raw_metrics"]["file_level"]["sloc"] = analysis.nloc
    m["raw_metrics"]["file_level"]["lloc"] = analysis.nloc

    return m


# ─── public dispatch ─────────────────────────────────────────────────────────


def calculate_metrics_multilingual(
    new_content: str, old_content: str | None, language: str, file_path: str | None = None,
) -> Dict[str, Any]:
    """Same output schema as calculate_metrics(). Dispatches by language."""
    language = _LANG_ALIASES.get(language, language)
    if language in _RCA_LANGUAGES:
        calc = _calc_rca
    elif language == "go":
        calc = _calc_go
    else:
        calc = _calc_lizard

    result = {"new": calc(new_content, file_path, language)}
    result["old"] = calc(old_content, file_path, language) if old_content is not None else None
    return result
