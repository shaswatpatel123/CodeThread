from __future__ import annotations

from radon.complexity import cc_visit
from radon.raw import analyze as raw_analyze
from radon.metrics import h_visit, mi_rank
from complexipy import code_complexity
from typing import Dict, Any
import ast
import math
import textwrap


def calculate_metrics(
    new_content: str, old_content: str | None,
    language: str = "python", file_path: str | None = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive code metrics (cyclomatic, raw, halstead, cognitive, MI).

    For non-Python languages, delegates to calculate_metrics_multilingual.
    """
    if language != "python":
        from minisweagent.environments.utils.calculate_metrics_multilingual import calculate_metrics_multilingual
        return calculate_metrics_multilingual(new_content, old_content, language, file_path)

    result = {"new": _calc(new_content)}
    result["old"] = _calc(old_content) if old_content is not None else None
    return result


# ─── helpers ───────────────────────────────────────────────────────────────────

def _class_map(content: str) -> list[tuple[int, int, str]]:
    """Return [(start_line, end_line, classname), ...] from AST."""
    try:
        return [(n.lineno, n.end_lineno or n.lineno, n.name)
                for n in ast.walk(ast.parse(content)) if isinstance(n, ast.ClassDef)]
    except Exception:
        return []


def _classname(line: int, cmap: list) -> str | None:
    for s, e, name in cmap:
        if s <= line <= e:
            return name
    return None


def _rnd(x, nd=3):
    try:
        return round(float(x), nd)
    except Exception:
        return None


def _halstead_dict(h) -> dict:
    """All Halstead sub-metrics from a radon Halstead namedtuple."""
    return {k: _rnd(getattr(h, k, None)) for k in
            ("h1", "h2", "N1", "N2", "vocabulary", "length",
             "calculated_length", "volume", "difficulty", "effort", "time", "bugs")}


def _compute_mi(vol, cc, sloc, comment_pct) -> dict:
    """Maintainability Index with and without comments."""
    if any(v is None or v <= 0 for v in (vol, cc, sloc)):
        return {"mi_comments": 100.0, "mi_no_comments": 100.0, "rank": "A"}
    base = 171 - 5.2 * math.log(vol) - 0.23 * cc - 16.2 * math.log(sloc)
    mi_no = max(0, 100 * base / 171)
    scale = math.sqrt(2.46 * math.radians(max(0, comment_pct)))
    mi_c = max(0, 100 * (base + 50 * math.sin(scale)) / 171)
    return {"mi_comments": _rnd(mi_c), "mi_no_comments": _rnd(mi_no), "rank": mi_rank(mi_c)}


def _func_src(lines: list[str], start: int, all_starts: list[int]) -> str:
    """Extract source from start line to the next function or EOF (dedented)."""
    nxt = [s for s in all_starts if s > start]
    end = (min(nxt) - 1) if nxt else len(lines)
    return textwrap.dedent("\n".join(lines[start - 1 : end]))


# ─── core ──────────────────────────────────────────────────────────────────────

def _calc(content: str) -> Dict[str, Any]:
    content = textwrap.dedent(content)
    lines = content.splitlines()
    cmap = _class_map(content)
    m: Dict[str, Any] = {}

    # 1 ── Cyclomatic complexity (radon) ──────────────────────────────────────
    try:
        cc_items = cc_visit(content)
        total_cc = sum(i.complexity for i in cc_items)
        m["cyclomatic_complexity"] = {
            "file_level": {
                "total": total_cc,
                "average": _rnd(total_cc / len(cc_items)) if cc_items else 0.0,
                "max": max((i.complexity for i in cc_items), default=0),
                "function_count": len(cc_items),
            },
            "per_function": [{
                "name": i.name,
                "classname": getattr(i, "classname", None),
                "complexity": i.complexity,
                "line": i.lineno,
                "is_method": getattr(i, "is_method", False),
            } for i in cc_items],
        }
    except Exception as e:
        raise CyclomaticComplexityError(str(e))

    # Pre-extract per-function source (reused by raw, halstead, MI)
    all_starts = [i.lineno for i in cc_items]
    funcs = [(i.name, getattr(i, "classname", None), i.lineno,
              _func_src(lines, i.lineno, all_starts)) for i in cc_items]

    # 2 ── Raw metrics (radon) ────────────────────────────────────────────────
    try:
        raw = raw_analyze(content)
        per_raw = []
        for name, cls, _, src in funcs:
            try:
                r = raw_analyze(src)
                per_raw.append({"name": name, "classname": cls,
                                "loc": r.loc, "lloc": r.lloc, "sloc": r.sloc,
                                "comments": r.comments, "blank": r.blank})
            except Exception:
                continue
        m["raw_metrics"] = {
            "file_level": {
                "loc": raw.loc, "lloc": raw.lloc, "sloc": raw.sloc,
                "comments": raw.comments, "multi": raw.multi, "blank": raw.blank,
                "single_comments": getattr(raw, "single_comments", 0),
            },
            "per_function": per_raw,
        }
    except Exception as e:
        raise RawMetricsError(str(e))

    # 3 ── Halstead metrics (radon) ───────────────────────────────────────────
    try:
        per_h = []
        for name, cls, _, src in funcs:
            try:
                h = h_visit(src).total
                entry = {"name": name, "classname": cls}
                entry.update(_halstead_dict(h))
                per_h.append(entry)
            except Exception:
                continue
        m["halstead"] = {
            "file_level": _halstead_dict(h_visit(content).total),
            "per_function": per_h,
        }
    except Exception as e:
        raise HalsteadError(str(e))

    # 4 ── Cognitive complexity (complexipy) ──────────────────────────────────
    try:
        cog = code_complexity(content)
        per_cog = []
        if hasattr(cog, "functions") and cog.functions:
            per_cog = [{"name": f.name,
                        "classname": _classname(f.line_start, cmap),
                        "complexity": f.complexity,
                        "line_start": f.line_start, "line_end": f.line_end}
                       for f in cog.functions]
        m["cognitive_complexity"] = {
            "file_level": {"total": int(getattr(cog, "complexity", 0))},
            "per_function": per_cog,
        }
    except Exception as e:
        raise CognitiveComplexityError(str(e))

    # 5 ── Maintainability Index ──────────────────────────────────────────────
    try:
        fl = m["raw_metrics"]["file_level"]
        f_loc = fl["loc"] or 1
        file_cpct = (fl["comments"] / f_loc) * 100 if f_loc > 0 else 0

        # file-level MI
        m["maintainability_index"] = {
            "file_level": _compute_mi(
                m["halstead"]["file_level"].get("volume", 0),
                m["cyclomatic_complexity"]["file_level"]["total"],
                fl["sloc"], file_cpct),
            "per_function": [],
        }

        # per-function MI  (join cc, halstead, raw by (name, classname) key)
        cc_k = {(f["name"], f["classname"]): f["complexity"]
                for f in m["cyclomatic_complexity"]["per_function"]}
        h_k  = {(f["name"], f["classname"]): f.get("volume", 0)
                for f in m["halstead"]["per_function"]}
        r_k  = {(f["name"], f["classname"]): f
                for f in m["raw_metrics"]["per_function"]}

        for key in cc_k:
            r = r_k.get(key, {})
            s = r.get("sloc", 0) or 0
            loc = r.get("loc", 1) or 1
            cpct = (r.get("comments", 0) / loc) * 100 if loc > 0 else 0
            mi = _compute_mi(h_k.get(key, 0) or 0, cc_k[key], s, cpct)
            mi["name"], mi["classname"] = key
            m["maintainability_index"]["per_function"].append(mi)
    except Exception as e:
        raise MaintainabilityIndexError(str(e))

    return m


# ─── exceptions ────────────────────────────────────────────────────────────────

class CyclomaticComplexityError(Exception): pass
class RawMetricsError(Exception): pass
class CognitiveComplexityError(Exception): pass
class HalsteadError(Exception): pass
class MaintainabilityIndexError(Exception): pass

