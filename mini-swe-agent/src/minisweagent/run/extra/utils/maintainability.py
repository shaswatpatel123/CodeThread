from __future__ import annotations

from typing import Any

from complexipy import code_complexity, file_complexity
from radon.complexity import cc_rank, cc_visit
from radon.metrics import h_visit, mi_rank, mi_visit
from radon.raw import analyze as raw_analyze


def analyze_cognitive_from_file(path: str) -> dict[str, Any]:
    res = file_complexity(path)  # complexipy return type with .path, .complexity, .functions
    return {
        "file": getattr(res, "path", path),
        "complexity": int(getattr(res, "complexity", 0)),
        "functions": [
            {
                "name": getattr(f, "name", "<unknown>"),
                "complexity": int(getattr(f, "complexity", 0)),
                "line_start": int(getattr(f, "line_start", 0)),
                "line_end": int(getattr(f, "line_end", 0)),
            }
            for f in getattr(res, "functions", []) or []
        ],
    }


def _cc_symbol(item) -> str:
    """
    Make a stable 'symbol' name for a CC item.
    Prefers ClassName.func for methods, else func name.
    """
    cls = getattr(item, "classname", None)
    name = getattr(item, "name", "<unknown>")
    return f"{cls}.{name}" if cls else name


def _round_or_none(x: float | None, nd: int = 3) -> float | None:
    try:
        return round(float(x), nd)
    except Exception:
        return None

def analyze_code(code: str, *, filename: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"source": {"filename": filename, "ok": True}}
    lines = code.splitlines(keepends=True)

    # ---- Cyclomatic Complexity (per function) + per-function MI/raw/Halstead ----
    try:
        cc_items = cc_visit(code)
        cc_funcs = []
        total = 0
        maxcc = 0

        for it in cc_items:
            c = int(getattr(it, "complexity", 0))
            total += c
            maxcc = max(maxcc, c)

            lineno = getattr(it, "lineno", None)
            endline = getattr(it, "endline", None)

            mi_func = None
            raw_func = None
            hal_func = None
            cog_func = None

            if isinstance(lineno, int) and isinstance(endline, int):
                try:
                    snippet = "".join(lines[lineno - 1 : endline])

                    # per-function MI
                    try:
                        mi_score_f = float(mi_visit(snippet, multi=False))
                        mi_func = {
                            "score": round(mi_score_f, 3),
                            "rank": mi_rank(mi_score_f),
                        }
                    except Exception:
                        mi_func = None

                    # per-function raw
                    try:
                        raw_f = raw_analyze(snippet)
                        raw_func = {
                            "loc": raw_f.loc,
                            "lloc": raw_f.lloc,
                            "sloc": raw_f.sloc,
                            "comments": raw_f.comments,
                            "multi": raw_f.multi,
                            "blank": raw_f.blank,
                            "single_comments": getattr(raw_f, "single_comments", None),
                        }
                    except Exception:
                        raw_func = None

                    # per-function Halstead
                    try:
                        h_res_f = h_visit(snippet)
                        hr = h_res_f.total  # HalsteadReport for this snippet
                        hal_func = {
                            "h1": hr.h1,
                            "h2": hr.h2,
                            "N1": hr.N1,
                            "N2": hr.N2,
                            "vocabulary": hr.h,
                            "length": hr.N,
                            "calculated_length": hr.calculated_length,
                            "volume": _round_or_none(hr.volume),
                            "difficulty": _round_or_none(hr.difficulty),
                            "effort": _round_or_none(hr.effort),
                            "time": _round_or_none(hr.time),
                            "bugs": _round_or_none(hr.bugs),
                        }
                    except Exception:
                        hal_func = None

                    # per-function cognitive
                    try:
                        cog_res = code_complexity(snippet)
                        cog_func = { "complexity": cog_res.complexity }
                    except Exception:
                        cog_func = None
                except Exception:
                    pass  # slicing failed; skip per-function metrics

            func_entry = {
                "symbol": _cc_symbol(it),
                "name": getattr(it, "name", None),
                "class": getattr(it, "classname", None),
                "lineno": lineno,
                "endline": endline,
                "complexity": c,
                "rank": cc_rank(c),
            }
            if mi_func is not None:
                func_entry["mi"] = mi_func
            if raw_func is not None:
                func_entry["raw"] = raw_func
            if hal_func is not None:
                func_entry["halstead"] = hal_func
            if cog_func is not None:
                func_entry['cognitive_complexity'] = cog_func

            cc_funcs.append(func_entry)

        avg = (total / len(cc_funcs)) if cc_funcs else 0.0
        result["cc"] = {
            "functions": cc_funcs,
            "summary": {
                "count": len(cc_funcs),
                "total": total,
                "avg": round(avg, 3),
                "max": maxcc,
            },
        }
    except Exception as e:
        result["source"]["ok"] = False
        result["cc_error"] = repr(e)

    return result


def analyze(file_path: str) -> dict[str, Any]:
    res: dict[str, Any] = {}
    with open(file_path, encoding="utf-8") as f:
        res = analyze_code(f.read(), filename=file_path)

    # (minor typo, but keeping your key name as-is)
    # res["cogative_complexity"] = analyze_cognitive_from_file(file_path)

    return res

