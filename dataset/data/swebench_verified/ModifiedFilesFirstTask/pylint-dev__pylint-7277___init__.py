# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# For details: https://github.com/PyCQA/pylint/blob/main/LICENSE
# Copyright (c) https://github.com/PyCQA/pylint/blob/main/CONTRIBUTORS.txt

from __future__ import annotations

__all__ = [
    "__version__",
    "version",
    "modify_sys_path",
    "run_pylint",
    "run_epylint",
    "run_symilar",
    "run_pyreverse",
]

import os
import sys
from collections.abc import Sequence
from typing import NoReturn

from pylint.__pkginfo__ import __version__

# pylint: disable=import-outside-toplevel


def run_pylint(argv: Sequence[str] | None = None) -> None:
    """Run pylint.

    argv can be a sequence of strings normally supplied as arguments on the command line
    """
    from pylint.lint import Run as PylintRun

    try:
        PylintRun(argv or sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(1)


def _run_pylint_config(argv: Sequence[str] | None = None) -> None:
    """Run pylint-config.

    argv can be a sequence of strings normally supplied as arguments on the command line
    """
    from pylint.lint.run import _PylintConfigRun

    _PylintConfigRun(argv or sys.argv[1:])


def run_epylint(argv: Sequence[str] | None = None) -> NoReturn:
    """Run epylint.

    argv can be a list of strings normally supplied as arguments on the command line
    """
    from pylint.epylint import Run as EpylintRun

    EpylintRun(argv)


def run_pyreverse(argv: Sequence[str] | None = None) -> NoReturn:  # type: ignore[misc]
    """Run pyreverse.

    argv can be a sequence of strings normally supplied as arguments on the command line
    """
    from pylint.pyreverse.main import Run as PyreverseRun

    PyreverseRun(argv or sys.argv[1:])


def run_symilar(argv: Sequence[str] | None = None) -> NoReturn:
    """Run symilar.

    argv can be a sequence of strings normally supplied as arguments on the command line
    """
    from pylint.checkers.similar import Run as SimilarRun

    SimilarRun(argv or sys.argv[1:])


def modify_sys_path() -> None:
    """
    Implement your code here
    """
    return None


version = __version__
