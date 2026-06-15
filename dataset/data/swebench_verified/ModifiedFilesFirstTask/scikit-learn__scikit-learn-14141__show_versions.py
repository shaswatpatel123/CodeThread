"""
Utility methods to print system info for debugging

adapted from :func:`pandas.show_versions`
"""
# License: BSD 3 clause

import platform
import sys
import importlib


def _get_sys_info():
    """System information

    Return
    ------
    sys_info : dict
        system and Python version information

    """
    python = sys.version.replace('\n', ' ')

    blob = [
        ("python", python),
        ('executable', sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info():
    """
    Implement your code here
    """
    return None


def _get_blas_info():
    """Information on system BLAS

    Uses the `scikit-learn` builtin method
    :func:`sklearn._build_utils.get_blas_info` which may fail from time to time

    Returns
    -------
    blas_info: dict
        system BLAS information

    """
    from .._build_utils import get_blas_info

    cblas_libs, blas_dict = get_blas_info()

    macros = ['{key}={val}'.format(key=a, val=b)
              for (a, b) in blas_dict.get('define_macros', [])]

    blas_blob = [
        ('macros', ', '.join(macros)),
        ('lib_dirs', ':'.join(blas_dict.get('library_dirs', ''))),
        ('cblas_libs', ', '.join(cblas_libs)),
    ]

    return dict(blas_blob)


def show_versions():
    "Print useful debugging information"

    sys_info = _get_sys_info()
    deps_info = _get_deps_info()
    blas_info = _get_blas_info()

    print('\nSystem:')
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))

    print('\nBLAS:')
    for k, stat in blas_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))

    print('\nPython deps:')
    for k, stat in deps_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))
