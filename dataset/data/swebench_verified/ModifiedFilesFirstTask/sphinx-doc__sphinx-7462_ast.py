"""
    sphinx.pycode.ast
    ~~~~~~~~~~~~~~~~~

    Helpers for AST (Abstract Syntax Tree).

    :copyright: Copyright 2007-2020 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import sys
from typing import Dict, List, Type

if sys.version_info > (3, 8):
    import ast
else:
    try:
        # use typed_ast module if installed
        from typed_ast import ast3 as ast
    except ImportError:
        import ast  # type: ignore


OPERATORS = {
    ast.Add: "+",
    ast.And: "and",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Invert: "~",
    ast.LShift: "<<",
    ast.MatMult: "@",
    ast.Mult: "*",
    ast.Mod: "%",
    ast.Not: "not",
    ast.Pow: "**",
    ast.Or: "or",
    ast.RShift: ">>",
    ast.Sub: "-",
    ast.UAdd: "+",
    ast.USub: "-",
}  # type: Dict[Type[ast.AST], str]


def parse(code: str, mode: str = 'exec') -> "ast.AST":
    """Parse the *code* using built-in ast or typed_ast.

    This enables "type_comments" feature if possible.
    """
    try:
        # type_comments parameter is available on py38+
        return ast.parse(code, mode=mode, type_comments=True)  # type: ignore
    except TypeError:
        # fallback to ast module.
        # typed_ast is used to parse type_comments if installed.
        return ast.parse(code, mode=mode)


def unparse(node: ast.AST) -> str:
    """
    Implement your code here
    """
    return None


def unparse_arguments(node: ast.arguments) -> str:
    """Unparse an arguments to string."""
    defaults = list(node.defaults)
    positionals = len(node.args)
    posonlyargs = 0
    if hasattr(node, "posonlyargs"):  # for py38+
        posonlyargs += len(node.posonlyargs)  # type:ignore
        positionals += posonlyargs
    for _ in range(len(defaults), positionals):
        defaults.insert(0, None)

    kw_defaults = list(node.kw_defaults)
    for _ in range(len(kw_defaults), len(node.kwonlyargs)):
        kw_defaults.insert(0, None)

    args = []  # type: List[str]
    if hasattr(node, "posonlyargs"):  # for py38+
        for i, arg in enumerate(node.posonlyargs):  # type: ignore
            name = unparse(arg)
            if defaults[i]:
                if arg.annotation:
                    name += " = %s" % unparse(defaults[i])
                else:
                    name += "=%s" % unparse(defaults[i])
            args.append(name)

        if node.posonlyargs:  # type: ignore
            args.append('/')

    for i, arg in enumerate(node.args):
        name = unparse(arg)
        if defaults[i + posonlyargs]:
            if arg.annotation:
                name += " = %s" % unparse(defaults[i + posonlyargs])
            else:
                name += "=%s" % unparse(defaults[i + posonlyargs])
        args.append(name)

    if node.vararg:
        args.append("*" + unparse(node.vararg))

    if node.kwonlyargs and not node.vararg:
        args.append('*')
    for i, arg in enumerate(node.kwonlyargs):
        name = unparse(arg)
        if kw_defaults[i]:
            if arg.annotation:
                name += " = %s" % unparse(kw_defaults[i])
            else:
                name += "=%s" % unparse(kw_defaults[i])
        args.append(name)

    if node.kwarg:
        args.append("**" + unparse(node.kwarg))

    return ", ".join(args)
