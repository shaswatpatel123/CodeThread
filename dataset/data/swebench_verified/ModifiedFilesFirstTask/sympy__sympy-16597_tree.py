from __future__ import print_function, division


def pprint_nodes(subtrees):
    """
    Prettyprints systems of nodes.

    Examples
    ========

    >>> from sympy.printing.tree import pprint_nodes
    >>> print(pprint_nodes(["a", "b1\\nb2", "c"]))
    +-a
    +-b1
    | b2
    +-c

    """
    def indent(s, type=1):
        x = s.split("\n")
        r = "+-%s\n" % x[0]
        for a in x[1:]:
            if a == "":
                continue
            if type == 1:
                r += "| %s\n" % a
            else:
                r += "  %s\n" % a
        return r
    if not subtrees:
        return ""
    f = ""
    for a in subtrees[:-1]:
        f += indent(a)
    f += indent(subtrees[-1], 2)
    return f


def print_node(node):
    """
    Returns information about the "node".

    This includes class name, string representation and assumptions.
    """
    s = "%s: %s\n" % (node.__class__.__name__, str(node))
    d = node._assumptions
    if d:
        for a in sorted(d):
            v = d[a]
            if v is None:
                continue
            s += "%s: %s\n" % (a, v)
    return s


def tree(node):
    """
    Returns a tree representation of "node" as a string.

    It uses print_node() together with pprint_nodes() on node.args recursively.

    See Also
    ========

    print_tree

    """
    subtrees = []
    for arg in node.args:
        subtrees.append(tree(arg))
    s = print_node(node) + pprint_nodes(subtrees)
    return s


def print_tree(node):
    """
    Implement your code here
    """
    return None
