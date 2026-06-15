"""
Code to manage the creation and SQL rendering of 'where' constraints.
"""
import operator
from functools import reduce

from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.models.expressions import Case, When
from django.db.models.lookups import Exact
from django.utils import tree
from django.utils.functional import cached_property

# Connection types
AND = "AND"
OR = "OR"
XOR = "XOR"


class WhereNode(tree.Node):
    """
    An SQL WHERE clause.

    The class is tied to the Query class that created it (in order to create
    the correct SQL).

    A child is usually an expression producing boolean values. Most likely the
    expression is a Lookup instance.

    However, a child could also be any class with as_sql() and either
    relabeled_clone() method or relabel_aliases() and clone() methods and
    contains_aggregate attribute.
    """

    default = AND
    resolved = False
    conditional = True

    def split_having_qualify(self, negated=False, must_group_by=False):
        """
        Return three possibly None nodes: one for those parts of self that
        should be included in the WHERE clause, one for those parts of self
        that must be included in the HAVING clause, and one for those parts
        that refer to window functions.
        """
        if not self.contains_aggregate and not self.contains_over_clause:
            return self, None, None
        in_negated = negated ^ self.negated
        # Whether or not children must be connected in the same filtering
        # clause (WHERE > HAVING > QUALIFY) to maintain logical semantic.
        must_remain_connected = (
            (in_negated and self.connector == AND)
            or (not in_negated and self.connector == OR)
            or self.connector == XOR
        )
        if (
            must_remain_connected
            and self.contains_aggregate
            and not self.contains_over_clause
        ):
            # It's must cheaper to short-circuit and stash everything in the
            # HAVING clause than split children if possible.
            return None, self, None
        where_parts = []
        having_parts = []
        qualify_parts = []
        for c in self.children:
            if hasattr(c, "split_having_qualify"):
                where_part, having_part, qualify_part = c.split_having_qualify(
                    in_negated, must_group_by
                )
                if where_part is not None:
                    where_parts.append(where_part)
                if having_part is not None:
                    having_parts.append(having_part)
                if qualify_part is not None:
                    qualify_parts.append(qualify_part)
            elif c.contains_over_clause:
                qualify_parts.append(c)
            elif c.contains_aggregate:
                having_parts.append(c)
            else:
                where_parts.append(c)
        if must_remain_connected and qualify_parts:
            # Disjunctive heterogeneous predicates can be pushed down to
            # qualify as long as no conditional aggregation is involved.
            if not where_parts or (where_parts and not must_group_by):
                return None, None, self
            elif where_parts:
                # In theory this should only be enforced when dealing with
                # where_parts containing predicates against multi-valued
                # relationships that could affect aggregation results but this
                # is complex to infer properly.
                raise NotImplementedError(
                    "Heterogeneous disjunctive predicates against window functions are "
                    "not implemented when performing conditional aggregation."
                )
        where_node = (
            self.create(where_parts, self.connector, self.negated)
            if where_parts
            else None
        )
        having_node = (
            self.create(having_parts, self.connector, self.negated)
            if having_parts
            else None
        )
        qualify_node = (
            self.create(qualify_parts, self.connector, self.negated)
            if qualify_parts
            else None
        )
        return where_node, having_node, qualify_node

    def as_sql(self, compiler, connection):
        """
        Implement your code here
        """
        return None

    def get_group_by_cols(self):
        cols = []
        for child in self.children:
            cols.extend(child.get_group_by_cols())
        return cols

    def get_source_expressions(self):
        return self.children[:]

    def set_source_expressions(self, children):
        assert len(children) == len(self.children)
        self.children = children

    def relabel_aliases(self, change_map):
        """
        Relabel the alias values of any children. 'change_map' is a dictionary
        mapping old (current) alias values to the new values.
        """
        for pos, child in enumerate(self.children):
            if hasattr(child, "relabel_aliases"):
                # For example another WhereNode
                child.relabel_aliases(change_map)
            elif hasattr(child, "relabeled_clone"):
                self.children[pos] = child.relabeled_clone(change_map)

    def clone(self):
        clone = self.create(connector=self.connector, negated=self.negated)
        for child in self.children:
            if hasattr(child, "clone"):
                child = child.clone()
            clone.children.append(child)
        return clone

    def relabeled_clone(self, change_map):
        clone = self.clone()
        clone.relabel_aliases(change_map)
        return clone

    def replace_expressions(self, replacements):
        if replacement := replacements.get(self):
            return replacement
        clone = self.create(connector=self.connector, negated=self.negated)
        for child in self.children:
            clone.children.append(child.replace_expressions(replacements))
        return clone

    def get_refs(self):
        refs = set()
        for child in self.children:
            refs |= child.get_refs()
        return refs

    @classmethod
    def _contains_aggregate(cls, obj):
        if isinstance(obj, tree.Node):
            return any(cls._contains_aggregate(c) for c in obj.children)
        return obj.contains_aggregate

    @cached_property
    def contains_aggregate(self):
        return self._contains_aggregate(self)

    @classmethod
    def _contains_over_clause(cls, obj):
        if isinstance(obj, tree.Node):
            return any(cls._contains_over_clause(c) for c in obj.children)
        return obj.contains_over_clause

    @cached_property
    def contains_over_clause(self):
        return self._contains_over_clause(self)

    @property
    def is_summary(self):
        return any(child.is_summary for child in self.children)

    @staticmethod
    def _resolve_leaf(expr, query, *args, **kwargs):
        if hasattr(expr, "resolve_expression"):
            expr = expr.resolve_expression(query, *args, **kwargs)
        return expr

    @classmethod
    def _resolve_node(cls, node, query, *args, **kwargs):
        if hasattr(node, "children"):
            for child in node.children:
                cls._resolve_node(child, query, *args, **kwargs)
        if hasattr(node, "lhs"):
            node.lhs = cls._resolve_leaf(node.lhs, query, *args, **kwargs)
        if hasattr(node, "rhs"):
            node.rhs = cls._resolve_leaf(node.rhs, query, *args, **kwargs)

    def resolve_expression(self, *args, **kwargs):
        clone = self.clone()
        clone._resolve_node(clone, *args, **kwargs)
        clone.resolved = True
        return clone

    @cached_property
    def output_field(self):
        from django.db.models import BooleanField

        return BooleanField()

    @property
    def _output_field_or_none(self):
        return self.output_field

    def select_format(self, compiler, sql, params):
        # Wrap filters with a CASE WHEN expression if a database backend
        # (e.g. Oracle) doesn't support boolean expression in SELECT or GROUP
        # BY list.
        if not compiler.connection.features.supports_boolean_expr_in_select_clause:
            sql = f"CASE WHEN {sql} THEN 1 ELSE 0 END"
        return sql, params

    def get_db_converters(self, connection):
        return self.output_field.get_db_converters(connection)

    def get_lookup(self, lookup):
        return self.output_field.get_lookup(lookup)

    def leaves(self):
        for child in self.children:
            if isinstance(child, WhereNode):
                yield from child.leaves()
            else:
                yield child


class NothingNode:
    """A node that matches nothing."""

    contains_aggregate = False
    contains_over_clause = False

    def as_sql(self, compiler=None, connection=None):
        raise EmptyResultSet


class ExtraWhere:
    # The contents are a black box - assume no aggregates or windows are used.
    contains_aggregate = False
    contains_over_clause = False

    def __init__(self, sqls, params):
        self.sqls = sqls
        self.params = params

    def as_sql(self, compiler=None, connection=None):
        sqls = ["(%s)" % sql for sql in self.sqls]
        return " AND ".join(sqls), list(self.params or ())


class SubqueryConstraint:
    # Even if aggregates or windows would be used in a subquery,
    # the outer query isn't interested about those.
    contains_aggregate = False
    contains_over_clause = False

    def __init__(self, alias, columns, targets, query_object):
        self.alias = alias
        self.columns = columns
        self.targets = targets
        query_object.clear_ordering(clear_default=True)
        self.query_object = query_object

    def as_sql(self, compiler, connection):
        query = self.query_object
        query.set_values(self.targets)
        query_compiler = query.get_compiler(connection=connection)
        return query_compiler.as_subquery_condition(self.alias, self.columns, compiler)
