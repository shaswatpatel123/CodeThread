# Licensed under a 3-clause BSD style license - see LICNSE.rst

# This module includes files automatically generated from ply (these end in
# _lextab.py and _parsetab.py). To generate these files, remove them from this
# folder, then build astropy and run the tests in-place:
#
#   python setup.py build_ext --inplace
#   pytest astropy/units
#
# You can then commit the changes to the re-generated _lextab.py and
# _parsetab.py files.

"""Handles the CDS string format for units."""

import operator
import re

from astropy.units.utils import is_effectively_unity
from astropy.utils import classproperty, parsing
from astropy.utils.misc import did_you_mean

from . import core, utils
from .base import Base


class CDS(Base):
    """
    Support the `Centre de Données astronomiques de Strasbourg
    <http://cds.u-strasbg.fr/>`_ `Standards for Astronomical
    Catalogues 2.0 <http://vizier.u-strasbg.fr/vizier/doc/catstd-3.2.htx>`_
    format, and the `complete set of supported units
    <https://vizier.u-strasbg.fr/viz-bin/Unit>`_.  This format is used
    by VOTable up to version 1.2.
    """

    _tokens = (
        "PRODUCT",
        "DIVISION",
        "OPEN_PAREN",
        "CLOSE_PAREN",
        "OPEN_BRACKET",
        "CLOSE_BRACKET",
        "X",
        "SIGN",
        "UINT",
        "UFLOAT",
        "UNIT",
        "DIMENSIONLESS",
    )

    @classproperty(lazy=True)
    def _units(cls):
        return cls._generate_unit_names()

    @classproperty(lazy=True)
    def _parser(cls):
        return cls._make_parser()

    @classproperty(lazy=True)
    def _lexer(cls):
        return cls._make_lexer()

    @staticmethod
    def _generate_unit_names():
        from astropy import units as u
        from astropy.units import cds

        names = {}

        for key, val in cds.__dict__.items():
            if isinstance(val, u.UnitBase):
                names[key] = val

        return names

    @classmethod
    def _make_lexer(cls):
        tokens = cls._tokens

        t_PRODUCT = r"\."
        t_DIVISION = r"/"
        t_OPEN_PAREN = r"\("
        t_CLOSE_PAREN = r"\)"
        t_OPEN_BRACKET = r"\["
        t_CLOSE_BRACKET = r"\]"

        # NOTE THE ORDERING OF THESE RULES IS IMPORTANT!!
        # Regular expression rules for simple tokens

        def t_UFLOAT(t):
            r"((\d+\.?\d+)|(\.\d+))([eE][+-]?\d+)?"
            if not re.search(r"[eE\.]", t.value):
                t.type = "UINT"
                t.value = int(t.value)
            else:
                t.value = float(t.value)
            return t

        def t_UINT(t):
            r"\d+"
            t.value = int(t.value)
            return t

        def t_SIGN(t):
            r"[+-](?=\d)"
            t.value = float(t.value + "1")
            return t

        def t_X(t):  # multiplication for factor in front of unit
            r"[x×]"
            return t

        def t_UNIT(t):
            r"\%|°|\\h|((?!\d)\w)+"
            t.value = cls._get_unit(t)
            return t

        def t_DIMENSIONLESS(t):
            r"---|-"
            # These are separate from t_UNIT since they cannot have a prefactor.
            t.value = cls._get_unit(t)
            return t

        t_ignore = ""

        # Error handling rule
        def t_error(t):
            raise ValueError(f"Invalid character at col {t.lexpos}")

        return parsing.lex(
            lextab="cds_lextab", package="astropy/units", reflags=int(re.UNICODE)
        )

    @classmethod
    def _make_parser(cls):
        """
        Implement your code here
        """
        return None

    @classmethod
    def _get_unit(cls, t):
        try:
            return cls._parse_unit(t.value)
        except ValueError as e:
            registry = core.get_current_unit_registry()
            if t.value in registry.aliases:
                return registry.aliases[t.value]

            raise ValueError(f"At col {t.lexpos}, {str(e)}")

    @classmethod
    def _parse_unit(cls, unit, detailed_exception=True):
        if unit not in cls._units:
            if detailed_exception:
                raise ValueError(
                    "Unit '{}' not supported by the CDS SAC standard. {}".format(
                        unit, did_you_mean(unit, cls._units)
                    )
                )
            else:
                raise ValueError()

        return cls._units[unit]

    @classmethod
    def parse(cls, s, debug=False):
        if " " in s:
            raise ValueError("CDS unit must not contain whitespace")

        if not isinstance(s, str):
            s = s.decode("ascii")

        # This is a short circuit for the case where the string
        # is just a single unit name
        try:
            return cls._parse_unit(s, detailed_exception=False)
        except ValueError:
            try:
                return cls._parser.parse(s, lexer=cls._lexer, debug=debug)
            except ValueError as e:
                if str(e):
                    raise ValueError(str(e))
                else:
                    raise ValueError("Syntax error")

    @staticmethod
    def _get_unit_name(unit):
        return unit.get_format_name("cds")

    @classmethod
    def _format_unit_list(cls, units):
        out = []
        for base, power in units:
            if power == 1:
                out.append(cls._get_unit_name(base))
            else:
                out.append(f"{cls._get_unit_name(base)}{int(power)}")
        return ".".join(out)

    @classmethod
    def to_string(cls, unit):
        # Remove units that aren't known to the format
        unit = utils.decompose_to_known_units(unit, cls._get_unit_name)

        if isinstance(unit, core.CompositeUnit):
            if unit == core.dimensionless_unscaled:
                return "---"
            elif is_effectively_unity(unit.scale * 100.0):
                return "%"

            if unit.scale == 1:
                s = ""
            else:
                m, e = utils.split_mantissa_exponent(unit.scale)
                parts = []
                if m not in ("", "1"):
                    parts.append(m)
                if e:
                    if not e.startswith("-"):
                        e = "+" + e
                    parts.append(f"10{e}")
                s = "x".join(parts)

            pairs = list(zip(unit.bases, unit.powers))
            if len(pairs) > 0:
                pairs.sort(key=operator.itemgetter(1), reverse=True)

                s += cls._format_unit_list(pairs)

        elif isinstance(unit, core.NamedUnit):
            s = cls._get_unit_name(unit)

        return s
