# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ['quantity_input']

import inspect
from ..utils.decorators import wraps
from ..utils.misc import isiterable

from .core import Unit, UnitsError, add_enabled_equivalencies
from .physical import _unit_physical_mapping


def _get_allowed_units(targets):
    """
    From a list of target units (either as strings or unit objects) and physical
    types, return a list of Unit objects.
    """

    allowed_units = []
    for target in targets:

        try:  # unit passed in as a string
            target_unit = Unit(target)

        except ValueError:

            try:  # See if the function writer specified a physical type
                physical_type_id = _unit_physical_mapping[target]

            except KeyError:  # Function argument target is invalid
                raise ValueError("Invalid unit or physical type '{0}'."
                                 .format(target))

            # get unit directly from physical type id
            target_unit = Unit._from_physical_type_id(physical_type_id)

        allowed_units.append(target_unit)

    return allowed_units


def _validate_arg_value(param_name, func_name, arg, targets, equivalencies):
    """
    Validates the object passed in to the wrapped function, ``arg``, with target
    unit or physical type, ``target``.
    """

    allowed_units = _get_allowed_units(targets)

    for allowed_unit in allowed_units:
        try:
            is_equivalent = arg.unit.is_equivalent(allowed_unit,
                                                   equivalencies=equivalencies)

            if is_equivalent:
                break

        except AttributeError:  # Either there is no .unit or no .is_equivalent
            if hasattr(arg, "unit"):
                error_msg = "a 'unit' attribute without an 'is_equivalent' method"
            else:
                error_msg = "no 'unit' attribute"

            raise TypeError("Argument '{0}' to function '{1}' has {2}. "
                  "You may want to pass in an astropy Quantity instead."
                     .format(param_name, func_name, error_msg))

    else:
        if len(targets) > 1:
            raise UnitsError("Argument '{0}' to function '{1}' must be in units"
                             " convertible to one of: {2}."
                             .format(param_name, func_name,
                                     [str(targ) for targ in targets]))
        else:
            raise UnitsError("Argument '{0}' to function '{1}' must be in units"
                             " convertible to '{2}'."
                             .format(param_name, func_name,
                                     str(targets[0])))


class QuantityInput:

    @classmethod
    def as_decorator(cls, func=None, **kwargs):
        r"""
        A decorator for validating the units of arguments to functions.

        Unit specifications can be provided as keyword arguments to the decorator,
        or by using function annotation syntax. Arguments to the decorator
        take precedence over any function annotations present.

        A `~astropy.units.UnitsError` will be raised if the unit attribute of
        the argument is not equivalent to the unit specified to the decorator
        or in the annotation.
        If the argument has no unit attribute, i.e. it is not a Quantity object, a
        `ValueError` will be raised.

        Where an equivalency is specified in the decorator, the function will be
        executed with that equivalency in force.

        Notes
        -----

        The checking of arguments inside variable arguments to a function is not
        supported (i.e. \*arg or \**kwargs).

        Examples
        --------

        .. code-block:: python

            import astropy.units as u
            @u.quantity_input(myangle=u.arcsec)
            def myfunction(myangle):
                return myangle**2


        .. code-block:: python

            import astropy.units as u
            @u.quantity_input
            def myfunction(myangle: u.arcsec):
                return myangle**2

        Also you can specify a return value annotation, which will
        cause the function to always return a `~astropy.units.Quantity` in that
        unit.

        .. code-block:: python

            import astropy.units as u
            @u.quantity_input
            def myfunction(myangle: u.arcsec) -> u.deg**2:
                return myangle**2

        Using equivalencies::

            import astropy.units as u
            @u.quantity_input(myenergy=u.eV, equivalencies=u.mass_energy())
            def myfunction(myenergy):
                return myenergy**2

        """
        self = cls(**kwargs)
        if func is not None and not kwargs:
            return self(func)
        else:
            return self

    def __init__(self, func=None, **kwargs):
        self.equivalencies = kwargs.pop('equivalencies', [])
        self.decorator_kwargs = kwargs

    def __call__(self, wrapped_function):
        """
        Implement your code here
        """
        return None


quantity_input = QuantityInput.as_decorator
