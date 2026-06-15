from itertools import chain
from types import MethodType

from django.apps import apps
from django.conf import settings
from django.core import checks

from .management import _get_builtin_permissions


def check_user_model(app_configs=None, **kwargs):
    """
    Implement your code here
    """
    return None


def check_models_permissions(app_configs=None, **kwargs):
    if app_configs is None:
        models = apps.get_models()
    else:
        models = chain.from_iterable(app_config.get_models() for app_config in app_configs)

    Permission = apps.get_model('auth', 'Permission')
    permission_name_max_length = Permission._meta.get_field('name').max_length
    permission_codename_max_length = Permission._meta.get_field('codename').max_length
    errors = []

    for model in models:
        opts = model._meta
        builtin_permissions = dict(_get_builtin_permissions(opts))
        # Check builtin permission name length.
        max_builtin_permission_name_length = (
            max(len(name) for name in builtin_permissions.values())
            if builtin_permissions else 0
        )
        if max_builtin_permission_name_length > permission_name_max_length:
            verbose_name_max_length = (
                permission_name_max_length - (max_builtin_permission_name_length - len(opts.verbose_name_raw))
            )
            errors.append(
                checks.Error(
                    "The verbose_name of model '%s' must be at most %d "
                    "characters for its builtin permission names to be at "
                    "most %d characters." % (
                        opts.label, verbose_name_max_length, permission_name_max_length
                    ),
                    obj=model,
                    id='auth.E007',
                )
            )
        # Check builtin permission codename length.
        max_builtin_permission_codename_length = (
            max(len(codename) for codename in builtin_permissions.keys())
            if builtin_permissions else 0
        )
        if max_builtin_permission_codename_length > permission_codename_max_length:
            model_name_max_length = permission_codename_max_length - (
                max_builtin_permission_codename_length - len(opts.model_name)
            )
            errors.append(
                checks.Error(
                    "The name of model '%s' must be at most %d characters "
                    "for its builtin permission codenames to be at most %d "
                    "characters." % (
                        opts.label,
                        model_name_max_length,
                        permission_codename_max_length,
                    ),
                    obj=model,
                    id='auth.E011',
                )
            )
        codenames = set()
        for codename, name in opts.permissions:
            # Check custom permission name length.
            if len(name) > permission_name_max_length:
                errors.append(
                    checks.Error(
                        "The permission named '%s' of model '%s' is longer "
                        "than %d characters." % (
                            name, opts.label, permission_name_max_length,
                        ),
                        obj=model,
                        id='auth.E008',
                    )
                )
            # Check custom permission codename length.
            if len(codename) > permission_codename_max_length:
                errors.append(
                    checks.Error(
                        "The permission codenamed '%s' of model '%s' is "
                        "longer than %d characters." % (
                            codename,
                            opts.label,
                            permission_codename_max_length,
                        ),
                        obj=model,
                        id='auth.E012',
                    )
                )
            # Check custom permissions codename clashing.
            if codename in builtin_permissions:
                errors.append(
                    checks.Error(
                        "The permission codenamed '%s' clashes with a builtin permission "
                        "for model '%s'." % (codename, opts.label),
                        obj=model,
                        id='auth.E005',
                    )
                )
            elif codename in codenames:
                errors.append(
                    checks.Error(
                        "The permission codenamed '%s' is duplicated for "
                        "model '%s'." % (codename, opts.label),
                        obj=model,
                        id='auth.E006',
                    )
                )
            codenames.add(codename)

    return errors
