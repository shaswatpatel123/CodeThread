from importlib import import_module

from django.apps import apps
from django.core.management.base import BaseCommand, CommandError
from django.core.management.color import no_style
from django.core.management.sql import emit_post_migrate_signal, sql_flush
from django.db import DEFAULT_DB_ALIAS, connections


class Command(BaseCommand):
    help = (
        'Removes ALL DATA from the database, including data added during '
        'migrations. Does not achieve a "fresh install" state.'
    )
    stealth_options = ('reset_sequences', 'allow_cascade', 'inhibit_post_migrate')

    def add_arguments(self, parser):
        parser.add_argument(
            '--noinput', '--no-input', action='store_false', dest='interactive',
            help='Tells Django to NOT prompt the user for input of any kind.',
        )
        parser.add_argument(
            '--database', default=DEFAULT_DB_ALIAS,
            help='Nominates a database to flush. Defaults to the "default" database.',
        )

    def handle(self, **options):
        """
        Implement your code here
        """
        return None
