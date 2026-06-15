import keyword
import re

from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, connections
from django.db.models.constants import LOOKUP_SEP


class Command(BaseCommand):
    help = "Introspects the database tables in the given database and outputs a Django model module."
    requires_system_checks = []
    stealth_options = ('table_name_filter',)
    db_module = 'django.db'

    def add_arguments(self, parser):
        parser.add_argument(
            'table', nargs='*', type=str,
            help='Selects what tables or views should be introspected.',
        )
        parser.add_argument(
            '--database', default=DEFAULT_DB_ALIAS,
            help='Nominates a database to introspect. Defaults to using the "default" database.',
        )
        parser.add_argument(
            '--include-partitions', action='store_true', help='Also output models for partition tables.',
        )
        parser.add_argument(
            '--include-views', action='store_true', help='Also output models for database views.',
        )

    def handle(self, **options):
        try:
            for line in self.handle_inspection(options):
                self.stdout.write(line)
        except NotImplementedError:
            raise CommandError("Database inspection isn't supported for the currently selected database backend.")

    def handle_inspection(self, options):
        """
        Implement your code here
        """
        return None

    def normalize_col_name(self, col_name, used_column_names, is_relation):
        """
        Modify the column name to make it Python-compatible as a field name
        """
        field_params = {}
        field_notes = []

        new_name = col_name.lower()
        if new_name != col_name:
            field_notes.append('Field name made lowercase.')

        if is_relation:
            if new_name.endswith('_id'):
                new_name = new_name[:-3]
            else:
                field_params['db_column'] = col_name

        new_name, num_repl = re.subn(r'\W', '_', new_name)
        if num_repl > 0:
            field_notes.append('Field renamed to remove unsuitable characters.')

        if new_name.find(LOOKUP_SEP) >= 0:
            while new_name.find(LOOKUP_SEP) >= 0:
                new_name = new_name.replace(LOOKUP_SEP, '_')
            if col_name.lower().find(LOOKUP_SEP) >= 0:
                # Only add the comment if the double underscore was in the original name
                field_notes.append("Field renamed because it contained more than one '_' in a row.")

        if new_name.startswith('_'):
            new_name = 'field%s' % new_name
            field_notes.append("Field renamed because it started with '_'.")

        if new_name.endswith('_'):
            new_name = '%sfield' % new_name
            field_notes.append("Field renamed because it ended with '_'.")

        if keyword.iskeyword(new_name):
            new_name += '_field'
            field_notes.append('Field renamed because it was a Python reserved word.')

        if new_name[0].isdigit():
            new_name = 'number_%s' % new_name
            field_notes.append("Field renamed because it wasn't a valid Python identifier.")

        if new_name in used_column_names:
            num = 0
            while '%s_%d' % (new_name, num) in used_column_names:
                num += 1
            new_name = '%s_%d' % (new_name, num)
            field_notes.append('Field renamed because of name conflict.')

        if col_name != new_name and field_notes:
            field_params['db_column'] = col_name

        return new_name, field_params, field_notes

    def get_field_type(self, connection, table_name, row):
        """
        Given the database connection, the table name, and the cursor row
        description, this routine will return the given field type name, as
        well as any additional keyword parameters and notes for the field.
        """
        field_params = {}
        field_notes = []

        try:
            field_type = connection.introspection.get_field_type(row.type_code, row)
        except KeyError:
            field_type = 'TextField'
            field_notes.append('This field type is a guess.')

        # Add max_length for all CharFields.
        if field_type == 'CharField' and row.internal_size:
            field_params['max_length'] = int(row.internal_size)

        if field_type in {'CharField', 'TextField'} and row.collation:
            field_params['db_collation'] = row.collation

        if field_type == 'DecimalField':
            if row.precision is None or row.scale is None:
                field_notes.append(
                    'max_digits and decimal_places have been guessed, as this '
                    'database handles decimal fields as float')
                field_params['max_digits'] = row.precision if row.precision is not None else 10
                field_params['decimal_places'] = row.scale if row.scale is not None else 5
            else:
                field_params['max_digits'] = row.precision
                field_params['decimal_places'] = row.scale

        return field_type, field_params, field_notes

    def get_meta(self, table_name, constraints, column_to_field_name, is_view, is_partition):
        """
        Return a sequence comprising the lines of code necessary
        to construct the inner Meta class for the model corresponding
        to the given database table name.
        """
        unique_together = []
        has_unsupported_constraint = False
        for params in constraints.values():
            if params['unique']:
                columns = params['columns']
                if None in columns:
                    has_unsupported_constraint = True
                columns = [x for x in columns if x is not None]
                if len(columns) > 1:
                    unique_together.append(str(tuple(column_to_field_name[c] for c in columns)))
        if is_view:
            managed_comment = "  # Created from a view. Don't remove."
        elif is_partition:
            managed_comment = "  # Created from a partition. Don't remove."
        else:
            managed_comment = ''
        meta = ['']
        if has_unsupported_constraint:
            meta.append('    # A unique constraint could not be introspected.')
        meta += [
            '    class Meta:',
            '        managed = False%s' % managed_comment,
            '        db_table = %r' % table_name
        ]
        if unique_together:
            tup = '(' + ', '.join(unique_together) + ',)'
            meta += ["        unique_together = %s" % tup]
        return meta
