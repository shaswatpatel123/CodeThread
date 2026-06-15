import os
import signal
import subprocess

from django.core.files.temp import NamedTemporaryFile
from django.db.backends.base.client import BaseDatabaseClient


def _escape_pgpass(txt):
    """
    Escape a fragment of a PostgreSQL .pgpass file.
    """
    return txt.replace('\\', '\\\\').replace(':', '\\:')


class DatabaseClient(BaseDatabaseClient):
    executable_name = 'psql'

    @classmethod
    def runshell_db(cls, conn_params):
        """
        Implement your code here
        """
        return None

    def runshell(self):
        DatabaseClient.runshell_db(self.connection.get_connection_params())
