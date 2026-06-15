import os
import signal
import subprocess

from django.db.backends.base.client import BaseDatabaseClient


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
