"""Manage connections to SSH servers."""

from dataclasses import dataclass

from nova.database.connect import Connect
from nova.utilities.importmanager import check_import

with check_import("ssh"):
    import paramiko


@dataclass
class ConnectSSH(Connect):
    """Extend filepath.Connect. Methods to manage connection to SSH server."""

    def __call__(self):
        """Implement instance call to check UDA connection."""
        try:
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.connect(self.hostname)
            return True
        except (NameError, paramiko.ssh_exception.socket.gaierror):
            return False
