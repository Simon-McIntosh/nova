"""Manage connections to UDA servers."""
from dataclasses import dataclass

from nova.database.connect import Connect
from nova.database.connectssh import ConnectSSH
from nova.utilities.importmanager import check_import

with check_import("codac_uda"):
    from nova.datachain.uda import UdaInfo


@dataclass
class ConnectUDA(Connect):
    """Extend filepath.Connect. Methods to manage connection to UDA server."""

    def __call__(self):
        """Implement instance call to check SDCC then UDA connections."""
        if not ConnectSSH("sdcc-login02.iter.org")():  # test sdcc connection first
            return False
        try:
            uda = UdaInfo(self.hostname)
            return uda.client.isConnected()
        except ConnectionError:
            return False
