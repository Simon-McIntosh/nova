"""Manage connections to UDA servers."""
from dataclasses import dataclass

from nova.database.connect import Connect
from nova.utilities.importmanager import check_import

with check_import("codac_uda"):
    from nova.datachain.uda import UdaInfo


@dataclass
class ConnectUDA(Connect):
    """Extend filepath.Connect. Methods to manage connection to UDA server."""

    def __call__(self):
        """Implement instance call to check UDA connection."""
        try:
            UdaInfo(self.hostname)
            return True
        except ConnectionError:
            return False
