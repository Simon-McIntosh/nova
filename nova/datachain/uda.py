"""Manage access to ITER data via UDA."""
from dataclasses import dataclass, field

from uda_client_reader.uda_client_reader_python import UdaClientReaderPython


@dataclass
class UDA:
    """UDA base class."""

    host: str = "10.153.0.254"
    port: int = 3090
    client: UdaClientReaderPython = field(init=False, repr=False)

    def __post_init__(self):
        """Lanuch UDA client reader."""
        self.client = UdaClientReaderPython(self.host, self.port)
        print(self.client.getVariableList("*"))


if __name__ == "__main__":
    uda = UDA()
