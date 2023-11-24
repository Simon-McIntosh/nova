"""Define connection baseclass."""
from dataclasses import dataclass

from importlib import import_module


@dataclass
class Connect:
    """Methods to validate server connections."""

    hostname: str | tuple[str, int]

    def __call__(self):
        """Return True if connection to hostname is valid else False - extend."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return method name."""
        return type(self).__name__

    @property
    def _ssh(self):
        """Return True if connection to hostname is valid else False."""

    @property
    def mark(self):
        """Return connection pytest decorator."""
        return import_module("pytest").mark.skipif(
            not self(),
            reason=f"unable to connect via {self.name} to host {self.hostname}",
        )
