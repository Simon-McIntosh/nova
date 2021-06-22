"""Manage Ansys simulation data directories."""
from dataclasses import dataclass, field
import os

from nova.definitions import root_dir


@dataclass
class AnsysDataDir:
    """Manage file paths."""

    file: str
    subset: str = 'all'
    directory: str = field(repr=False, default=None)

    def __post_init__(self):
        """Set data directory."""
        if self.directory is None:
            self.directory = os.path.join(root_dir, 'data/Ansys/TFC18')

    @property
    def metadata(self):
        """Return file metadata."""
        return self.file, self.name, self.directory
