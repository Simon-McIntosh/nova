"""Manage Ansys simulation data directories."""
from dataclasses import dataclass, field
import os

from nova.definitions import root_dir


@dataclass
class AnsysDataDir:
    """Manage file paths."""

    folder: str
    file: str
    subset: str = 'all'
    data_dir: str = 'data/Ansys'

    def __post_init__(self):
        """Set data directory."""
        self.directory = os.path.join(root_dir, self.data_dir, self.folder)

    @property
    def metadata(self):
        """Return file metadata."""
        return self.file, self.name, self.directory
