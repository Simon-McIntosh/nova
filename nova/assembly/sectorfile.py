"""Query sector metrology datasets."""
from dataclasses import dataclass
from functools import cached_property
from glob import glob
from pathlib import Path

import numpy as np


@dataclass
class SectorFile:
    r"""
    Sector filename path and group.

    Sector module data is downloaded from IDM and stored in IO shared folder at:
    \\\\io-ws-ccstore1\\ANSYS_Data\\mcintos\\sector_modules

    datadir : str
        Data directory. Set as mount point location to access IO shared folder
    """

    sector: int
    filename: str = ""
    version: str | int = "latest"
    datadir: str = "/mnt/share/sector_modules"

    def __post_init__(self):
        """Locate source datafiles."""
        self.locate_file()

    def locate_file(self):
        """Locate remote datafile and set version and set version number."""
        match self.filename:
            case "":
                self._locate_file()
            case str():
                self._update_metadata()

    def _update_metadata(self):
        """Update sector and version number from filename."""
        self.version = self._get_version(self.filename)
        self.sector = self._get_sector(self.filename)

    @staticmethod
    def _get_version(filename: str) -> float:
        """Return filename version."""
        return float(filename.split("v")[-1].split(".")[0].replace("_", "."))

    @staticmethod
    def _get_sector(filename: str) -> int:
        """Return sector index."""
        return int(filename.split("#")[-1].split("_")[0])

    @cached_property
    def filenames(self) -> list[str]:
        """Return filename list."""
        return [
            Path(path).with_suffix("").name
            for path in glob(self.datadir + f"/Sector_Module_#{self.sector}*.xlsx")
        ]

    @cached_property
    def versions(self) -> list[float]:
        """Return version list."""
        versions = [self._get_version(filename) for filename in self.filenames]
        if len(versions) == 0:
            raise FileNotFoundError(
                f"Unable to locate RE data files at {self.datadir}. "
                "Check contents of data directory. "
                "Confirm connection to the IO network if datadir is a mounted share."
            )
        return versions

    def _locate_file(self):
        """Locate source datafile and update filename and version."""
        match self.version:
            case "latest":
                index = np.argmax(self.versions)
            case int() | float():
                index = self.versions.index(self.version)
            case _:
                raise ValueError(
                    f"version {self.version} not latest or in {self.versions}"
                )
        self.filename = self.filenames[index]
        self.version = self.versions[index]
