"""Manage TFC fiducial data for coil and sector allignment."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas

from nova.assembly.fiducialccl import Fiducial, FiducialRE, FiducialIDM
from nova.assembly.sectordata import SectorData


@dataclass
class FiducialSector(Fiducial):
    """Manage Reverse Engineering fiducial data."""

    phase: str = "FAT supplier"
    sector: dict[int, int] = field(init=False, repr=False, default_factory=dict)
    sectors: dict[int, list] = field(
        init=False, repr=False, default_factory=lambda: dict.fromkeys(range(1, 9), [])
    )
    fiducial: dict[str, pandas.DataFrame] | dict = field(
        init=False, repr=False, default_factory=dict
    )
    variance: dict[str, pandas.DataFrame] | dict = field(
        init=False, repr=False, default_factory=dict
    )

    sheets: ClassVar[dict[str, str]] = {"FATsup": "FAT supplier", "SSAT": "SSAT BR"}

    def __post_init__(self):
        """Propogate origin."""
        self._set_phase()
        super().__post_init__()
        self.source = "Reverse Engineering IDM datasets (xls workbooks)"
        self.origin = [self.origin[coil - 1] for coil in self.delta]
        self._load_fiducials()
        self._load_variance()

    def _set_phase(self):
        """Expand short string phase label."""
        self.phase = self.sheets.get(self.phase, self.phase)

    def _load_deltas(self):
        """Implement load deltas abstractmethod."""
        columns = ["dx", "dy", "dz"]
        for sector in self.sectors:
            data = SectorData(sector)
            self.sectors[sector] = data.coil
            for coil, ccl in data.ccl[self.phase].items():
                self.sector[coil] = sector
                self.delta[coil] = ccl.loc[self.target, columns]

    def _load_fiducials(self):
        """Load unique fiducial targets."""
        columns = ["x", "y", "z"]
        for sector in self.sectors:
            data = SectorData(sector)
            for coil, fiducial in data.data.items():
                nominal = fiducial["Nominal"]
                nominal.index = nominal.index.droplevel([0, 1])
                nominal.rename(index={"F'": "F"}, inplace=True)
                self.fiducial[coil] = nominal.loc[self.target, columns]

    def _load_variance(self):
        columns = ["ux", "uy", "uz"]
        for sector in self.sectors:
            data = SectorData(sector)
            for coil, ccl in data.ccl[self.phase].items():
                two_sigma = ccl.loc[self.target, columns]
                self.variance[coil] = (two_sigma / 2) ** 2
                self.variance[coil] = self.variance[coil].rename(
                    columns={col: f"s2{col[-1]}" for col in columns}
                )

    def compare(self, source="RE"):
        """Compare fiducial sector data with previous RE dataset."""
        match source:
            case "IDM":
                previous = FiducialIDM()
            case "RE":
                previous = FiducialRE()
            case _:
                raise ValueError(f"source {source} not in [RE, IDM]")

        for coil, ccl in self.delta.items():
            if coil not in previous.delta:
                continue
            _ccl = previous.delta[coil]
            if source == "RE":
                _ccl = _ccl = previous.delta[coil].xs("FAT", 1)
            change = ccl.loc[:, ["dx", "dy", "dz"]] - _ccl
            if not np.allclose(np.array(change, float), 0):
                print(f"\ncoil #{coil}")
                print(change)


if __name__ == "__main__":
    fiducial = FiducialSector(phase="FATsup")  # , sectors=[8]
    # fiducial.compare("IDM")
    # fiducial.plot()

    # for coil, ccl in fiducial.delta.items():
    #    print(f"Coil {coil}")
    #    print(ccl)
    #    print()
