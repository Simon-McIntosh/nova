"""Manage TFC fiducial data for coil and sector allignment."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
import os
from typing import ClassVar

import openpyxl
import pandas

from nova.assembly.fiducialccl import Fiducial, FiducialRE
from nova.definitions import root_dir


@dataclass
class SectorData:
    """Manage fiducial coil and sector assembly data sourced from IDM."""

    file: str
    data: dict = field(init=False, repr=False, default_factory=dict)
    ccl: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        """Build mesurment dataset."""
        self.build_data()
        self.build_ccl()

    @property
    def xls_file(self):
        """Return xls filename."""
        return os.path.join(root_dir, "input/ITER", f"{self.file}.xlsx")

    def _initialize_data(self):
        """Initialize data as a bare nested dict with coil name entries."""
        self.data = {name: {} for name in self._coil_names("Nominal")}

    def build_data(self):
        """Build dataset."""
        with self.openbook():
            self._initialize_data()
            for worksheet in self.book:
                sheet = worksheet.title
                if sheet == "Metadata":
                    continue
                for index, name in enumerate(self._coil_names(sheet)):
                    self.data[name][sheet] = self.read_frame(index, sheet)

    @cached_property
    def coil(self) -> list[str, str]:
        """Return list of coil names."""
        return [name for name in self.data]

    @cached_property
    def phase(self) -> list[str, ...]:
        """Return list of assembly phases."""
        return [phase for phase in self.data[self.coil[0]] if phase != "Nominal"]

    def _initalize_ccl(self):
        """Init ccl as bare nested dict with assembly stage entries."""
        self.ccl = {phase: {coil: None for coil in self.coil} for phase in self.phase}

    def build_ccl(self):
        """Build ccl data."""
        self._initalize_ccl()
        for coil in self.coil:
            nominal = self.data[coil]["Nominal"]
            for phase in self.phase:
                self.ccl[phase][coil] = self.data[coil][phase].loc[nominal.index]
                try:
                    self.ccl[phase][coil].loc[:, nominal.columns] -= nominal
                except TypeError:
                    self.ccl[phase][coil].loc[:, nominal.columns] = 0.0
                self.ccl[phase][coil] = self.ccl[phase][coil].rename(
                    columns={col: f"d{col}" for col in "xyz"}
                )
                self.ccl[phase][coil] = self.ccl[phase][coil].rename(index={"F'": "F"})
                self.ccl[phase][coil].index = self.ccl[phase][coil].index.droplevel(
                    [0, 1]
                )
                self.ccl[phase][coil].index.name = None

    @contextmanager
    def openbook(self):
        """Manage access to source workbook."""
        self.book = openpyxl.load_workbook(self.xls_file, data_only=True)
        yield
        self.book.close()

    def locate(self, item: str, sheet: str):
        """Return item row/column locations in worksheet."""
        index = []
        for col in self.book[sheet].iter_cols():
            for cell in col:
                if cell.value == item:
                    index.append((cell.row, cell.column))
                    break
        assert len(index) == 2
        return index

    def _coil_index(self, sheet: str):
        """Return list dataset origins."""
        return self.locate("Coil", sheet)

    def _coil_names(self, sheet: str):
        """Return list of coil names."""
        name = []
        for row, cell in self._coil_index(sheet):
            name.append(self.book[sheet].cell(row + 1, cell).value)
        return name

    def _column_number(self, index, sheet: str):
        """Return column number."""
        for ncol, cell in enumerate(
            self.book[sheet].iter_cols(
                min_row=index[0], max_row=index[0], min_col=index[1]
            )
        ):
            if cell[0].value is None:
                ncol -= 1
                break
        return ncol + 1

    def read_frame(self, coil: int, sheet: str):
        """Return pandas dataframe from indexed sheet."""
        index = self._coil_index(sheet)[coil]
        ncol = self._column_number(index, sheet)
        usecols = list(range(index[1] - 1, index[1] - 1 + ncol))
        data = pandas.read_excel(
            self.xls_file,
            sheet_name=sheet,
            skiprows=index[0] - 1,
            usecols=usecols,
            index_col=[0, 1, 2],
            keep_default_na=False,
        )
        data = data.rename(
            columns={col: col.split(".")[0].lower() for col in data.columns}
        )
        data.index.rename(
            [name.split(".")[0] for name in data.index.names], inplace=True
        )
        return data


@dataclass
class FiducialSector(Fiducial):
    """Manage Reverse Engineering fiducial data."""

    phase: str = "FAT supplier"
    variance: dict[str, pandas.DataFrame] | dict = field(
        init=False, default_factory=dict
    )

    sectors: ClassVar[list[str]] = [
        "Sector_Module_#6_CCL_as-built_data_8NQVKS_v2_1",
        "Sector_Module_#7_CCL_as-built_data_8NR9J7_v2_1",
    ]

    def __post_init__(self):
        """Propogate origin."""
        super().__post_init__()
        self.source = "Reverse Engineering IDM datasets (xls workbooks)"
        self.origin = [
            origin for i, origin in enumerate(self.origin) if i + 1 in self.delta
        ]
        self._load_variance()

    def _load_deltas(self):
        """Implement load deltas abstractmethod."""
        columns = ["dx", "dy", "dz"]
        for sector in self.sectors:
            data = SectorData(sector)
            for coil, ccl in data.ccl[self.phase].items():
                self.delta[coil] = ccl.loc[self.target, columns]

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

    def compare(self):
        """Compare fiducial sector data with previous RE dataset."""
        previous = FiducialRE()
        for coil, ccl in self.delta.items():
            print(coil)
            _ccl = previous.delta[coil].xs("FAT", 1)
            print(ccl.loc[:, ["dx", "dy", "dz"]] - _ccl)
            print("\n")


if __name__ == "__main__":
    sector = SectorData("Sector_Module_#7_CCL_as-built_data_8NR9J7_v2_1")

    fiducial = FiducialSector(phase="SSAT BR")
    # fiducial.compare()

    for coil, ccl in fiducial.delta.items():
        print(f"Coil {coil}")
        print(ccl)
        print()
