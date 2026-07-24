"""Load magnetic diagnostic geometries and save to IDS."""

from functools import cached_property
from dataclasses import dataclass, field
import glob
import itertools
from pathlib import Path

import numpy as np


@dataclass
class Magnetics:
    """Read 3D magnetic diagnostic loop data."""

    datadir: Path = field(
        default_factory=lambda: Path("//io-ws-ccstore1/ANSYS_Data/mcintos/magnetics")
    )

    @property
    def _magnetics_xlsx(self) -> Path:
        """Return the magnetics positions workbook path."""
        return (
            self.datadir / "List_of_Current_Magnetic_Coil_Positions_24V7KU_v2_11.xlsx"
        )

    @cached_property
    def loops(self) -> list[str]:
        """Return the loop names from the first column of the Loops sheet."""
        import openpyxl

        workbook = openpyxl.load_workbook(
            self._magnetics_xlsx, read_only=True, data_only=True
        )
        try:
            sheet = workbook["Loops"]
            names = [
                row[0]
                for row in sheet.iter_rows(min_row=2, max_col=1, values_only=True)
                if row[0] is not None
            ]
        finally:
            workbook.close()
        return names

    @cached_property
    def _loop_names(self):
        """Return cached grouped loop diagnostic functional part number generator."""
        return {
            key: list(group)
            for key, group in itertools.groupby(
                self.loops, lambda name: name.split(".")[1]
            )
        }

    def loop_names(self, group: str):
        """Return loop name list for diagnostic group."""
        return self._loop_names[group]

    def _read_text_file(self, filepath: str) -> list[np.ndarray]:
        """Return the whitespace/comma separated xyz blocks of a loop file."""
        data = []
        with open(filepath, "r") as file:
            for block in (
                line
                for newline, line in itertools.groupby(
                    file, lambda line: line[0] == "\n"
                )
                if not newline
            ):
                data.append(np.atleast_2d(np.genfromtxt(block, delimiter=",")))
        return data

    @cached_property
    def _loop_files(self) -> list[str]:
        """Return list of 3D loop filepaths."""
        return glob.glob((self.datadir / "loops/ITRSensorGeometry*.txt").as_posix())

    def build(self):
        """Read 3D loop coordinates from file."""
        data = {}
        for filepath in self._loop_files:
            group = Path(filepath).name.split("_")[1]
            data[group] = dict(
                zip(self.loop_names(group), self._read_text_file(filepath))
            )
        return data


if __name__ == "__main__":
    mag = Magnetics()
    data = mag.build()
