"""Load magnetic diagnostic geometries and save to IDS."""

from functools import cached_property
from dataclasses import dataclass, field
import glob
import itertools
from pathlib import Path

import io
import pandas
import xarray


@dataclass
class Magnetics:
    """Read 3D magnetic diagnostic loop data."""

    datadir: Path = field(
        default_factory=lambda: Path("//io-ws-ccstore1/ANSYS_Data/mcintos/magnetics")
    )

    @property
    def _magnetics_xlsx(self):
        """Retrun magnetics positions xlsx context manager."""
        return pandas.ExcelFile(
            self.datadir / "List_of_Current_Magnetic_Coil_Positions_24V7KU_v2_11.xlsx",
            engine="openpyxl",
        )

    @cached_property
    def loops(self):
        """Return loops metadata."""
        with self._magnetics_xlsx as xls:
            return pandas.read_excel(xls, "Loops")

    @cached_property
    def _loop_names(self):
        """Return cached grouped loop diagnostic functional part number generator."""
        return {
            key: list(group)
            for key, group in itertools.groupby(
                self.loops.iloc[:, 0].values, lambda name: name.split(".")[1]
            )
        }

    def loop_names(self, group: str):
        """Return loop name list for diagnostic group."""
        return self._loop_names[group]

    def _read_text_file(self, filepath: str):
        data = []
        with open(filepath, "r") as f:
            for block in (
                line
                for newline, line in itertools.groupby(f, lambda line: line[0] == "\n")
                if not newline
            ):
                data.append(
                    pandas.read_csv(
                        io.StringIO("".join(block)), header=None, names=["x", "y", "z"]
                    )
                )
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
    # dataset = xarray.Dataset()
    dataset = xarray.DataArray(data)
