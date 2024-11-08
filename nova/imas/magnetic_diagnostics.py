"""Load magnetic diagnostic geometries and save to IDS."""

from functools import cached_property
from dataclasses import dataclass, field
import glob
import itertools
from pathlib import Path

import io
import pandas


@dataclass
class Magnetics:
    """Read 3D magnetic diagnostic loop data."""

    datadir: Path = field(
        default_factory=lambda: Path("//io-ws-ccstore1/ANSYS_Data/mcintos/magnetics")
    )

    # def coil_positions

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

    def loop_names(self, group: str):
        """Return loop diagnostic functional part numbers for group."""
        return {
            key: list(group)
            for key, group in itertools.groupby(
                mag.loops.iloc[:, 0].values, lambda name: name.split(".")[1]
            )
        }
        # print(data)

    def build(self):
        """Read 3D loop coordinates from file."""
        for file in glob.glob((self.datadir / "loops/*.txt").as_posix()):
            dataset = self._read_text_file(file)
        return dataset

    def _read_text_file(self, filepath: str):
        data = {}
        with open(filepath, "r") as f:
            for group in (
                group
                for newline, group in itertools.groupby(f, lambda line: line[0] == "\n")
                if not newline
            ):
                data["a"] = pandas.read_csv(
                    io.StringIO("".join(group)), header=None, names=["x", "y", "z"]
                )
        return data


if __name__ == "__main__":

    mag = Magnetics()

    mag.build()
