"""Manage fiducial ccl data sources."""
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
import string
from typing import ClassVar

import pandas

from nova.structural.asbuilt import AsBuilt


@dataclass
class Fiducial(ABC):
    """Fiducial CCL base class."""

    target: list[str] = field(default_factory=lambda: list(string.ascii_uppercase[:8]))
    delta: dict[int, pandas.DataFrame] | dict = field(init=False, default_factory=dict)
    origin: list[str] = field(
        init=False,
        default_factory=lambda: [
            "EU",
            "JA",
            "EU",
            "EU",
            "EU",
            "EU",
            "JA",
            "JA",
            "EU",
            "JA",
            "EU",
            "JA",
            "JA",
            "JA",
            "JA",
            "JA",
            "EU",
            "EU",
            "JA",
        ],
    )
    source: str = ""

    def __post_init__(self):
        """Load avalible fiducial deltas."""
        self._load_deltas()

    @abstractmethod
    def _load_deltas(self):
        """Load CCL fiducial deltas - set delta and origin."""

    @property
    def data(self):
        """Return (delta, origin)."""
        return self.delta, self.origin


@dataclass
class FiducialIDM(Fiducial):
    """Manage ccl fiducial data extracted from IDM documents."""

    origin: list[str] = field(init=False, default_factory=list)
    source: str = "IDM dataset"

    name: ClassVar[str] = ""

    def _load_deltas(self):
        for i in range(1, 20):
            index = f"{i:02d}"
            try:
                data = getattr(self, f"_tfc{index}")
                self.delta[i] = data[0].reindex(self.target)
                self.origin.append(data[1])
            except NotImplementedError:
                continue

    @property
    def _tfc01(self):
        """Return TFC01 fiducial data."""
        raise NotImplementedError("TFC01 - EU - pending")

    @property
    def _tfc02(self):
        """Return TFC02 fiducial data - JA."""
        return (
            pandas.DataFrame(
                index=["A", "B", "C", "D", "E", "F", "G", "H"],
                columns=["dx", "dy", "dz"],
                data=[
                    [-0.63, 0.41, 0.09],
                    [-0.68, 0.43, -0.12],
                    [-0.52, 0.92, 1.67],
                    [-2.67, 0.92, -2.35],
                    [-4.41, 0.53, -0.32],
                    [-1.9, 1.69, -0.05],
                    [-4.76, -0.68, -1.73],
                    [0.25, -0.49, 0.04],
                ],
            ),
            "JA",
        )

    @property
    def _tfc03(self):
        """Return TFC03 fiducidal data - 52Z4PV - F4E_D_2REWA9 v2.0."""
        return (
            pandas.DataFrame(
                index=[
                    "A",
                    "1-A",
                    "1",
                    "1-21",
                    "1-2",
                    "1-22",
                    "2",
                    "B-2",
                    "B",
                    "3",
                    "4",
                    "5",
                    "C",
                    "6",
                    "7",
                    "8",
                    "D",
                    "9",
                    "10",
                    "11",
                    "G",
                    "12",
                    "13",
                    "14",
                    "E",
                    "15",
                    "16",
                    "18",
                    "19",
                    "20",
                ],
                columns=["dx", "dy", "dz"],
                data=[
                    [-0.8, 0.14, -0.25],
                    [-0.5, 0.15, -0.21],
                    [0.2, -0.19, -0.01],
                    [0.15, -0.45, -0.01],
                    [-0.03, -0.39, -0.01],
                    [0.05, -0.23, 0.04],
                    [-0.18, 0.23, 0.08],
                    [-0.65, 0.06, 0.08],
                    [-1.16, 0.54, 0.01],
                    [-1.66, 1.19, 0.82],
                    [-1.51, 1.57, 1.57],
                    [-1.26, 0.54, 1.7],
                    [-1.4, 1.09, 1.76],
                    [-1.72, 0.61, 0.76],
                    [-2.26, -0.16, -0.01],
                    [-2.8, -0.66, -0.49],
                    [-3.42, -0.69, -0.88],
                    [-4.08, -0.59, -1.04],
                    [-4.75, -0.9, -0.85],
                    [-5.15, -0.74, -0.48],
                    [-5.31, -0.54, -0.15],
                    [-5.35, -0.7, 0.09],
                    [-4.88, -0.54, 0.32],
                    [-4.24, -0.32, 0.48],
                    [-3.6, -0.16, 0.64],
                    [-2.63, 0.35, 0.36],
                    [0.19, 1.21, -2.7],
                    [-0.22, 0.87, -0.82],
                    [-0.28, 0.37, -0.62],
                    [-0.74, 0.23, -0.6],
                ],
            ),
            "EU",
        )

    @property
    def _tfc04(self):
        """Return TFC04 fiducial data."""
        raise NotImplementedError("TFC04 - EU - pending")

    @property
    def _tfc05(self):
        """Return TFC05 fiducidal data - 4HMUWH - F4E_D_2PYAKN v2.0."""
        return (
            pandas.DataFrame(
                index=[
                    "A",
                    "1-A",
                    "1",
                    "1-21",
                    "1-2",
                    "1-22",
                    "2",
                    "B-2",
                    "B",
                    "3",
                    "4",
                    "5",
                    "C",
                    "6",
                    "7",
                    "8",
                    "D",
                    "9",
                    "10",
                    "11",
                    "G",
                    "12",
                    "13",
                    "14",
                    "E",
                    "15",
                    "16",
                    "18",
                    "19",
                    "20",
                ],
                columns=["dx", "dy", "dz"],
                data=[
                    [-0.66, 0.82, -0.04],
                    [-0.24, -0.11, 0.03],
                    [0.21, -0.61, 0.16],
                    [0.21, -0.67, 0.13],
                    [0.13, -0.85, 0.1],
                    [0.0, -0.78, 0.05],
                    [-0.12, -0.33, 0.0],
                    [-0.07, 0.23, -0.03],
                    [-0.6, 0.98, 0.0],
                    [-1.28, 2.03, 0.85],
                    [-1.23, 2.24, 1.64],
                    [-1.05, 1.69, 1.93],
                    [-1.17, 0.98, 2.18],
                    [-1.39, 0.89, 1.43],
                    [-1.8, -0.33, 0.91],
                    [-2.22, -0.86, 0.54],
                    [-2.7, -1.01, 0.2],
                    [-3.31, -0.96, -0.12],
                    [-3.78, -0.76, -0.26],
                    [-4.35, -0.95, -0.32],
                    [-4.27, -0.99, -0.22],
                    [-4.26, -0.97, -0.08],
                    [-4.08, -0.43, 0.11],
                    [-3.9, -0.02, 0.37],
                    [-3.51, 0.11, 0.59],
                    [-2.92, 0.72, 0.81],
                    [-0.29, 1.77, -1.96],
                    [0.06, 2.74, 0.17],
                    [0.23, 2.71, 0.04],
                    [-0.28, 2.03, -0.29],
                ],
            ),
            "EU",
        )

    @property
    def _tfc06(self):
        """Return TFC06 fiducidal data - 5PPCAF - F4E_D_2RTP8J v1.2 EU."""
        return (
            pandas.DataFrame(
                index=[
                    "A",
                    "1-A",
                    "1",
                    "1-21",
                    "1-2",
                    "1-22",
                    "2",
                    "B-2",
                    "B",
                    "3",
                    "4",
                    "5",
                    "C",
                    "6",
                    "7",
                    "8",
                    "D",
                    "9",
                    "10",
                    "11",
                    "G",
                    "12",
                    "13",
                    "14",
                    "E",
                    "15",
                    "16",
                    "18",
                    "19",
                    "20",
                ],
                columns=["dx", "dy", "dz"],
                data=[
                    [-0.5, 0.64, -0.12],
                    [0.03, 0.07, -0.08],
                    [0.55, 0.02, 0.09],
                    [0.54, -0.21, 0.02],
                    [0.47, -0.74, -0.05],
                    [0.38, -0.88, 0.08],
                    [0.22, -0.38, 0.21],
                    [-0.33, 0.27, 0.04],
                    [-0.9, 1.42, -0.04],
                    [-1.44, 1.85, 0.79],
                    [-1.21, 2.17, 1.49],
                    [-0.94, 2.25, 1.65],
                    [-1.04, 1.55, 1.66],
                    [-1.37, 1.34, 0.73],
                    [-1.89, 0.45, 0.16],
                    [-2.37, -0.22, -0.19],
                    [-2.95, -0.38, -0.48],
                    [-3.65, -0.62, -0.69],
                    [-4.41, -0.79, -0.73],
                    [-4.94, -1.01, -0.6],
                    [-4.69, -0.9, -0.1],
                    [-4.78, -0.91, 0.23],
                    [-4.41, -0.76, 0.33],
                    [-3.7, -0.49, 0.28],
                    [-3.37, 0.08, 0.41],
                    [-2.22, 1.08, 0.48],
                    [0.38, 2.25, -2.42],
                    [0.12, 2.67, -0.58],
                    [-0.01, 2.65, -0.54],
                    [-0.44, 2.32, -0.5],
                ],
            ),
            "EU",
        )

    @property
    def _tfc07(self):
        """Return TFC07 fiducial data."""
        raise NotImplementedError("TFC07 - JA - pending")

    @property
    def _tfc08(self):
        """Return TFC08 fiducidal data - JA."""
        return (
            pandas.DataFrame(
                index=["A", "B", "C", "D", "E", "F", "G", "H"],
                columns=["dx", "dy", "dz"],
                data=[
                    [-0.69, -0.62, -0.52],
                    [-0.72, -0.72, -1.08],
                    [-1.19, 1.11, -0.08],
                    [-2.56, 0.25, -0.3],
                    [-3.17, -1.97, -0.07],
                    [-1.48, -0.9, 0.21],
                    [-3.17, 0.04, -0.22],
                    [0.34, 0.48, -2.31],
                ],
            ),
            "JA",
        )

    @property
    def _tfc09(self):
        """Return TFC09 fiducidal data - 2SU8F4 - F4E_D_2KYN3R v2.0 EU."""
        return (
            pandas.DataFrame(
                index=[
                    "A",
                    "1-A",
                    "1",
                    "1-21",
                    "1-2",
                    "1-22",
                    "2",
                    "B-2",
                    "B",
                    "3",
                    "4",
                    "5",
                    "C",
                    "6",
                    "7",
                    "8",
                    "D",
                    "9",
                    "10",
                    "11",
                    "G",
                    "12",
                    "13",
                    "14",
                    "E",
                    "15",
                    "16",
                    "18",
                    "19",
                    "20",
                ],
                columns=["dx", "dy", "dz"],
                data=[
                    [-0.3, 0.32, -0.16],
                    [-0.02, 0.32, -0.04],
                    [0.2, 0.23, 0.06],
                    [0.25, -0.05, 0.14],
                    [0.28, -0.31, 0.21],
                    [0.06, -0.25, 0.18],
                    [-0.16, -0.18, 0.15],
                    [-0.38, -0.16, 0.12],
                    [-0.64, -0.26, 0.07],
                    [-0.99, -0.11, 0.22],
                    [-1.29, 0.25, 0.5],
                    [-1.59, 0.6, 0.77],
                    [-1.88, 0.92, 1.02],
                    [-2.09, 0.79, 0.79],
                    [-2.3, 0.65, 0.56],
                    [-2.5, 0.52, 0.33],
                    [-2.72, 0.39, 0.1],
                    [-2.95, 0.23, -0.17],
                    [-3.2, 0.07, -0.44],
                    [-3.44, -0.08, -0.71],
                    [-3.34, -0.15, -0.92],
                    [-3.09, -0.16, -1.07],
                    [-2.84, -0.18, -1.22],
                    [-2.6, -0.19, -1.36],
                    [-2.35, -0.22, -1.5],
                    [-2.03, -0.23, -1.68],
                    [-1.71, -0.23, -1.83],
                    [-1.08, 0.17, -0.77],
                    [-0.9, 0.29, -0.47],
                    [-0.61, 0.32, -0.29],
                ],
            ),
            "EU",
        )

    @property
    def _tfc10(self):
        """Return TFC10 fiducial data - JA."""
        return (
            pandas.DataFrame(
                index=["A", "B", "C", "D", "E", "F", "G", "H"],
                columns=["dx", "dy", "dz"],
                data=[
                    [0.14, 0.3, -0.72],
                    [0.08, 0.17, -1.07],
                    [-1.81, -0.61, -1.25],
                    [-1.99, 0.11, -2.13],
                    [-2.13, 0.06, -0.51],
                    [0.3, 0.98, 0.85],
                    [-2.85, 0.5, -1.09],
                    [0.19, -0.36, -1.0],
                ],
            ),
            "JA",
        )

    @property
    def _tfc11(self):
        """Return TFC11 fiducial data - 3T6WVX - F4E_D_2NNX2Y v3.0 EU."""
        return (
            pandas.DataFrame(
                index=[
                    "A",
                    "1-A",
                    "1",
                    "1-21",
                    "1-2",
                    "1-22",
                    "2",
                    "B-2",
                    "B",
                    "3",
                    "4",
                    "5",
                    "C",
                    "6",
                    "7",
                    "8",
                    "D",
                    "9",
                    "10",
                    "11",
                    "G",
                    "12",
                    "13",
                    "14",
                    "E",
                    "15",
                    "16",
                    "18",
                    "19",
                    "20",
                ],
                columns=["dx", "dy", "dz"],
                data=[
                    [-0.84, -0.09, -0.16],
                    [-0.31, -0.39, 0.0],
                    [0.04, -0.07, 0.12],
                    [0.09, 0.04, 0.1],
                    [-0.03, -0.07, 0.07],
                    [-0.04, -0.11, 0.05],
                    [-0.12, -0.24, 0.03],
                    [-0.45, -0.07, 0.01],
                    [-1.1, 0.1, -0.01],
                    [-2.12, 0.14, 0.88],
                    [-1.83, 0.22, 1.5],
                    [-1.58, 0.23, 1.6],
                    [-1.73, -0.31, 1.65],
                    [-1.94, -0.49, 0.81],
                    [-2.32, -0.72, 0.33],
                    [-2.65, -0.73, 0.11],
                    [-3.24, -0.65, -0.27],
                    [-3.86, -0.64, -0.44],
                    [-4.47, -0.73, -0.46],
                    [-4.92, -0.22, -0.33],
                    [-4.96, -0.01, -0.17],
                    [-4.78, -0.19, -0.02],
                    [-4.45, 0.06, 0.15],
                    [-4.03, 0.22, 0.29],
                    [-3.56, -0.02, 0.44],
                    [-2.76, 0.21, 0.48],
                    [-0.29, 0.56, -2.07],
                    [-0.17, 1.57, -0.79],
                    [-0.26, 1.09, -0.7],
                    [-0.82, 0.83, -0.67],
                ],
            ),
            "EU",
        )

    @property
    def _tfc12(self):
        """Return TFC12 fiducial data - JA 2UD358."""
        return (
            self.coordinate_transform(
                pandas.DataFrame(
                    index=["A", "B", "C", "D", "E", "F", "G", "H"],
                    columns=["dx", "dy", "dz"],
                    data=[
                        [-0.21, 0.17, -0.7],
                        [-1.34, -0.82, -0.32],
                        [0.27, 0.63, -1.78],
                        [1.29, -2.06, -3.4],
                        [1.74, -0.97, -5.1],
                        [0.57, -0.89, -2.51],
                        [0.96, -2.12, -6.22],
                        [-0.77, -0.08, 0.62],
                    ],
                )
            ),
            "JA",
        )

    @property
    def _tfc13(self):
        """Return TFC13 fiducial data - 3B5YEM - JA."""
        return (
            pandas.DataFrame(
                index=["A", "B", "C", "D", "E", "F", "G", "H"],
                columns=["dx", "dy", "dz"],
                data=[
                    [-0.3, 0.26, -0.8],
                    [-0.29, -0.32, -1.21],
                    [-2.26, 2.19, 1.43],
                    [-0.44, -0.05, -1.87],
                    [-4.78, -0.54, -1.69],
                    [0.44, 0.53, -0.44],
                    [-2.86, -1.28, -2.27],
                    [-0.08, -0.71, -1.36],
                ],
            ),
            "JA",
        )

    @property
    def _tfc14(self):
        """Return TFC14 fiducial data - EU."""
        raise NotImplementedError("TFC14 - JA - pending")

    @property
    def _tfc15(self):
        """Return TFC15 fiducial data - JA."""
        raise NotImplementedError("TFC15 - JA - pending")

    @property
    def _tfc16(self):
        """Return TFC16 fiducial data - JA."""
        raise NotImplementedError("TFC16 - JA - pending")

    @property
    def _tfc17(self):
        """Return TFC17 fiducial data - EU."""
        raise NotImplementedError("TFC17 - EU - pending")

    @property
    def _tfc18(self):
        """Return TFC18 fiducial data - EU."""
        raise NotImplementedError("TFC18 - EU - pending")

    @property
    def _tfc19(self):
        """Return TFC19 fiducial data - JA."""
        raise NotImplementedError("TFC19 - JA - pending")

    @staticmethod
    def coordinate_transform(mcs):
        """
        Convert ccl delta coordinates from AU (JA) to space.

        space to MCS:
            𝑋𝑀𝐶𝑆 = 5334.4 − 𝑋𝑇𝐺𝐶𝑆
            𝑌𝑀𝐶𝑆 = 29 − 𝑍𝑇𝐺𝐶𝑆
            𝑍𝑀𝐶𝑆 = 𝑌𝑇𝐺𝐶𝑆

        From MCS to space:
            𝑋𝑇𝐺𝐶𝑆 = 5334.4 − 𝑋𝑀𝐶𝑆
            𝑌𝑇𝐺𝐶𝑆 = 𝑍𝑀𝐶𝑆
            𝑍𝑇𝐺𝐶𝑆 = 29 - 𝑌𝑀𝐶𝑆

        """
        space = pandas.DataFrame(index=mcs.index, columns=mcs.columns)
        space.loc[:, "dx"] = mcs.dz
        space.loc[:, "dy"] = mcs.dx
        space.loc[:, "dz"] = mcs.dy
        return space

    @staticmethod
    def read_clipboard(column_index=slice(3, 6)):
        """Read displacment data from clipboard."""
        # pylint: disable=no-member
        ccl = pandas.read_clipboard(header=None)
        ccl.set_index(0, inplace=True)
        ccl.index.name = None
        ccl = ccl.iloc[:, column_index]
        ccl.columns = ["dx", "dy", "dz"]
        ccl = ccl.iloc[~(ccl == "-").any(axis=1).values, :]
        return ccl.dropna(0).astype(float)


@dataclass
class FiducialRE(Fiducial):
    """Manage Reverse Engineering fiducial data."""

    file: str = "TFC18_asbuilt"
    phase: str = "FAT supplier"

    def __post_init__(self):
        """Propogate origin."""
        super().__post_init__()
        self.source = f"Reverse Engineering dataset {self.file}.xlsx"
        self.origin = [
            origin for i, origin in enumerate(self.origin) if i + 1 in self.delta
        ]

    def _load_deltas(self):
        """Load asbuilt ccl deltas."""
        self.delta = AsBuilt().ccl_deltas()
        for coil in self.delta:
            self.delta[coil] = self.delta[coil].reindex(self.target)


if __name__ == "__main__":
    idm = FiducialIDM()
    re = FiducialRE()
