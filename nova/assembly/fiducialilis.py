from dataclasses import dataclass, field
from typing import ClassVar
from functools import cached_property

import numpy as np
import pandas
import sklearn.decomposition


@dataclass
class FiducialIlis:
    """Intersection Line Intersection Surface class.

    A class to handle intersection line and surface calculations.
    """

    data: pandas.DataFrame = field(repr=False)
    pcr: bool = True  # incorporate pcr data for ILIS offsets
    outlier_limit: float = 20
    planes: pandas.DataFrame = field(init=False)

    deviation: ClassVar[dict[int, list]] = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [-1, 0],
        9: [0, 1],
        10: [],
        11: [],
        12: [0, -1.5],
        13: [0, 0],
        14: [],
        15: [],
        16: [],
        17: [],
        18: [],
    }  # ilis deviation [positive side, negative side]

    def __post_init__(self):
        """Build geometry from input data."""
        self.data = self.data.loc[self.data.mahalanobis < self.outlier_limit, :]
        self._extract_planes()

    @cached_property
    def ilis_offset(self):
        """Extract ILIS offset from deviation data."""
        return pandas.DataFrame(
            [
                {
                    "coil": coil,
                    "feature": f"ILIS {side}",
                    "offset": (-1) ** i * offset[i] if offset else 0,
                }
                for coil, offset in self.deviation.items()
                for i, side in enumerate(["+1", "-1"])
            ]
        ).set_index(["coil", "feature"])

    def _extract_planes(self):
        """Extract ilis and center planes from input data."""
        pca = sklearn.decomposition.PCA(3)
        normals = (
            self.data.loc[:, ["x", "y", "z"]]
            .groupby([self.data.coil, self.data.feature])
            .apply(
                lambda x: pandas.Series(
                    pca.fit(x).components_[-1], index=["nx", "ny", "nz"]
                )
            )
        )
        points = self.data.loc[:, ["x", "y", "z"]].groupby(self.data.feature).mean()
        self.planes = points.join(normals).join(
            self.ilis_offset, how="inner", on=["coil", "feature"]
        )
        if self.pcr:  # offset ilis planes by deviation
            self.planes.loc[:, ["x", "y", "z"]] -= (
                self.planes.loc[:, "offset"].values[:, np.newaxis]
                * self.planes.loc[:, ["nx", "ny", "nz"]].values
            )
        midplane = self.planes.groupby(level=0).mean()
        midplane.loc[:, ["nx", "ny", "nz"]] = midplane.loc[:, ["nx", "ny", "nz"]].agg(
            lambda x: x / np.linalg.norm(x), axis=1
        )
        midplane.loc[:, "feature"] = "ILIS 0"
        midplane.set_index("feature", append=True, inplace=True)
        self.planes = pandas.concat([self.planes, midplane]).sort_index()

    def project(
        self, points: pandas.DataFrame, plane: str = "ILIS 0"
    ) -> pandas.DataFrame:
        """Project points onto plane."""
        return points.groupby(["coil"], group_keys=False).apply(
            lambda x: self._project(x.loc[:, ["x", "y", "z"]], (x.name, plane))
        )

    def _project(
        self, points: pandas.DataFrame, plane: pandas.DataFrame | pandas.Index | tuple
    ) -> pandas.DataFrame:
        """Project points onto midplane (ILIS 0)."""

        if isinstance(plane, (pandas.Index, tuple)):
            plane = self.planes.loc[plane]

        normal = plane.loc[["nx", "ny", "nz"]].values
        point = plane.loc[["x", "y", "z"]].values

        # Ensure normal is normalized
        normal = normal / np.linalg.norm(normal)

        # Calculate signed distance from each point to plane
        v = points - point
        dist = np.dot(v, normal)

        # Project by subtracting distance along normal
        projected = points - np.outer(dist, normal)

        return projected


if __name__ == "__main__":
    from nova.assembly.fiducialsector import FiducialSector

    fiducial = FiducialSector(phase="SSAT BR", sectors={7: [8, 9]})

    ilis = FiducialIlis(fiducial.ilis)

    print(ilis.project(ilis.planes, "ILIS 0"))
