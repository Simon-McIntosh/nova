from dataclasses import dataclass, field
from typing import ClassVar
from functools import cached_property

import numpy as np
import pandas

from nova.assembly.ilisnominal import NominalIlis


@dataclass
class FiducialIlis:
    """Fit planes to Ilis points"""

    data: pandas.DataFrame = field(repr=False)
    pcr: bool = True  # incorporate pcr data for ILIS offsets
    # outlier_limit: float = 20
    planes: pandas.DataFrame = field(init=False)

    deviation: ClassVar[dict[int, list]] = {
        1: [np.nan, np.nan],
        2: [0, 0],
        3: [-0.1, 0],
        4: [0.15 + 0.4, 0],  # reduce inter-sector gap
        5: [1.5, 0 + 0.1],  # S5+ reduce inter-sector gap
        6: [0, 0],
        7: [0, 0],
        8: [-1, 0],  # S7-
        9: [0, 1],  # S7+
        10: [0.1, 0],
        11: [0, -0.15 + 0.4],  # reduce inter-sector gap
        12: [0, -1.5],  # S6-
        13: [0, 0],  # S6+
        14: [0, 0],
        15: [0, 0],
        16: [0 + 0.1, 0.2],  # S5- reduce inter-sector gap
        17: [np.nan, np.nan],
        18: [0, 0],
    }  # ilis deviation [positive side, negative side]

    def __post_init__(self):
        """Build geometry from input data."""
        self._identify_outliers()
        self._extract_planes()

    def _identify_outliers(self):
        """Identify outliers in input point-cloud ILIS plane measurement sets."""
        self.planes = NominalIlis.fit_plane(
            self.data
        )  # use a temporary set of planes for outlier detection algorithm

        outlier = self.data.groupby(["coil", "feature"], group_keys=False).apply(
            lambda x: pandas.Series(self._detect(x, x.name, 3))
        )

        # Ensure consistent flat Series output regardless of number of coil types
        if isinstance(outlier, pandas.DataFrame):
            # For single coil case: flatten the DataFrame to a Series
            outlier = outlier.stack().droplevel(-1)

        self.data.loc[:, "outlier"] = outlier.reset_index(drop=True)

        offset = self.data.groupby(["coil", "feature"], group_keys=False).apply(
            lambda x: pandas.Series(self.offset(x.loc[:, ["x", "y", "z"]], x.name))
        )

        # Ensure consistent flat Series output regardless of number of coil types
        if isinstance(offset, pandas.DataFrame):
            # For single coil case: flatten the DataFrame to a Series
            offset = offset.stack().droplevel(-1)

        self.data.loc[:, "offset"] = offset.reset_index(drop=True)

    @property
    def outliers(self):
        """Return DataFrame of detected outlier points."""
        return self.data[self.data.outlier]

    @property
    def n_outliers(self):
        """Return count of outliers by coil and feature."""
        return self.data.groupby(["coil", "feature"])["outlier"].sum()

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

    @staticmethod
    def intersect(planes):
        """Intersect planes to find midplane."""
        points = planes.loc[:, ["x", "y", "z"]].copy().values
        normals = planes.loc[:, ["nx", "ny", "nz"]].copy().values
        dot_normals = np.dot(*normals)
        distance = np.einsum("ij,ij->i", points, normals)
        coef = np.linalg.solve(np.array([[1, dot_normals], [dot_normals, 1]]), distance)
        point = coef @ normals
        midplane = planes.mean(axis=0)
        midplane.loc[["x", "y", "z"]] = point
        return midplane

    def _extract_planes(self):
        """Extract ilis and center planes from input data."""
        """
        pca = sklearn.decomposition.PCA(3, random_state=2025)
        normals = (
            self.data.loc[:, ["x", "y", "z"]]
            .groupby([self.data.coil, self.data.feature])
            .apply(
                lambda x: pandas.Series(
                    pca.fit(x).components_[-1], index=["nx", "ny", "nz"]
                )
            )
        )
        points = (
            self.data.loc[:, ["x", "y", "z"]]
            .groupby([self.data.coil, self.data.feature])
            .mean()
        )
        self.planes = points.join(normals).join(
            self.ilis_offset, how="inner", on=["coil", "feature"]
        )

        print(NominalIlis.fit_plane(self.data))
        """
        # Filter out outliers for plane fitting
        filtered_data = self.data[~self.data.outlier]
        self.planes = NominalIlis.fit_plane(filtered_data).join(
            self.ilis_offset, how="inner", on=["coil", "feature"]
        )

        if self.pcr:  # offset ilis planes by deviation
            self.planes.loc[:, ["x", "y", "z"]] -= (
                self.planes.loc[:, "offset"].values[:, np.newaxis]
                * self.planes.loc[:, ["nx", "ny", "nz"]].values
            )

        midplane = self.planes.groupby(level=0).apply(lambda x: self.intersect(x))
        """
        midplane = self.planes.groupby(level=0).mean()
        midplane.loc[:, ["nx", "ny", "nz"]] = midplane.loc[:, ["nx", "ny", "nz"]].agg(
            lambda x: x / np.linalg.norm(x), axis=1
        )
        """
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

    def _detect(self, points, plane, standard_deviations=3):
        """Detect outliers in points relative to plane.

        Outliers are detected using two criteria:
        1. Points > N standard deviations from the best-fit plane
        2. Points with signed offset inconsistent with their ILIS side
           (e.g., a point labeled +ILIS but located on the -ILIS side)
        """
        offset = self.offset(points, plane, return_normal=False)
        dist = np.abs(offset)
        std = np.std(dist)
        threshold = standard_deviations * std
        outliers_distance = dist > threshold

        # Check for sign consistency: +ILIS points should have positive y (approx)
        # and -ILIS points should have negative y in the raw data
        feature = plane[1] if isinstance(plane, tuple) else plane
        if "+1" in str(feature):
            # ILIS +1 points should generally have positive y
            outliers_sign = points["y"].values < -50  # Allow some tolerance
        elif "-1" in str(feature):
            # ILIS -1 points should generally have negative y
            outliers_sign = points["y"].values > 50  # Allow some tolerance
        else:
            outliers_sign = np.zeros(len(points), dtype=bool)

        outliers = outliers_distance | outliers_sign

        return outliers

    def offset(self, points, plane, return_normal=False):
        """Return signed offset between points and plane."""

        if isinstance(plane, (pandas.Index, tuple)):
            plane = self.planes.loc[plane]

        normal = plane.loc[["nx", "ny", "nz"]].values
        point = plane.loc[["x", "y", "z"]].values

        # Ensure normal is normalized
        normal = normal / np.linalg.norm(normal, axis=0)

        # Calculate signed distance from each point to plane
        v = points.loc[:, ["x", "y", "z"]] - point
        offset = np.dot(v, normal)
        if return_normal:
            return offset, normal
        return offset

    def _project(
        self, points: pandas.DataFrame, plane: pandas.DataFrame | pandas.Index | tuple
    ) -> pandas.DataFrame:
        """Project points onto midplane (ILIS 0)."""
        """
        if isinstance(plane, (pandas.Index, tuple)):
            plane = self.planes.loc[plane]

        normal = plane.loc[["nx", "ny", "nz"]].values
        point = plane.loc[["x", "y", "z"]].values

        # Ensure normal is normalized
        normal = normal / np.linalg.norm(normal)

        # Calculate signed distance from each point to plane
        v = points - point
        dist = np.dot(v, normal)
        """
        offset, normal = self.offset(points, plane, return_normal=True)
        # Project by subtracting distance along normal
        projected = points - np.outer(offset, normal)
        return projected


if __name__ == "__main__":
    from nova.assembly.fiducialsector import FiducialSector

    fiducial = FiducialSector(phase="SSAT AL", sectors={8: [4, 11]})

    ilis = FiducialIlis(fiducial.ilis)

    # print(ilis.project(ilis.planes, "ILIS 0"))
