"""Load and process nominal ILIS data from CSV file."""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas
import sklearn.decomposition

from nova.database.filepath import FilePath


@dataclass
class NominalIlis:
    """Load and process ILIS data from CSV file.

    Parameters
    ----------
    filename : str
        Name of the CSV file
    dirname : str
        Directory path containing the file

    Examples
    --------
    >>> data = IlisNominal("ILIS_nominal.txt", "//path/to/data")
    >>> print(data.data)
    """

    filename: str = "ILIS_nominal.txt"
    dirname: str = "//io-ws-ccstore1/ANSYS_Data/mcintos/sector_modules"
    _data: pandas.DataFrame = field(init=False, repr=False)
    _cache_file: Path = field(init=False, repr=False)
    _cache_dir: Path | str = ".nova/sector_modules"

    def __post_init__(self):
        self._cache_file = FilePath(
            Path(self.filename).stem + ".pickle", dirname=self._cache_dir
        ).filepath
        if self._cache_file.exists():
            self._data = pandas.read_pickle(self._cache_file)
        else:
            filepath = FilePath(self.filename, dirname=self.dirname).filepath
            self._data = pandas.read_csv(
                filepath, sep=",", header=0, names=["name", "x", "y", "z"]
            )
            self._process_data()
            self._data.to_pickle(self._cache_file)

    def clear_cache(self):
        """Clear cache file."""
        if self._cache_file.exists():
            self._cache_file.unlink()

    def _process_data(self) -> None:
        """Process the raw data by adding features and setting index."""
        self._data["coil"] = 0
        self._data["feature"] = self._data.y.map(
            lambda x: "ILIS -1" if x < 0 else "ILIS +1"
        )
        self._data.set_index(["coil", "feature"], inplace=True)

    @property
    def data(self) -> pandas.DataFrame:
        """Return the processed DataFrame."""
        return self._data

    @classmethod
    def angle_to_xz(cls, planes: pandas.DataFrame):
        """Reutrn angle in degrees of planes grouped by `feature` to xz plane."""
        return planes[["nx", "ny", "nz"]].apply(
            lambda x: np.degrees(cls.dihedral_angle(x, np.array([0, 1, 0]))),
            axis=1,
        )

    @classmethod
    def angle_to_xy(cls, planes: pandas.DataFrame):
        """Reutrn angle in degrees of planes grouped by `feature` to xz plane."""
        return planes[["nx", "ny", "nz"]].apply(
            lambda x: np.degrees(cls.dihedral_angle(x, np.array([0, 0, 1]))),
            axis=1,
        )

    @staticmethod
    def dihedral_angle(n1, n2) -> float:
        """Return the signed dihedral angle between two planes."""
        n0 = np.cross(n1, n2)
        n01 = np.cross(n0, n1)
        n02 = np.cross(n0, n2)
        return np.arccos(
            (n01 @ n02) / (np.linalg.norm(n01) * np.linalg.norm(n02)), dtype=float
        )

    @cached_property
    def planes(self):
        """Return nominal ILIS planes with outwards facing normals."""
        data = self.data.reset_index()
        data["coil"] = 0
        planes = self.fit_plane(data)
        planes.loc[(0, "ILIS 0"), :] = planes.mean()
        planes.loc[:, ["nx", "ny", "nz"]] = (
            planes[["nx", "ny", "nz"]]
            .groupby("feature", group_keys=False)
            .apply(
                lambda x: sign * x
                if (sign := np.sign(int(x.name.split()[-1]))) in [-1, 1]
                else x
            )
        )
        planes.loc[:, ["nx", "ny", "nz"]] /= np.linalg.norm(
            planes.loc[:, ["nx", "ny", "nz"]], axis=1
        )[:, np.newaxis]
        return planes

    @staticmethod
    def fit_plane(data):
        """Return best fit plane mean point and normal vector for input dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            Input dataframe containing x,y,z coordinates and coil,feature grouping columns.

        Returns
        -------
        pandas.DataFrame
            DataFrame with mean point and normal vector for each coil/feature group.

        Examples
        --------
        >>> data = pandas.DataFrame({
        ...     'x': [1,2,3], 'y': [4,5,6], 'z': [7,8,9],
        ...     'coil': [1,1,1], 'feature': ['A','A','A']
        ... })
        >>> fit_plane(data)
        """
        pca = sklearn.decomposition.PCA(3, random_state=2025)
        normals = (
            data.loc[:, ["x", "y", "z"]]
            .groupby([data.coil, data.feature])
            .apply(
                lambda x: pandas.Series(
                    pca.fit(x).components_[-1], index=["nx", "ny", "nz"]
                )
            )
        )
        points = data.loc[:, ["x", "y", "z"]].groupby([data.coil, data.feature]).mean()
        return points.join(normals)

    def get_offset(self, points, plane, return_normal=False):
        """Return signed offset between points and plane."""
        if isinstance(plane, (pandas.Index, tuple)):
            plane = self.planes.loc[(0, plane[1])]

        normal = plane.loc[["nx", "ny", "nz"]].values
        point = plane.loc[["x", "y", "z"]].values

        # Ensure normal is normalized
        normal = normal / np.linalg.norm(normal)

        # Calculate signed distance from each point to plane
        vector = points.loc[:, ["x", "y", "z"]] - point
        offset = vector @ normal
        if return_normal:
            return offset, normal
        return offset

    def analize_offsets(self, points):
        """Return offset statstistics between points and plane."""
        return points.groupby(["coil", "feature"]).apply(
            lambda x: self.get_offset(x, x.name).agg(["min", "max", "mean", "std"])
        )


if __name__ == "__main__":
    # Example usage:
    print(NominalIlis().planes)

    nominal = NominalIlis()

    print(nominal.analize_offsets(nominal.data))
