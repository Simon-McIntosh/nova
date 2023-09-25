"""Biot specific Frame class."""
import numpy as np

from nova.frame.framespace import FrameSpace
from nova.frame.metamethod import CrossSection, Shape, Reduce, PolyGeo


# pylint: disable=too-many-ancestors


class BiotFrame(FrameSpace):
    """Extend FrameSpace class with biot specific attributes and methods."""

    _metadata = ["turns", "reduce"]

    def __init__(self, data=None, index=None, columns=None, attrs=None, **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(PolyGeo, Shape, CrossSection, Reduce)
        for attr in self._metadata:
            setattr(self, attr, None)

    def update_metaframe(self, metadata):
        """Extend metaframe update."""
        self.metaframe.update(
            {
                "required": ["x", "z"],
                "additional": ["plasma", "nturn", "link", "segment", "frame"],
                "array": ["x", "y", "z", "dx", "dy", "dz", "nturn", "plasma"],
            }
        )
        super().update_metaframe(metadata)

    def __call__(self, attr):
        """Return attribute matrix, shape(target, source)."""
        region = self.biotshape.region
        if self.biotshape.region == "":
            raise IndexError(
                "Frame region not specified.\n"
                "Define partner source or target number.\n"
                "self.set_target(number)\n"
                "self.set_source(number)"
            )
        assert region in ["source", "target"]
        partner = next(partner for partner in ["source", "target"] if partner != region)
        reps = getattr(self.biotshape, partner)
        matrix = np.tile(self[attr], reps=(reps, 1))
        if region == "target":
            matrix = np.transpose(matrix)
        return matrix

    def set_target(self, number):
        """Set target number."""
        return self.biotshape.set_target(number)

    def set_source(self, number):
        """Set source number."""
        return self.biotshape.set_source(number)

    @property
    def delta_r(self):
        """Return normalized r-coordinate distance from PF coil centroid."""
        return (self.x - self.xo.values) / self.dx

    @property
    def delta_z(self):
        """Return normalized z-coordinate distance from PF coil centroid."""
        return (self.z - self.zo.values) / self.dz

    @staticmethod
    def _to_local(self, points: np.ndarray, transform: np.ndarray):
        """Return point array mapped to local coordinate system."""
        return np.einsum("ij,ijk->ik", points, transform)

    @staticmethod
    def _to_global(self, points: np.ndarray):
        """Return point array mapped to global coordinate system."""
        return np.einsum("ij,ikj->ik", points, self.transform)


class Source(BiotFrame):
    """Extend BiotFrame with modified additional and available metadata."""

    def update_metaframe(self, metadata):
        """Extend metaframe update."""
        self.metaframe.update(
            {
                "available": [
                    "section",
                    "poly",
                    "x1",
                    "y1",
                    "z1",
                    "x2",
                    "y2",
                    "z2",
                    "ax",
                    "ay",
                    "az",
                    "nx",
                    "ny",
                    "nz",
                ],
                "array": [
                    "x1",
                    "y1",
                    "z1",
                    "x2",
                    "y2",
                    "z2",
                    "ax",
                    "ay",
                    "az",
                    "nx",
                    "ny",
                    "nz",
                    "area",
                ],
            }
        )
        super().update_metaframe(metadata)

    @property
    def center(self):
        """Return arc center."""
        return np.c_[self.loc["x"], self.loc["y"], self.loc["z"]]

    @property
    def start_point(self):
        """Return arc start point."""
        return np.c_[self.loc["x1"], self.loc["y1"], self.loc["z1"]]

    @property
    def end_point(self):
        """Return arc end point."""
        return np.c_[self.loc["x2"], self.loc["y2"], self.loc["z2"]]


class Target(BiotFrame):
    """Extend BiotFrame with modified additional and available metadata."""

    def update_metaframe(self, metadata):
        """Extend metaframe update."""
        self.metaframe.update(
            {
                "additional": ["xo", "zo"],
                "array": ["x"],
            }
        )
        super().update_metaframe(metadata)


if __name__ == "__main__":
    biotframe = BiotFrame()
    biotframe.insert(range(3), 0, dl=0.95, dt=0.95, section="hex")
    biotframe.insert(range(3), 1, dl=0.95, dt=0.95, section="disc", link=True)
    biotframe.insert(range(3), 2, dl=0.95, dt=0.95, section="square", link=False)
    biotframe.insert(range(3), 3, dl=0.95, dt=0.6, section="skin", link=True)

    biotframe.multipoint.link(["Coil0", "Coil11", "Coil2", "Coil8"])
