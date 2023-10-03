"""Biot specific Frame class."""
import numpy as np

from nova.frame.framespace import FrameSpace
from nova.frame.metamethod import CrossSection, Shape, Space, Reduce, PolyGeo


# pylint: disable=too-many-ancestors


class BiotFrame(FrameSpace):
    """Extend FrameSpace class with biot specific attributes and methods."""

    _metadata = ["turns", "reduce"]

    def __init__(self, data=None, index=None, columns=None, attrs=None, **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(PolyGeo, Shape, Space, CrossSection, Reduce)
        for attr in self._metadata:
            setattr(self, attr, None)

    def update_metaframe(self, metadata):
        """Extend metaframe update."""
        self.metaframe.update(
            {
                "required": ["x", "z"],
                "base": ["x", "y", "z"],
                "additional": ["plasma", "nturn", "link", "frame"],
                "array": [
                    "x",
                    "y",
                    "z",
                    "r",
                    "ax",
                    "ay",
                    "az",
                    "nx",
                    "ny",
                    "nz",
                    "dx",
                    "dy",
                    "dz",
                    "nturn",
                    "plasma",
                ],
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

    def stack(self, *args):
        """Return stacked attribute array combining instance calls along last axis."""
        return np.stack([self(attr) for attr in args], axis=-1)

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


class Source(BiotFrame):
    """Extend BiotFrame with modified additional and available metadata."""

    def update_metaframe(self, metadata):
        """Extend metaframe update."""
        self.metaframe.update(
            {
                "available": [
                    "segment",
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
        """Return element center."""
        return np.c_[self.aloc["x"], self.aloc["y"], self.aloc["z"]]

    @property
    def start_point(self):
        """Return element start point."""
        return np.c_[self.aloc["x1"], self.aloc["y1"], self.aloc["z1"]]

    @property
    def end_point(self):
        """Return element end point."""
        return np.c_[self.aloc["x2"], self.aloc["y2"], self.aloc["z2"]]

    @property
    def axis(self):
        """Return element axis."""
        return np.c_[self.aloc["ax"], self.aloc["ay"], self.aloc["az"]]

    @property
    def normal(self):
        """Return element normal."""
        return np.c_[self.aloc["nx"], self.aloc["ny"], self.aloc["nz"]]


class Target(BiotFrame):
    """Extend BiotFrame with modified additional and available metadata."""

    def update_metaframe(self, metadata):
        """Extend metaframe update."""
        self.metaframe.update(
            {
                "additional": ["xo", "zo", "dx", "dz"],
                "array": ["x", "y", "z"],
                "available": [],
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
