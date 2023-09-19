"""Biot specific Frame class."""
import numpy as np

from nova.frame.framelink import FrameLink
from nova.frame.metamethod import CrossSection, Shape, Reduce, PolyGeo


# pylint: disable=too-many-ancestors


class BiotFrame(FrameLink):
    """Extend FrameSpace class with biot specific attributes and methods."""

    _metadata = ["turns", "reduce"]

    def __init__(self, data=None, index=None, columns=None, attrs=None, **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.frame_attrs(PolyGeo, Shape, CrossSection, Reduce)
        for attr in self._metadata:
            setattr(self, attr, None)

    def update_metadata(self, data, columns, attrs, metadata):
        """Extend FrameAttrs update_metadata."""
        metadata = {
            "required": ["x", "z"],
            "additional": ["xo", "zo", "plasma", "nturn", "link", "segment"],
            "available": ["section", "poly", "x1", "y1", "z1", "x2", "y2", "z2"],
            "array": [
                "x",
                "y",
                "z",
                "dx",
                "dy",
                "dz",
                "x1",
                "y1",
                "z1",
                "x2",
                "y2",
                "z2",
                "area",
                "nturn",
            ],
        } | metadata
        super().update_metadata(data, columns, attrs, metadata)

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
        """Return normalized r-coordinate distance from coil centroid."""
        return (self.x - self.xo.values) / self.dx

    @property
    def delta_z(self):
        """Return normalized z-coordinate distance from coil centroid."""
        return (self.z - self.zo.values) / self.dz


class Target(BiotFrame):
    """Extend BiotFrame with modified additional and available metadata."""

    def __init__(self, data=None, index=None, columns=None, attrs=None, **metadata):
        metadata["available"] = []
        metadata["additional"] = ["plasma"]
        super().__init__(data, index, columns, attrs, **metadata)


if __name__ == "__main__":
    biotframe = BiotFrame()
    biotframe.insert(range(3), 0, dl=0.95, dt=0.95, section="hex")
    biotframe.insert(range(3), 1, dl=0.95, dt=0.95, section="disc", link=True)
    biotframe.insert(range(3), 2, dl=0.95, dt=0.95, section="square", link=False)
    biotframe.insert(range(3), 3, dl=0.95, dt=0.6, section="skin", link=True)

    biotframe.multipoint.link(["Coil0", "Coil11", "Coil2", "Coil8"])
