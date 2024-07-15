"""Extend pandas.DataFrame to manage coil and subcoil data."""

from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
from typing import ClassVar

import netCDF4
import pandas

from nova.database.netcdf import netCDF
from nova.frame.framedata import FrameData
from nova.frame.framesetloc import FrameSetLoc
from nova.frame.framespace import FrameSpace
from nova.frame.select import Select


def frame_factory(frame_method):
    """Automatic loader for frame methods."""

    def decorator(method):
        """Return initialized frame method."""

        @cached_property
        def wrapper(self):
            nonlocal frame_method
            kwargs = {"name": method.__name__} | method(self)
            try:
                return frame_method(*self.frames, **kwargs)
            except TypeError:  # import_module from DeferredImport.load()
                frame_method = frame_method.load()
                return frame_method(*self.frames, **kwargs)

        return wrapper

    return decorator


@dataclass
class FrameSet(netCDF, FrameSetLoc):
    """Manage FrameSet instances."""

    base: list[str] = field(
        repr=False,
        default_factory=lambda: [
            "x",
            "y",
            "z",
            "r",
        ],
    )
    required: list[str] = field(repr=False, default_factory=lambda: [])
    additional: list[str] = field(
        repr=False,
        default_factory=lambda: [
            "x0",
            "y0",
            "z0",
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
            "dx",
            "dy",
            "dz",
            "turn",
            "frame",
            "plasma",
            "Ic",
            "nturn",
            "Imin",
            "Imax",
            "vtk",
        ],
    )
    available: list[str] = field(repr=False, default_factory=list)
    subspace: list[str] = field(repr=False, default_factory=lambda: ["Ic", "part"])
    array: list[str] = field(
        repr=False,
        default_factory=lambda: [
            "Ic",
            "nturn",
            "active",
            "passive",
            "plasma",
            "coil",
            "fix",
            "free",
            "ferritic",
            "area",
            "volume",
        ],
    )
    vtk: bool = True

    _available: ClassVar[list[str]] = [
        "link",
        "part",
        "frame",
        "dx",
        "dy",
        "dz",
        "area",
        "volume",
        "vtk",
        "delta",
        "section",
        "turn",
        "scale",
        "nturn",
        "nfilament",
        "Ic",
        "It",
        "Psi",
        "Bx",
        "Bz",
        "B",
        "acloss",
    ]

    def __post_init__(self):
        """Create frame and subframe."""
        self.available = list(dict.fromkeys(self.available + self._available))
        self.frame = FrameSpace(
            base=self.base,
            required=self.required,
            additional=self.additional,
            available=self.available,
            subspace=["Imin", "Imax"],
            exclude=[
                "frame",
                "Ic",
                "It",
                "fix",
                "free",
            ],
            array=["coil"],
            version=["index"],
        )
        self.frame.frame_attr(Select)
        self.subframe = FrameSpace(
            base=self.base,
            required=self.required,
            additional=self.additional,
            available=self.available,
            subspace=self.subspace,
            exclude=["turn", "scale", "nfilament", "delta"],
            array=self.array,
            delim="_",
            version=["index", "nturn"],
        )
        self.subframe.frame_attr(Select, ["Ic"])
        super().__post_init__()

    def __str__(self):
        """Return string representation of coilset frame."""
        return str(self.superframe)

    @property
    def superframe(self):
        """Return descriptive superframe including net coil currents."""
        columns = [
            col for col in ["link", "part", "segment", "nturn"] if col in self.frame
        ]
        superframe = pandas.DataFrame(self.Loc[:, columns])
        superframe["Ic"] = self.sloc["Ic"][self.frame.subref]
        superframe["It"] = superframe["Ic"] * superframe["nturn"]
        return superframe

    @staticmethod
    def import_method(name: str, package: str | None):
        """Return method imported from dot seperated module lookup."""
        module_name = ".".join(name.split(".")[:-1])
        method_name = name.split(".")[-1]
        module = import_module(module_name, package=package)
        return getattr(module, method_name)

    def clear_frameset(self):
        """Clear all frameset instances."""
        delattrs = []
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), FrameData):
                delattrs.append(attr)
        for attr in delattrs:
            delattr(self, attr)

    def subset(self, dataset):
        """Return group from dataset."""
        if self.group is not None:
            return dataset[self.group]
        return dataset

    def load(self):
        """Load frameset from file."""
        self.frame.load(self.filepath, self.subgroup("frame"))
        self.subframe.load(self.filepath, self.subgroup("subframe"))
        self.clear_frameset()
        with netCDF4.Dataset(self.filepath) as dataset:
            dataset = self.subset(dataset)
            for attr in dataset.groups:
                if attr in dir(self.__class__) and isinstance(
                    data := getattr(self, attr), netCDF
                ):
                    data.filepath = self.filepath
                    data.group = self.subgroup(data.name)
                    data.load()
        super().load()
        return self

    def store(self, vtk=True):
        """Store frame, subframe and methods as groups within netCDF file."""
        self.frame.store(self.filepath, self.subgroup("frame"), self.get_mode(), vtk)
        self.subframe.store(self.filepath, self.subgroup("subframe"), "a", vtk)
        for attr in self.__dict__:
            data = getattr(self, attr)
            if isinstance(data, netCDF) and isinstance(data, FrameData):
                data.filepath = self.filepath
                data.group = self.subgroup(data.name)
                data.store()
        super().store()
        return self

    def plot(self, index=None, axes=None, **kwargs):
        """Plot coilset subframe via polyplot instance."""
        self.subframe.polyplot(index=index, axes=axes, **kwargs)
        if hasattr(super(), "plot"):
            super().plot(axes=axes, **kwargs)


if __name__ == "__main__":
    frameset = FrameSet(required=["rms"], additional=["Ic"])
    frameset.subframe.insert([2, 4], It=6, link=True)
