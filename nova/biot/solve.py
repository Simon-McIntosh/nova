"""Biot-Savart calculation base class."""

from dataclasses import dataclass, field
from functools import cached_property
from itertools import zip_longest
from typing import ClassVar

import numpy as np
from tqdm import tqdm
import xarray

from nova.biot.arc import Arc
from nova.biot.beam import Beam
from nova.biot.bow import Bow
from nova.biot.circle import Circle
from nova.biot.cylinder import Cylinder
from nova.biot.line import Line
from nova.biot.polygon import Polygon
from nova.biot.groupset import GroupSet


@dataclass
class Solve(GroupSet):
    """Manage biot interaction between multiple filament types."""

    name: str = "biot"
    attrs: list[str] = field(default_factory=lambda: ["Aphi", "Psi", "Br", "Bz"])
    source_segment: np.ndarray = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)

    generator: ClassVar[dict] = {
        "arc": Arc,
        "beam": Beam,
        "bow": Bow,
        "circle": Circle,
        "cylinder": Cylinder,
        "line": Line,
        "polygon": Polygon,
    }

    def __post_init__(self):
        """Initialise dataset and compute biot interaction."""
        super().__post_init__()
        self.check_segments()
        self.initialize()
        self.compose()
        # self.decompose()

    def check_segments(self):
        """Check for segment in self.generator."""
        self.source_segment = self.source.segment.copy()
        for segment in self.source_segment.unique():
            if segment not in self.generator:
                raise NotImplementedError(
                    f"segment <{segment}> not implemented "
                    f"in Biot.generator: {self.generator.keys()}"
                )
            index = self.source.index[self.source_segment == segment]
            for i, chunk in enumerate(self.group_segments(index, 50, index[-1])):
                self.source_segment.loc[list(chunk)] = f"{segment}_{i}"

    @staticmethod
    def group_segments(iterable, length, fillvalue):
        """Return grouped itterable."""
        length = min([length, len(iterable)])
        args = length * [iter(iterable)]
        return zip_longest(*args, fillvalue=fillvalue)

    def initialize(self):
        """Initialize dataset."""
        source_plasma = np.any(self.source.plasma)
        target_plasma = np.any(self.target.plasma)
        self.data = xarray.Dataset(
            coords=dict(
                source=self.get_index("source"),
                target=self.get_index("target"),
                source_plasma=self.source.index[self.source.plasma].to_list(),
                target_plasma=self.target.index[self.target.plasma].to_list(),
            )
        )
        self.data.attrs["attributes"] = self.attrs
        for row, col, prefix, postfix in zip(
            ["target", "target", "target_plasma", "target_plasma"],
            ["source", "source_plasma", "source", "source_plasma"],
            ["", "", "_", "_"],
            ["", "_", "", "_"],
        ):
            if row == "target_plasma" and not target_plasma:
                continue
            if col == "source_plasma" and not source_plasma:
                continue

            for attr in self.attrs:
                self.data[f"{prefix}{attr}{postfix}"] = xarray.DataArray(
                    0.0, dims=[row, col], coords=[self.data[row], self.data[col]]
                )

        for frame in ["source", "target"]:
            self.data.attrs[f"{frame}_plasma_index"] = self.get_plasma_index(frame)

        # self._initialize_svd('target', 'source')
        # self._initialize_svd('target', 'plasma', prefix='_')
        """
        if self.data.sizes['plasma'] < self.data.sizes['target']:
            sigma = 'plasma'
        else:
            sigma = 'target'
        for attr in self.attrs:  # unit filament svd matricies
            self.data[f'_U{attr}'] = xarray.DataArray(
                0., dims=['target', sigma],
                coords=[self.data.target, self.data[sigma]])
            self.data[f'_s{attr}'] = xarray.DataArray(
                0., dims=[sigma], coords=[self.data[sigma]])
            self.data[f'_V{attr}'] = xarray.DataArray(
                0., dims=[sigma, 'plasma'],
                coords=[self.data[sigma], self.data.plasma])
        """

    def _initialize_svd(self, row: str, column: str, prefix=""):
        """Initialize svd data structures."""
        if self.data.sizes[column] < self.data.sizes[row]:
            sigma = column
        else:
            sigma = row
        for attr in self.attrs:  # unit filament svd matricies
            self.data[f"{prefix}{attr}_U"] = xarray.DataArray(
                0.0, dims=[row, sigma], coords=[self.data[row], self.data[sigma]]
            )
            self.data[f"{prefix}{attr}_s"] = xarray.DataArray(
                0.0, dims=[sigma], coords=[self.data[sigma]]
            )
            self.data[f"{prefix}{attr}_V"] = xarray.DataArray(
                0.0, dims=[sigma, column], coords=[self.data[sigma], self.data[column]]
            )

    def get_plasma_index(self, frame: str) -> int:
        """Return frame plasma index."""
        biotframe = getattr(self, frame)
        try:
            return next(
                biotframe.subspace.index.get_loc(name)
                for name in biotframe.frame[biotframe.aloc["plasma"]].unique()
            )
        except StopIteration:
            return -1

    def get_index(self, frame: str) -> list[str]:
        """Return matrix coordinate, reduce if flag True."""
        biotframe = getattr(self, frame)
        if biotframe.reduce:
            return biotframe.biotreduce.index.to_list()
        return biotframe.index.to_list()

    def compose(self):
        """Calculate full ensemble biot interaction."""
        for segment in tqdm(self.source_segment.unique(), ncols=65, desc=self.name):
            self.compute(segment)

    @cached_property
    def _frame_link(self):
        """Return frame link."""
        link = self.source.biotreduce.frame.link.copy()
        link.loc[link == ""] = link.index[link == ""]
        return link

    def source_index(self, segment):
        """Return source segment index."""
        frame = [
            self._frame_link.loc[frame]
            for frame in np.unique(self.source.frame[self.source_segment == segment])
        ]
        return np.isin(self.get_index("source"), frame)

    def plasma_index(self, segment):
        """Return plasma segment index."""
        plasma = self.source_segment[self.source.index[self.source.plasma]]
        return np.array(plasma == segment)

    def compute(self, segment: str):
        """Compute segment and update dataset."""
        source_index = self.source_index(segment)
        plasma_index = self.plasma_index(segment)
        generator = self.generator[segment.split("_")[0]](
            self.source.loc[self.source_segment == segment, :].to_dict(),
            self.target,
            turns=self.turns,
            reduce=self.reduce,
        )
        for attr in self.attrs:
            matrix, target_plasma, plasma_source, plasma_plasma = generator.compute(
                attr
            )
            self.data[attr].loc[:, source_index] += matrix
            if np.prod(target_plasma.shape) > 0:
                self.data[f"{attr}_"].loc[:, plasma_index] += target_plasma
            if np.prod(plasma_source.shape) > 0:
                self.data[f"_{attr}"].loc[:, source_index] += plasma_source
            if np.prod(plasma_plasma.shape) > 0:
                self.data[f"_{attr}_"].data[:, plasma_index] += plasma_plasma

    def decompose(self):
        """Compute plasma svd and update dataset."""
        for source, prefix in zip(["source", "plasma"], ["", "_"]):
            if self.data.sizes[source] < self.data.sizes["target"]:
                sigma = source
            else:
                sigma = "target"
            for attr in self.attrs:
                matrix = self.data[f"{prefix}{attr}"]
                UsV = np.linalg.svd(matrix, full_matrices=False)
                self.data[f"{prefix}{attr}_U"] = ("target", sigma), UsV[0]
                self.data[f"{prefix}{attr}_s"] = sigma, UsV[1]
                self.data[f"{prefix}{attr}_V"] = (sigma, source), UsV[2]
