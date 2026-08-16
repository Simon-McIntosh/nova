"""Biot-Savart calculation base class."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from tqdm import tqdm
import xarray

from nova.biot.arc import Arc
from nova.biot.beam import Beam
from nova.biot.biotframe import Source
from nova.biot.bow import Bow
from nova.biot.circle import Circle
from nova.biot.cylinder import Cylinder
from nova.biot.line import Line
from nova.biot.polybeam import PolyBeam
from nova.biot.polybow import PolyBow
from nova.biot.polysection import (
    DEFAULT_POLYSECTION_POLICY,
    PolySection,
    PolySectionPolicy,
    TiledPolySection,
)
from nova.biot.groupset import GroupSet


_PLASMA_MOMENT_ATTRIBUTES = frozenset({"PsiR", "PsiZ", "BrR", "BrZ", "BzR", "BzZ"})


@dataclass(frozen=True)
class SourceBatch:
    """One bounded source evaluation with a complete immutable route identity."""

    segment: str
    lane: str
    positions: tuple[int, ...]
    ordinal: int
    policy: PolySectionPolicy | None = None

    @property
    def identity(self) -> str:
        """Return the stable identity used for diagnostics and batch separation."""
        policy = "not_applicable" if self.policy is None else self.policy.key
        return f"{self.segment}|{self.lane}|{policy}|{self.ordinal}"


@dataclass
class Solve(GroupSet):
    """Manage biot interaction between multiple filament types."""

    name: str = "biot"
    attrs: list[str] = field(default_factory=lambda: ["Aphi", "Psi", "Br", "Bz"])
    source_segment: np.ndarray = field(init=False, repr=False)
    source_batches: list[SourceBatch] = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)

    generator: ClassVar[dict] = {
        "arc": Arc,
        "beam": Beam,
        "bow": Bow,
        "circle": Circle,
        "cylinder": Cylinder,
        "line": Line,
        "polybeam": PolyBeam,
        "polybow": PolyBow,
        "polysection": PolySection,
    }

    def __post_init__(self):
        """Initialise dataset and compute biot interaction."""
        super().__post_init__()
        self.check_segments()
        self.initialize()
        self.compose()
        # self.decompose()

    def check_segments(self):
        """Split each kernel segment into chunks of at most 500 sources.

        ``source_segment`` is a per-source label array; each kernel segment is
        partitioned into sub-batches named ``<segment>_<i>`` so the interaction
        matrices stay bounded.
        """
        segment_column = np.asarray(self.source["segment"])
        plasma_column = np.asarray(self.source.plasma, dtype=bool)
        policy_column = (
            np.asarray(self.source["polysection_policy"], dtype=object)
            if "polysection_policy" in self.source
            else np.full(len(self.source), DEFAULT_POLYSECTION_POLICY, dtype=object)
        )
        grouped: dict[tuple[str, str, str], list[int]] = {}
        for position, (segment, plasma, policy_value) in enumerate(
            zip(segment_column, plasma_column, policy_column)
        ):
            if segment not in self.generator:
                raise NotImplementedError(
                    f"segment <{segment}> not implemented "
                    f"in Biot.generator: {self.generator.keys()}"
                )
            lane = "plasma" if plasma else "conductor"
            policy_key = (
                PolySectionPolicy.resolve(policy_value).key
                if segment == "polysection"
                else "not_applicable"
            )
            grouped.setdefault((str(segment), lane, policy_key), []).append(position)

        self.source_batches = []
        self.source_segment = np.empty(len(self.source), dtype=object)
        for (segment, lane, policy_key), positions in grouped.items():
            for i in range(0, len(positions), 500):
                batch_positions = tuple(positions[i : i + 500])
                batch = SourceBatch(
                    segment=segment,
                    lane=lane,
                    positions=batch_positions,
                    ordinal=i // 500,
                    policy=(
                        PolySectionPolicy.resolve(policy_key)
                        if segment == "polysection"
                        else None
                    ),
                )
                self.source_batches.append(batch)
                self.source_segment[list(batch_positions)] = batch.identity

    def initialize(self):
        """Initialize dataset."""
        source_plasma = np.any(self.source.plasma)
        target_plasma = np.any(self.logical_target.plasma)
        self.data = xarray.Dataset(
            coords=dict(
                source=self.get_index("source"),
                target=self.get_index("target"),
                source_plasma=np.asarray(self.source.index)[
                    np.asarray(self.source.plasma)
                ].tolist(),
                target_plasma=np.asarray(self.logical_target.index)[
                    np.asarray(self.logical_target.plasma)
                ].tolist(),
            )
        )
        self.data.attrs["attributes"] = self.attrs
        self.data.attrs["source_route_batches"] = [
            batch.identity for batch in self.source_batches
        ]
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
        if frame == "target" and self.target_quadrature is not None:
            parent_positions = np.flatnonzero(
                np.asarray(self.target_quadrature.physical_plasma, dtype=bool)
            )
            if len(parent_positions) == 0:
                return -1
            if len(parent_positions) > 1:
                raise ValueError(
                    "plasma-turn updates currently require one physical target "
                    "parent; multiple parents need an explicit parent-row "
                    "contraction"
                )
            return int(parent_positions[0])
        biotframe = self.logical_target if frame == "target" else self.source
        plasma = np.asarray(biotframe.aloc["plasma"])
        names = np.unique(np.asarray(biotframe["frame"])[plasma])
        try:
            return next(biotframe.subspace.index.get_loc(name) for name in names)
        except StopIteration:
            return -1

    def get_index(self, frame: str) -> list[str]:
        """Return matrix coordinate, reduce if flag True."""
        biotframe = self.logical_target if frame == "target" else self.source
        if (
            frame == "target"
            and self.target_quadrature is not None
            and biotframe.reduce
        ):
            return list(self.target_quadrature.physical_index)
        if biotframe.reduce:
            return list(biotframe.biotreduce.index)
        return biotframe.index.to_list()

    def compose(self):
        """Calculate full ensemble biot interaction."""
        for batch in tqdm(self.source_batches, ncols=65, desc=self.name):
            self.compute(batch)

    @cached_property
    def _frame_link(self):
        """Return {label: linked label} map, empty links resolving to self."""
        frame = self.source.biotreduce.frame
        labels = np.asarray(frame.index)
        link = np.asarray(frame["link"]).astype(object)
        return {
            label: (label if target == "" else target)
            for label, target in zip(labels, link)
        }

    def source_index(self, batch: SourceBatch):
        """Return source segment index."""
        segment_mask = np.zeros(len(self.source), dtype=bool)
        segment_mask[list(batch.positions)] = True
        if not self.source.reduce:
            return np.asarray(self.source.index)[segment_mask].tolist()
        frame = [
            self._frame_link[label]
            for label in np.unique(np.asarray(self.source["frame"])[segment_mask])
        ]
        source_index = np.asarray(self.get_index("source"))
        return source_index[np.isin(source_index, frame)].tolist()

    def plasma_index(self, batch: SourceBatch):
        """Return plasma segment index."""
        positions = np.asarray(batch.positions, dtype=int)
        plasma = np.asarray(self.source.plasma, dtype=bool)[positions]
        return np.asarray(self.source.index)[positions[plasma]].tolist()

    @cached_property
    def _source_reduction(self) -> tuple[np.ndarray, np.ndarray]:
        """Map every raw source column to one final column and link factor.

        Kernel batches are independent of electrical topology, so the reduction is
        derived once from the complete source frame and applied only after each raw
        batch returns. This keeps a dependent source's factor even when its reference
        belongs to another segment, policy, plasma lane, or 500-column chunk.
        """
        groups = [[(position, 1.0)] for position in range(len(self.source))]
        reduction = self.source.biotreduce
        if self.source.reduce and reduction.reduce:
            starts = np.asarray(reduction.indices, dtype=int)
            stops = np.r_[starts[1:], len(self.source)]
            groups = [
                [(position, 1.0) for position in range(start, stop)]
                for start, stop in zip(starts, stops)
            ]
        links = reduction.link if self.source.reduce else {}
        for link, (reference, factor) in links.items():
            groups[reference].extend(
                (position, coefficient * factor)
                for position, coefficient in groups[link]
            )
        groups = [group for index, group in enumerate(groups) if index not in links]

        output = np.empty(len(self.source), dtype=int)
        coefficient = np.empty(len(self.source), dtype=float)
        for final_position, group in enumerate(groups):
            for source_position, factor in group:
                output[source_position] = final_position
                coefficient[source_position] = factor
        if len(groups) != len(self.get_index("source")):
            raise ValueError("source reduction does not match the physical index")
        return output, coefficient

    def _reduce_source_columns(
        self, values: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Accumulate raw batch columns through the complete source topology."""
        output, coefficient = self._source_reduction
        reduced = np.zeros((values.shape[0], len(self.get_index("source"))))
        np.add.at(
            reduced.T,
            output[positions],
            (values * coefficient[positions][np.newaxis, :]).T,
        )
        return reduced

    def compute(self, batch: SourceBatch):
        """Compute segment and update dataset."""
        plasma_index = self.plasma_index(batch)
        positions = np.asarray(batch.positions, dtype=int)
        # slice by column but keep the row labels: link stores row labels, so
        # a label-less rebuild re-derives ref against fresh auto-labels and can
        # split one coil's filaments across reduction indices
        source = Source(
            {
                col: np.asarray(self.source[col])[positions]
                for col in self.source.columns
            },
            index=list(np.asarray(self.source.index)[positions]),
        )
        generator_kwargs = {"policy": batch.policy} if batch.policy is not None else {}
        generator_class = self.generator[batch.segment]
        if batch.policy is not None and batch.policy.backend == "jax":
            generator_class = TiledPolySection
        generator = generator_class(
            source,
            self.target,
            turns=self.turns,
            reduce=[False, self.reduce[1]],
            target_quadrature=self.target_quadrature,
            **generator_kwargs,
        )
        for attr in self.attrs:
            if attr in _PLASMA_MOMENT_ATTRIBUTES and batch.lane != "plasma":
                continue
            matrix, target_plasma, plasma_source, plasma_plasma = generator.compute(
                attr
            )
            self.data[attr].data += self._reduce_source_columns(matrix, positions)
            if np.prod(target_plasma.shape) > 0:
                self.data[f"{attr}_"].loc[:, plasma_index] += target_plasma
            if np.prod(plasma_source.shape) > 0:
                self.data[f"_{attr}"].data += self._reduce_source_columns(
                    plasma_source, positions
                )
            if np.prod(plasma_plasma.shape) > 0:
                self.data[f"_{attr}_"].loc[:, plasma_index] += plasma_plasma

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
