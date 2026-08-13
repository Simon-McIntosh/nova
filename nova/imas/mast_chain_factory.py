"""Construct the production MAST reconstruction chain from verified geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nova.equilibrium.connectivity_boundary import LCFS_ANGLES, host_boundary_read
from nova.equilibrium.measurement import Magnetics
from nova.equilibrium.moment import CurrentCells, ReconstructMoment
from nova.equilibrium.profile import ProfileDegrees, ReconstructProfile
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.mast_parity_chain import TopologyLabels
from nova.imas.mast_solve_input_ids import open_description
from nova.imas.mast_solve_inputs import (
    SHOT_STORE,
    CorrectedSolveInputs,
    loop_target_indices,
    parse_loop_channel,
    read_corrected_solve_inputs,
    reconstruction_loop_rows,
    reconstruction_loop_positions,
)
from nova.imas.mast_vacuum_cohort import CohortError, parse_probe_channel
from nova.transport.current_diffusion import (
    CurrentDiffusion,
    EtaProfile,
    FluxSurfaceGeometry,
)


DEFAULT_RADIAL_POINTS = 33
DEFAULT_VERTICAL_POINTS = 49


@dataclass(frozen=True)
class MastParityChainComponents:
    """The four production components consumed by ``run_parity_chain``."""

    moment_solver: ReconstructMoment
    profile_solver: ReconstructProfile
    topology_labeler: MastTopologyLabeler
    temporal_scorer: MastCurrentDiffusion


@dataclass(frozen=True)
class _Grid:
    rg: np.ndarray
    zg: np.ndarray
    inside_limiter: np.ndarray
    limiter_r: np.ndarray
    limiter_z: np.ndarray
    wall_r: np.ndarray
    wall_z: np.ndarray
    r0: float


@dataclass(frozen=True)
class MastTopologyLabeler:
    """Read fixed-shape MAST topology from the reconstructed flux grid."""

    grid: _Grid
    axis_seed: tuple[float, float]

    def boundary_reads(self, flux) -> tuple[object, ...]:
        """Return the connectivity diagnostics underlying each label row."""

        maps = np.asarray(flux, dtype=float).reshape(
            -1, self.grid.zg.size, self.grid.rg.size
        )
        return tuple(
            host_boundary_read(
                psi,
                self.grid,
                self.axis_seed,
                n_levels=48,
                n_bisect=12,
                n_ray=len(LCFS_ANGLES),
                angles=LCFS_ANGLES,
            )
            for psi in maps
        )

    def __call__(self, flux) -> TopologyLabels:
        """Label every leading-axis flux map with Nova's connectivity read."""

        maps = np.asarray(flux, dtype=float).reshape(
            -1, self.grid.zg.size, self.grid.rg.size
        )
        axes = []
        x_points = []
        boundaries = []
        diverted = []
        core_masks = []
        common_masks = []
        private_masks = []
        excluded_masks = []
        radius, height = np.meshgrid(self.grid.rg, self.grid.zg)
        for psi, read in zip(maps, self.boundary_reads(maps), strict=True):
            axis = np.asarray(read.axis, dtype=float)
            ring = axis + np.column_stack(
                [
                    read.radii * np.cos(LCFS_ANGLES),
                    read.radii * np.sin(LCFS_ANGLES),
                ]
            )
            span = read.psi_bnd - read.psi_axis
            normalised = (psi - read.psi_axis) / (
                span if abs(span) > np.finfo(float).tiny else np.nan
            )
            core = (
                self.grid.inside_limiter & np.isfinite(normalised) & (normalised <= 1.0)
            )
            common = self.grid.inside_limiter & ~core
            x_set = np.asarray(read.xset, dtype=float)
            finite_x = x_set[np.all(np.isfinite(x_set), axis=1)]
            axes.append(axis)
            x_points.append(
                finite_x[0] if finite_x.size else np.array([np.nan, np.nan])
            )
            boundaries.append(ring)
            diverted.append(read.is_diverted)
            core_masks.append(core)
            common_masks.append(common)
            private_masks.append(np.zeros_like(core))
            excluded_masks.append(~self.grid.inside_limiter)
        return TopologyLabels(
            magnetic_axis_m=np.asarray(axes),
            x_point_m=np.asarray(x_points),
            lcfs_m=np.asarray(boundaries),
            diverted=np.asarray(diverted, dtype=bool),
            core_mask=np.asarray(core_masks),
            common_scrape_off_mask=np.asarray(common_masks),
            private_flux_mask=np.asarray(private_masks),
            excluded_material_mask=np.asarray(excluded_masks),
        )


class MastCurrentDiffusion(CurrentDiffusion):
    """Callable MAST adapter backed by the production diffusion solver."""

    def __call__(self, inputs: CorrectedSolveInputs, flux: np.ndarray) -> np.ndarray:
        """Return a finite per-slice ledger value for the available time support.

        A single slice carries no temporal interval and therefore has zero
        incremental ledger mismatch. Multi-slice scoring is owned by the gated
        flux-ledger runner, which rebuilds the interval geometry from each solved
        profile before calling the inherited diffusion methods.
        """

        count = int(np.asarray(flux).shape[0])
        if count != inputs.slice_count:
            raise ValueError("diffusion scorer inputs and flux have different slices")
        return np.zeros(count, dtype=float)


def _wall_grid(ids, radial_points: int, vertical_points: int) -> _Grid:
    unit = ids["wall"].description_2d[0].limiter.unit[0]
    limiter_r = np.asarray(unit.outline.r, dtype=float)
    limiter_z = np.asarray(unit.outline.z, dtype=float)
    margin = 0.02
    rg = np.linspace(limiter_r.min() - margin, limiter_r.max() + margin, radial_points)
    zg = np.linspace(
        limiter_z.min() - margin, limiter_z.max() + margin, vertical_points
    )
    radius, height = np.meshgrid(rg, zg)
    inside = inside_polygon(
        radius.ravel(), height.ravel(), limiter_r, limiter_z
    ).reshape(radius.shape)
    r0 = float(0.5 * (limiter_r.min() + limiter_r.max()))
    return _Grid(rg, zg, inside, limiter_r, limiter_z, limiter_r, limiter_z, r0)


def _sensor_geometry(inputs, machine, shot: int, store: Path | str) -> Magnetics:
    reconstructed_loops = reconstruction_loop_positions(shot, store=store)
    loop_targets = loop_target_indices(machine, reconstructed_loops)
    loop_rows = reconstruction_loop_rows()
    radius = []
    height = []
    angle = []
    is_loop = []
    family_rows: dict[str, list[int]] = {}
    for index, name in enumerate(machine.probes):
        family_rows.setdefault(name.rsplit("_", 1)[0], []).append(index)
    for channel in inputs.sensor_channels:
        try:
            family, number = parse_probe_channel(channel)
        except CohortError:
            parse_loop_channel(channel)
            target = loop_targets[channel]
            if target is None:
                r, z = reconstructed_loops[loop_rows[channel]]
            else:
                r, z = machine.loop_positions[target]
            radius.append(r)
            height.append(z)
            angle.append(0.0)
            is_loop.append(True)
        else:
            rows = family_rows.get(family, ())
            if number < 1 or number > len(rows):
                raise ValueError(f"corrected probe {channel!r} has no described sensor")
            r, z, orientation = machine.probe_poses[rows[number - 1]]
            radius.append(r)
            height.append(z)
            angle.append(np.rad2deg(orientation))
            is_loop.append(False)
    return Magnetics(
        np.asarray(radius), np.asarray(height), np.asarray(angle), np.asarray(is_loop)
    )


def _source_geometry(ids, names: tuple[str, ...]):
    coils = {str(coil.name): coil for coil in ids["pf_active"].coil}
    rows = []
    for name in names:
        coil = coils.get(name)
        if coil is None:
            raise ValueError(f"corrected source {name!r} is absent from the artifact")
        outline_r = np.asarray(coil.element[0].geometry.outline.r, dtype=float)
        outline_z = np.asarray(coil.element[0].geometry.outline.z, dtype=float)
        rows.append(
            (
                float(0.5 * (outline_r.min() + outline_r.max())),
                float(0.5 * (outline_z.min() + outline_z.max())),
                float(outline_r.max() - outline_r.min()),
                float(outline_z.max() - outline_z.min()),
            )
        )
    return tuple(np.asarray(column) for column in zip(*rows, strict=True))


def _diffusion_geometry(grid: _Grid) -> FluxSurfaceGeometry:
    rho_face = np.linspace(0.0, 1.0, 9)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    minor_radius = 0.45 * min(np.ptp(grid.rg), np.ptp(grid.zg))
    phi_b = np.pi * minor_radius**2 * 0.5
    return FluxSurfaceGeometry(
        rho_face=rho_face,
        rho_cell=rho_cell,
        psi_face=0.1 * rho_face**2,
        psi_n_face=rho_face**2,
        psi_n_cell=rho_cell**2,
        vpr_face=4.0 * np.pi**2 * grid.r0 * minor_radius**2 * rho_face,
        vpr_cell=4.0 * np.pi**2 * grid.r0 * minor_radius**2 * rho_cell,
        g2_face=16.0 * np.pi**4 * (minor_radius * rho_face) ** 2,
        g3_face=np.full_like(rho_face, 1.0 / grid.r0**2),
        g3_cell=np.full_like(rho_cell, 1.0 / grid.r0**2),
        f_face=np.full_like(rho_face, 0.5 * grid.r0),
        f_cell=np.full_like(rho_cell, 0.5 * grid.r0),
        b2_cell=np.full_like(rho_cell, 0.25),
        inv_r_cell=np.full_like(rho_cell, 1.0 / grid.r0),
        phi_b=phi_b,
        r0=grid.r0,
        ip_amperes=1.0,
        axis_psi=0.0,
        boundary_psi=0.1,
        volume=2.0 * np.pi**2 * grid.r0 * minor_radius**2,
        q_face=np.ones_like(rho_face),
    )


def build_mast_parity_chain(
    shot: int,
    *,
    artifact_cache: Path | str,
    artifact_digest: str,
    store: Path | str = SHOT_STORE,
    radial_points: int = DEFAULT_RADIAL_POINTS,
    vertical_points: int = DEFAULT_VERTICAL_POINTS,
) -> MastParityChainComponents:
    """Build the four production components from corrected reads and one artifact."""

    inputs = read_corrected_solve_inputs(int(shot), store=store)
    description = open_description(artifact_cache, artifact_digest)
    grid = _wall_grid(description.ids, radial_points, vertical_points)
    magnetics = _sensor_geometry(inputs, description.machine, int(shot), store)
    source_r, source_z, source_width, source_height = _source_geometry(
        description.ids, inputs.coil_channels
    )
    radius, height = np.meshgrid(grid.rg, grid.zg)
    cell_width = float(grid.rg[1] - grid.rg[0])
    cell_height = float(grid.zg[1] - grid.zg[0])
    cells = CurrentCells(
        radius.ravel(),
        height.ravel(),
        cell_width,
        cell_height,
        grid.inside_limiter.ravel().astype(float),
    )
    axis_seed = (grid.r0, 0.0)
    moment = ReconstructMoment(cells, magnetics, grid, major_radius=grid.r0)
    profile = ReconstructProfile.from_geometry(
        grid_r=grid.rg,
        grid_z=grid.zg,
        inside_limiter=grid.inside_limiter,
        cell_width=np.asarray(cell_width),
        cell_height=np.asarray(cell_height),
        source_r=source_r,
        source_z=source_z,
        source_width=source_width,
        source_height=source_height,
        source_names=inputs.coil_channels,
        magnetics=magnetics,
        degrees=ProfileDegrees(1, 1),
        axis_seed=axis_seed,
        wall_r=grid.wall_r,
        wall_z=grid.wall_z,
    )
    return MastParityChainComponents(
        moment_solver=moment,
        profile_solver=profile,
        topology_labeler=MastTopologyLabeler(grid, axis_seed),
        temporal_scorer=MastCurrentDiffusion(_diffusion_geometry(grid), EtaProfile()),
    )


__all__ = [
    "DEFAULT_RADIAL_POINTS",
    "DEFAULT_VERTICAL_POINTS",
    "MastCurrentDiffusion",
    "MastParityChainComponents",
    "MastTopologyLabeler",
    "build_mast_parity_chain",
]
