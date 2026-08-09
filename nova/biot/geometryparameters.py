"""Named deformations of a winding pack, and the sensor-space geometry Jacobian.

The coupling kernels differentiate with respect to raw section vertices
(:mod:`nova.biot.polygon`, :mod:`nova.biot.tiledassembly`).  An inverse solve for
a coil's geometry wants one step up from that: a short parameter vector whose
components are deformations a machine can actually suffer -- the pack sitting a
few millimetres off its drawing, wound a per cent large, tilted, or with one
turn out of line -- and the Jacobian of the sensor readings with respect to it.

WHAT A PARAMETER IS.  Every block here is a LINEAR displacement field over the
pack's vertices, so a parameter vector maps to vertex offsets by one static
basis tensor and the parameter space is a vector space.  That matters because
the identifiability question is answered by singular values, and singular values
are only meaningful once the parameter metric is fixed: the metric here is the
pack's root-mean-square vertex displacement in metres, and every generator is
normalised to carry exactly one metre of it.  A whitened Jacobian's singular
values are then signal-to-noise per metre and their reciprocals are the
displacements the array resolves, independent of how the generators are scaled.

The blocks, at increasing freedom -- ``rigid`` (2) is contained in the affine
set ``rigid + dilation + stretch + shear + tilt`` (6), which is contained in
``vertex`` (2 per vertex); ``section`` (2 per section) frees each turn of a
lattice on its own and also contains ``rigid``.  The named affine generators are
NOT mutually orthogonal -- dilation and stretch overlap unless the pack is
square -- so the metric travels with them as a Gram matrix rather than being
assumed to be the identity.

Tilt is the rotation's GENERATOR rather than a finite rotation.  At the
linearisation point the two agree to first order, which is what a Jacobian is;
keeping the map linear is what makes the metric unambiguous.  The four
non-rigid affine generators are taken about the pack's vertex mean, which fixes
each one individually without changing what the set spans -- moving the centre
only pours some rigid displacement into them, and the rigid pair is in the set.

WHY THE JACOBIAN IS NOT TAKEN AT THE PACK'S OWN CONFIGURATION.  The polygon
kernel integrates each section edge in z: the edge is parametrised
``r'(u) = r1 + b1 u`` with ``u = z' - z``, so an edge of constant z spans an
interval of zero length.  Its contribution is exactly zero -- correctly, and
:func:`~nova.biot.polygon.pack_section` marks it so the slope ``b1`` never
divides by zero -- but a zero-weight edge also carries no DERIVATIVE, and a
deformation that tilts such an edge out of the horizontal has a first-order
effect the pack's own parametrisation cannot express.  A pack of axis-aligned
turns is made entirely of these edges, and taken in place its tilt and shear
derivatives are not the derivative of anything: measured against the shipped
kernel's own central difference they miss by 5 to 26 times the column's own
peak, sign included.

Rigid shifts and the two axis-aligned scalings keep a flat edge flat, so their
derivatives are exact in place (measured to 1e-8).  For the rest the Jacobian is
taken at a pair of
opposite small tilts of the whole pack, where no edge is flat and every edge
differentiates, and averaged: the odd term in the tilt cancels and what is left
is the true Jacobian to second order in the offset.  :data:`FLAT_EDGE_TILT` is
the offset, chosen on the plateau between the second-order bias it introduces
and the cancellation a nearly-flat edge suffers in the bracket over its own
u-limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from nova.biot.polygon import (
    _phi_rule,
    horizontal_edges,
    polygon_greens,
    traced_pack_section,
)
from nova.jax.config import Precision, resolve_precision
from nova.biot.tiledassembly import _traced_psi_gradient

# The full-turn quadrature rule the shipped kernel defaults to.
DEFAULT_RULE = (16, 48)

# Half-separation of the two evaluation tilts, in radians.  The pair average's
# bias falls as the square of the offset until it reaches the reference's own
# floor, and below ~1e-6 the nearly-flat edge loses its bracket to cancellation
# and the error climbs again: measured against the shipped kernel's central
# difference, 1e-2 costs 2e-4 of a column's peak, 1e-3 costs 2e-6, and the
# plateau runs from 1e-5 to 1e-4 at 1e-8 to 1e-7 before 1e-7 returns 7e-6.  This
# sits at the low-bias end of that plateau, two decades clear of the climb.
FLAT_EDGE_TILT = 1e-4

AFFINE_BLOCKS = ("rigid", "dilation", "stretch", "shear", "tilt")
BLOCKS = AFFINE_BLOCKS + ("section", "vertex")


@dataclass(frozen=True)
class SensorArray:
    """Poloidal-plane channels, each reading one combination of the kernel rows.

    ``projection`` is ``(n, 3)`` of weights on ``(psi, B_R, B_Z)``, so a field
    probe carries its sensitive axis ``(0, cos, sin)`` and a flux loop carries
    ``(1, 0, 0)``.  One contraction serves both, which keeps the traced path free
    of a channel-type branch and lets a mixed array be one Jacobian.
    """

    r: np.ndarray
    z: np.ndarray
    projection: np.ndarray

    @classmethod
    def field_probes(cls, r, z, cos, sin) -> SensorArray:
        """Return probes reading ``cos B_R + sin B_Z`` along a sensitive axis."""
        cos = np.asarray(cos, dtype=float)
        return cls(
            np.asarray(r, dtype=float),
            np.asarray(z, dtype=float),
            np.column_stack([np.zeros_like(cos), cos, np.asarray(sin, dtype=float)]),
        )

    @classmethod
    def flux_loops(cls, r, z) -> SensorArray:
        """Return loops reading the total poloidal flux through their contour."""
        r = np.asarray(r, dtype=float)
        return cls(
            r,
            np.asarray(z, dtype=float),
            np.column_stack([np.ones_like(r), np.zeros_like(r), np.zeros_like(r)]),
        )

    def join(self, other: SensorArray) -> SensorArray:
        """Return the two arrays as one, in order."""
        return SensorArray(
            np.concatenate([self.r, other.r]),
            np.concatenate([self.z, other.z]),
            np.concatenate([self.projection, other.projection]),
        )

    def take(self, keep) -> SensorArray:
        """Return the channels ``keep`` selects, as a smaller array."""
        keep = np.asarray(keep)
        return SensorArray(self.r[keep], self.z[keep], self.projection[keep])

    def __len__(self) -> int:
        return self.r.size


@dataclass(frozen=True)
class PackDeformation:
    """A pack's base geometry and the linear parameter blocks acting on it.

    ``sections`` is ``(S, N, 2)`` -- ``S`` polygons of ``N`` corners, one per
    turn of a lattice or one for a whole outline -- and ``generators`` is
    ``(P, S, N, 2)``, each row a displacement field of unit root-mean-square
    vertex displacement.  The deformation is that basis contracted with the
    parameter vector, so both the map and its derivative are exact.
    """

    sections: np.ndarray
    labels: tuple[str, ...]
    generators: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of parameters."""
        return len(self.labels)

    def offsets(self, xp, parameters):
        """Return the ``(S, N, 2)`` vertex displacement of a parameter vector."""
        parameters = xp.asarray(parameters)
        return xp.tensordot(
            parameters, xp.asarray(self.generators, dtype=parameters.dtype), 1
        )

    def deform(self, xp, parameters):
        """Return the deformed sections, base plus displacement."""
        parameters = xp.asarray(parameters)
        return xp.asarray(self.sections, dtype=parameters.dtype) + self.offsets(
            xp, parameters
        )

    def gram(self) -> np.ndarray:
        """Return the parameter metric: mean vertex-wise generator overlap.

        The diagonal is one by construction; the off-diagonal is what stops a
        singular spectrum from reading the generators' own non-orthogonality as
        structure in the data.
        """
        flat = self.generators.reshape(self.size, -1)
        return flat @ flat.T / (self.sections.shape[0] * self.sections.shape[1])


def _normalise(generators: np.ndarray) -> np.ndarray:
    """Return generators scaled to one metre of pack root-mean-square offset."""
    count = generators.shape[1] * generators.shape[2]
    scale = np.sqrt(np.sum(generators**2, axis=(1, 2, 3)) / count)
    return generators / scale[:, None, None, None]


def _block_generators(sections: np.ndarray, block: str):
    """Return one block's raw generators and labels for ``sections``."""
    count, corners = sections.shape[:2]
    local = sections - sections.reshape(-1, 2).mean(axis=0)
    ones = np.ones(sections.shape[:2])
    zeros = np.zeros(sections.shape[:2])
    match block:
        case "rigid":
            return (
                np.stack(
                    [
                        np.stack([ones, zeros], axis=-1),
                        np.stack([zeros, ones], axis=-1),
                    ]
                ),
                ("rigid_r", "rigid_z"),
            )
        case "dilation":
            return local[None], ("dilation",)
        case "stretch":
            return (
                np.stack([local[..., 0], -local[..., 1]], axis=-1)[None],
                ("stretch",),
            )
        case "shear":
            return (
                np.stack([local[..., 1], local[..., 0]], axis=-1)[None],
                ("shear",),
            )
        case "tilt":
            return (
                np.stack([-local[..., 1], local[..., 0]], axis=-1)[None],
                ("tilt",),
            )
        case "section":
            generators = np.zeros((2 * count, *sections.shape))
            labels = []
            for index in range(count):
                for axis, name in enumerate("rz"):
                    generators[2 * index + axis, index, :, axis] = 1.0
                    labels.append(f"section_{index}_{name}")
            return generators, tuple(labels)
        case "vertex":
            generators = np.zeros((2 * count * corners, *sections.shape))
            labels = []
            row = 0
            for index in range(count):
                for corner in range(corners):
                    for axis, name in enumerate("rz"):
                        generators[row, index, corner, axis] = 1.0
                        labels.append(f"vertex_{index}_{corner}_{name}")
                        row += 1
            return generators, tuple(labels)
    raise ValueError(f"unknown deformation block {block!r}, expected one of {BLOCKS}")


def pack_deformation(sections, blocks=AFFINE_BLOCKS) -> PackDeformation:
    """Return the parameter blocks ``blocks`` acting on a pack of sections.

    ``sections`` is ``(S, N, 2)``: every section of one pack carries the same
    corner count, which is what lets the whole basis be one dense tensor and the
    kernel be one fixed-shape call per section.
    """
    sections = np.asarray(sections, dtype=float)
    if sections.ndim != 3 or sections.shape[-1] != 2:
        raise ValueError(f"sections must be (S, N, 2), got {sections.shape}")
    rows, labels = [], []
    for block in blocks:
        generators, names = _block_generators(sections, block)
        rows.append(generators)
        labels.extend(names)
    return PackDeformation(
        sections, tuple(labels), _normalise(np.concatenate(rows, axis=0))
    )


def pack_coupling(sections, sensors: SensorArray, *, rule=DEFAULT_RULE) -> np.ndarray:
    """Return each channel's reading per ampere-turn, from the shipped kernel.

    Sections are averaged rather than summed because each one carries one turn
    of the same total current: a lattice of ``S`` turns driven at one ampere-turn
    puts one ampere in each turn and reads the mean of the per-turn columns.
    """
    sections = np.asarray(sections, dtype=float)
    total = np.zeros(len(sensors))
    for section in sections:
        rows = polygon_greens(
            sensors.r, sensors.z, section, n_panels=rule[0], n_nodes=rule[1]
        )
        total += np.einsum("ij,ji->i", sensors.projection, np.asarray(rows))
    return total / len(sections)


def traced_pack_coupling(
    xp, sections, sensors: SensorArray, masks, *, rule=DEFAULT_RULE
):
    """Return :func:`pack_coupling` with the geometry inside the trace.

    ``masks`` is the ``(S, N)`` static zero-weight topology from
    :func:`~nova.biot.polygon.horizontal_edges`, taken on the sections the
    Jacobian is evaluated at.  The section loop is over a static bound, so a
    trace unrolls it and the graph holds no control flow.
    """
    dtype = sections.dtype
    phi, weights = _phi_rule(*rule)
    nodes = tuple(
        xp.asarray(array, dtype=dtype)
        for array in (
            np.cos(phi),
            np.sin(phi),
            np.sin(2.0 * phi),
            weights * np.cos(phi),
        )
    )
    r = xp.asarray(sensors.r, dtype=dtype)[:, None]
    z = xp.asarray(sensors.z, dtype=dtype)[:, None]
    two_pi_r = 2.0 * np.pi * xp.asarray(sensors.r, dtype=dtype)
    projection = xp.asarray(sensors.projection, dtype=dtype)
    total = xp.zeros(len(sensors), dtype=dtype)
    for index in range(len(masks)):
        edge, weight, norm = traced_pack_section(xp, sections[index], masks[index])
        psi, dpsi_dr, dpsi_dz = _traced_psi_gradient(
            xp, r, z, edge[..., None], weight[:, None], *nodes, norm
        )
        rows = xp.stack([psi, -dpsi_dz / two_pi_r, dpsi_dr / two_pi_r], axis=-1)
        total = total + xp.sum(projection * rows, axis=-1)
    return total / len(masks)


def _rotate(sections: np.ndarray, angle: float) -> np.ndarray:
    """Return the pack turned about its own centroid."""
    if angle == 0.0:
        return sections
    centre = sections.reshape(-1, 2).mean(axis=0)
    cos, sin = np.cos(angle), np.sin(angle)
    turn = np.array([[cos, -sin], [sin, cos]])
    return centre + (sections - centre) @ turn.T


def _evaluation_tilts(sections: np.ndarray, offset: float | None) -> tuple[float, ...]:
    """Return the tilts the Jacobian is averaged over.

    A pack with no edge of constant z differentiates where it stands; one with
    such an edge is evaluated at a mirrored pair, which is what restores the
    gradient the zero-length edge interval cannot carry.
    """
    if offset is None:
        flat = any(horizontal_edges(section).any() for section in sections)
        offset = FLAT_EDGE_TILT if flat else 0.0
    return (0.0,) if offset == 0.0 else (offset, -offset)


def coupling_jacobian(
    deformation: PackDeformation,
    sensors: SensorArray,
    *,
    rule=DEFAULT_RULE,
    tilt_offset: float | None = None,
    mode: str = "forward",
    precision: Precision | str = Precision.AUTOMATIC,
) -> np.ndarray:
    """Return ``d(channel reading)/d(parameter)``, ``(n_channels, n_parameters)``.

    Units are the channel's own per ampere-turn -- tesla for a probe, weber for
    a loop -- per metre of pack root-mean-square displacement.  ``mode`` selects
    forward or reverse accumulation; they agree, and which is cheaper is set by
    the shape of the problem rather than by the kernel.
    """
    import jax
    import jax.numpy as jnp

    resolved = resolve_precision(precision, Precision.DOUBLE)
    dtype = jnp.float32 if resolved is Precision.SINGLE else jnp.float64
    generators = jnp.asarray(deformation.generators, dtype=dtype)
    start = jnp.zeros(deformation.size, dtype=dtype)
    jacobian = np.zeros((len(sensors), deformation.size))
    tilts = _evaluation_tilts(deformation.sections, tilt_offset)
    for angle in tilts:
        base = _rotate(deformation.sections, angle)
        masks = np.asarray([horizontal_edges(section) for section in base])
        anchor = jnp.asarray(base, dtype=dtype)

        def rows(parameters, anchor=anchor, masks=masks):
            sections = anchor + jnp.tensordot(parameters, generators, 1)
            return traced_pack_coupling(jnp, sections, sensors, masks, rule=rule)

        take = jax.jacfwd if mode == "forward" else jax.jacrev
        jacobian += np.asarray(take(rows)(start))
    return jacobian / len(tilts)


def mirror_rows(sections) -> np.ndarray:
    """Return the pack reflected about its own mid-height.

    A winding whose last row is partial sits either at maximum ``|z|`` or against
    the midplane; the two are a discrete state of the description, not a
    parameter, so they are compared as separate branches of the forward model.
    """
    sections = np.asarray(sections, dtype=float)
    height = sections[..., 1]
    mirrored = sections.copy()
    mirrored[..., 1] = (height.min() + height.max()) - height
    return mirrored


def branch_discriminability(one, other, noise) -> float:
    """Return the signal-to-noise separating two discrete forward models.

    Above one the array can tell the branches apart at the stated floor; below
    it the description's own choice stands unchallenged by the measurement.
    """
    return float(np.linalg.norm((np.asarray(one) - np.asarray(other)) / noise))


def standoff_widths(sensors: SensorArray, sections) -> np.ndarray:
    """Return each channel's distance to the pack in the pack's own width.

    The uniform-density reading of a winding breaks down within a couple of pack
    widths, so a screen stated in metres would travel wrongly between a solenoid
    and a divertor coil; stated in widths it travels.
    """
    sections = np.asarray(sections, dtype=float)
    hull = shapely.convex_hull(shapely.MultiPoint(sections.reshape(-1, 2)))
    low_r, low_z, high_r, high_z = hull.bounds
    width = min(high_r - low_r, high_z - low_z)
    return (
        np.array(
            [
                shapely.Point(r, z).distance(hull)
                for r, z in zip(sensors.r, sensors.z, strict=True)
            ]
        )
        / width
    )


def without_span(matrix, span) -> np.ndarray:
    """Return ``matrix`` with the column space of ``span`` projected out.

    A geometry mode that reproduces a redistribution of current among the
    described coils is not identified by the array, whatever its amplitude.
    Applied to whitened rows, this is the marginalisation of those freedoms.
    """
    basis, _ = np.linalg.qr(np.asarray(span))
    matrix = np.asarray(matrix)
    return matrix - basis @ (basis.T @ matrix)


@dataclass(frozen=True)
class ModeSpectrum:
    """Resolved geometry directions of an array, strongest first."""

    values: np.ndarray
    modes: np.ndarray
    labels: tuple[str, ...]

    @property
    def resolution(self) -> np.ndarray:
        """Return the pack displacement each mode needs to reach unit noise."""
        return 1.0 / self.values

    def composition(self, index: int) -> list[tuple[str, float]]:
        """Return one mode's named components, largest first."""
        weights = self.modes[index] / np.linalg.norm(self.modes[index])
        order = np.argsort(-np.abs(weights))
        return [(self.labels[position], float(weights[position])) for position in order]


def mode_spectrum(
    jacobian, gram, noise, *, labels=None, tolerance=1e-10
) -> ModeSpectrum:
    """Return the singular spectrum of a geometry Jacobian at a noise floor.

    ``jacobian`` is ``(n_channels, n_parameters)`` in each channel's own units,
    ``noise`` is the per-channel floor in those units, and ``gram`` is the
    parameter metric from :meth:`PackDeformation.gram`.  Whitening puts every
    channel on one scale and the metric puts every parameter on one scale, so a
    singular value is a signal-to-noise per metre and nothing about it depends on
    how the generators happen to be normalised.  Directions the generators do not
    span are dropped rather than returned as zeros.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    gram = np.asarray(gram, dtype=float)
    whitened = jacobian / np.asarray(noise, dtype=float)[:, None]
    weight, direction = np.linalg.eigh(gram)
    keep = weight > tolerance * weight.max()
    inverse_root = direction[:, keep] / np.sqrt(weight[keep])
    left, values, right = np.linalg.svd(whitened @ inverse_root, full_matrices=False)
    del left
    return ModeSpectrum(
        values,
        (inverse_root @ right.T).T,
        tuple(labels)
        if labels is not None
        else tuple(f"parameter_{index}" for index in range(gram.shape[0])),
    )


__all__ = [
    "AFFINE_BLOCKS",
    "BLOCKS",
    "FLAT_EDGE_TILT",
    "ModeSpectrum",
    "PackDeformation",
    "SensorArray",
    "branch_discriminability",
    "coupling_jacobian",
    "mirror_rows",
    "mode_spectrum",
    "pack_coupling",
    "pack_deformation",
    "standoff_widths",
    "traced_pack_coupling",
    "without_span",
]
