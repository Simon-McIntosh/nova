"""Flux orientation for external fields read by the topology adapter.

Every external field declares its own signed-flux orientation: a trace record
carries a polarity, and an IDS equilibrium carries the COCOS of the Data
Dictionary version it was written in together with the sign of its plasma
current. ``read_convention`` turns that declaration into the
``TopologyConvention`` the read consumes.

A declared convention fixes the type of the magnetic axis as a flux extremum,
so the flux a reference stores must agree with its own declaration. The check
has three outcomes and only one of them is acceptance:

* *agrees* - the stored span carries the sign the declaration requires;
* *contradicts* - the stored span carries the opposite sign, so the reference is
  internally inconsistent, and nova refuses it rather than reading it (a
  compensating sign flip is never inferred from the data);
* *undecidable* - the stored span sits within the sign tolerance of zero, so it
  fixes no extremum at all. An undecidable span is not an agreement and must not
  be read as one: it is refused with ``UndecidableSignError``, naming the span
  and the tolerance. A caller therefore cannot mistake "checked and agreed" for
  "could not tell".

The comparison uses the difference between the axis flux and the boundary flux,
which an additive gauge constant leaves unchanged.
"""

from __future__ import annotations

import numpy as np
import jax
from types import SimpleNamespace

from nova.equilibrium.topology import TopologyConvention
from nova.jax.config import configure_dtypes
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField

_SIGN_TOLERANCE = 1024.0 * float(np.finfo(np.float64).eps)
"""Relative floor below which a flux span's sign cannot be decided.

A span is the difference of two spline samples of the grid, so it is resolved
only to the rounding of the flux it is a difference of: the double-precision
unit roundoff ``eps = 2**-52`` bounds one sample's relative error and the
difference of two such samples accumulates a few multiples of it, scaled by the
flux width the samples span. The floor is a thousand-fold margin over that unit
roundoff -- ``1024 * eps``, about ``2.3e-13`` -- applied to a flux *width*: the
grid's peak-to-peak range, or the compared scalars' own difference. A range and
a difference are both invariant under an additive constant, so the band does not
move with the gauge. ``sign_tolerance`` is the one place the product lives.
"""

_CURVATURE_TOLERANCE = 1e-9
"""Relative floor on the axis Hessian eigenvalues for the curvature fallback.

Where no wall point lies in the grid the axis type is read from the spline
Hessian instead of a span. An eigenvalue whose magnitude lies within this
relative floor of zero fixes no curvature, so the axis type is undecidable
there too.
"""


class ReferenceSignError(ValueError):
    """A reference's stored flux contradicts the orientation it declares.

    Raised where the grid, or the stored scalars, read the magnetic axis as the
    opposite flux extremum to the one the declared convention requires. The
    message names the slice, the declaration, and the measured signs.
    """


class UndecidableSignError(ReferenceSignError):
    """A reference's stored flux span is within the sign tolerance of zero.

    Raised where a span (or a curvature eigenvalue) is too close to zero to fix
    the axis as either extremum. Such a reference is refused, not accepted: the
    message names the slice, the declaration, the span and the tolerance the
    span failed to clear.
    """


def read_convention(*, polarity=None, dd_version=None, plasma_current=None):
    """Use a trace's signed axis direction or the IDS's written convention."""
    if polarity is not None:
        if dd_version is not None or plasma_current is not None:
            raise ValueError("trace polarity and IDS orientation are exclusive")
        if polarity not in (-1, 1):
            raise ValueError("trace polarity must be -1 or 1")
        return TopologyConvention.from_cocos(17, polarity)
    if dd_version is None or plasma_current is None:
        raise ValueError("IDS orientation needs DD version and plasma current")
    return TopologyConvention.from_cocos(ids_cocos(dd_version), plasma_current)


def ids_cocos(dd_version):
    """COCOS of the Data Dictionary version that wrote an equilibrium."""
    major = int(str(dd_version).split(".", 1)[0])
    if major not in (3, 4):
        raise ValueError(f"unsupported equilibrium DD version: {dd_version}")
    return {3: 11, 4: 17}[major]


def ids_declaration(dd_version, plasma_current):
    """Name the metadata that fixes an equilibrium's flux orientation."""
    return (
        f"DD {dd_version} (COCOS {ids_cocos(dd_version)}) with "
        f"Ip = {plasma_current:+.3f} A"
    )


def ids_boundary_points(view):
    """Stored X-points, else the outline; None if neither is present."""
    if view["xpoints"]:
        return np.asarray(view["xpoints"], dtype=np.float64)
    if len(view["outline"]):
        return np.asarray(view["outline"], dtype=np.float64)
    return None


def grid_field(rgrid, zgrid, psi):
    """Build the same spline field used by the external equilibrium read."""
    configure_dtypes()
    assert jax.config.jax_enable_x64
    r = np.asarray(rgrid, dtype=np.float64).ravel()
    z = np.asarray(zgrid, dtype=np.float64).ravel()
    values = np.asarray(psi, dtype=np.float64)
    if values.shape != (r.size, z.size):
        raise ValueError(
            f"psi shape {values.shape} does not match grid {(r.size, z.size)}"
        )
    return TraceSplineField.from_record(SimpleNamespace(rgrid=r, zgrid=z, psi=values))


def grid_value(rgrid, zgrid, psi, point, *, field=None):
    """Sample the reference's spline, preserving the read's interpolation."""
    if field is None:
        field = grid_field(rgrid, zgrid, psi)
    r = np.asarray(rgrid, dtype=np.float64).ravel()
    z = np.asarray(zgrid, dtype=np.float64).ravel()
    r0, z0 = float(point[0]), float(point[1])
    if not (r[0] <= r0 <= r[-1] and z[0] <= z0 <= z[-1]):
        raise ValueError(f"point ({r0}, {z0}) lies outside the psi grid")
    return float(field.value((r0, z0)))


def _boundary_value(rgrid, zgrid, psi, boundary, *, field):
    """Mean grid sample over one boundary point or an (n, 2) set of them."""
    points = np.asarray(boundary, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        raise ValueError("boundary carries no point")
    samples = [grid_value(rgrid, zgrid, psi, point, field=field) for point in points]
    return float(np.mean(samples))


def sign_tolerance(scale):
    """The flux width within which a span's sign is undecidable, as a number.

    ``scale`` is a flux width, never the span under test with no other content:
    the grid's peak-to-peak range, or the compared scalars' own difference. Both
    are invariant under an additive gauge constant, so the band an additive
    constant cannot move is the band the comparison uses. The floor keeps the
    tolerance from collapsing to zero on a degenerate grid.
    """
    return _SIGN_TOLERANCE * max(abs(float(scale)), 1.0)


def flux_sign(span, scale):
    """Sign of a flux difference, or 0 where its sign cannot be decided.

    The scale is a flux width -- a range or a difference, never an absolute psi
    value -- and the zero band is a fixed multiple of the double-precision unit
    roundoff times that width, so a span whose sign the arithmetic resolves is
    decidable while a span at the level of the rounding that produced it is not.
    An additive gauge constant leaves the band unchanged, so it cannot turn a
    decisive contradiction into an undecided sign.
    """
    span = float(span)
    if not np.isfinite(span):
        return 0
    if abs(span) <= sign_tolerance(scale):
        return 0
    return 1 if span > 0.0 else -1


def _declared_sigma(convention):
    sigma = int(np.sign(float(convention.sigma)))
    if sigma == 0:
        raise ValueError("declared convention carries no flux orientation")
    return sigma


def _extremum(sigma):
    return "maximum" if sigma > 0 else "minimum"


def _refuse(required, signs, spans, tolerances, *, slice_index=None, declared=""):
    """Raise on a contradiction, else on an undecidable span, naming both.

    A contradiction is decisive and is reported first. An undecidable span is
    never accepted: where no span contradicts but one lies inside its tolerance
    of zero, the reference fixes no extremum and is refused rather than read.
    """
    contradiction = {
        name: sign for name, sign in signs.items() if sign not in (0, required)
    }
    if contradiction:
        read = "; ".join(
            f"{name} psi axis - boundary = {spans[name]:+.6e} (sign {signs[name]:+d})"
            for name in sorted(signs)
        )
        raise ReferenceSignError(
            f"slice {slice_index} declares {declared}, requiring the magnetic axis to "
            f"be a psi {_extremum(required)} (sign {required:+d}); the stored {read}"
        )
    undecidable = sorted(name for name, sign in signs.items() if sign == 0)
    if undecidable:
        name = undecidable[0]
        raise UndecidableSignError(
            f"slice {slice_index} declares {declared}, requiring the magnetic axis to "
            f"be a psi {_extremum(required)} (sign {required:+d}); the {name} span "
            f"{spans[name]:+.6e} lies within the sign tolerance "
            f"{tolerances[name]:.3e} of zero, so its sign is undecidable and the "
            "reference is refused rather than read as agreement"
        )


def check_axis_boundary_sign(
    convention,
    rgrid,
    zgrid,
    psi,
    axis,
    boundary,
    *,
    scalar_axis_psi=None,
    scalar_boundary_psi=None,
    slice_index=None,
    declared="",
    field=None,
):
    """Refuse a reference whose grid flux sign contradicts its declared orientation.

    ``axis`` is a point and ``boundary`` is a point or an ``(n, 2)`` set of
    them, in the reference's own coordinates; the grid is sampled at the axis
    and at each boundary point by bilinear interpolation, and the axis sample
    minus the mean boundary sample is a flux span an additive constant cannot
    change. Where the stored scalars are also supplied their span is checked
    against the same declaration, so a grid that happens to agree cannot hide
    scalars that do not (and the reverse).

    The declared convention fixes whether the axis is a psi maximum or minimum;
    a span whose sign disagrees is a refusal, and a span within the sign
    tolerance of zero fixes no extremum and is refused as undecidable rather
    than accepted as agreement.

    Returns the measured signs ``{name: sign}`` on success.
    """
    required = _declared_sigma(convention)
    if boundary is None or np.asarray(boundary).size == 0:
        raise ReferenceSignError(
            f"slice {slice_index} declares {declared}, requiring the magnetic axis to "
            f"be a psi {_extremum(required)} (sign {required:+d}); no boundary "
            "point is stored, so grid and scalar signs cannot be compared"
        )
    if field is None:
        field = grid_field(rgrid, zgrid, psi)
    axis_value = grid_value(rgrid, zgrid, psi, axis, field=field)
    boundary_value = _boundary_value(rgrid, zgrid, psi, boundary, field=field)
    grid_span = axis_value - boundary_value
    grid_width = float(np.ptp(np.asarray(psi, dtype=np.float64)))
    signs = {"grid": flux_sign(grid_span, grid_width)}
    spans = {"grid": grid_span}
    tolerances = {"grid": sign_tolerance(grid_width)}
    if scalar_axis_psi is not None and scalar_boundary_psi is not None:
        scalar_span = float(scalar_axis_psi) - float(scalar_boundary_psi)
        signs["scalar"] = flux_sign(scalar_span, abs(scalar_span))
        spans["scalar"] = scalar_span
        tolerances["scalar"] = sign_tolerance(scalar_span)
    _refuse(
        required,
        signs,
        spans,
        tolerances,
        slice_index=slice_index,
        declared=declared,
    )
    return signs


def trace_record_boundary(record):
    """In-grid wall-polygon vertices of a trace record as an (n, 2) array, or None."""
    r = np.asarray(record.rgrid, dtype=np.float64).ravel()
    z = np.asarray(record.zgrid, dtype=np.float64).ravel()
    vertices = np.column_stack(
        (
            np.concatenate(
                (np.asarray(record.wall_r)[:, 0], np.asarray(record.wall_r)[:, 1])
            ),
            np.concatenate(
                (np.asarray(record.wall_z)[:, 0], np.asarray(record.wall_z)[:, 1])
            ),
        )
    )
    inside = (
        (vertices[:, 0] >= r[0])
        & (vertices[:, 0] <= r[-1])
        & (vertices[:, 1] >= z[0])
        & (vertices[:, 1] <= z[-1])
    )
    if inside.sum() < 3:
        return None
    return vertices[inside]


def check_trace_record(record, *, boundary=None):
    """Refuse a trace record whose polarity contradicts its own grid.

    The record declares its orientation through its polarity; the grid is
    sampled at the stored axis and at the record's wall polygon and their
    difference is checked against the declaration. Where no wall point lies
    inside the grid, the spline's Hessian at the stored axis determines whether
    that axis is a maximum or minimum.
    """
    convention = read_convention(polarity=record.polarity)
    required = _declared_sigma(convention)
    point = boundary if boundary is not None else trace_record_boundary(record)
    field = grid_field(record.rgrid, record.zgrid, record.psi)
    if point is None:
        eigenvalues = np.linalg.eigvalsh(
            np.asarray(field.evaluate((record.axis_r, record.axis_z)).hessian)
        )
        largest = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        curvature_floor = _CURVATURE_TOLERANCE * max(largest, 1.0)
        curvature_sign = (
            1
            if np.all(eigenvalues < -curvature_floor)
            else -1
            if np.all(eigenvalues > curvature_floor)
            else 0
        )
        if curvature_sign == 0:
            raise UndecidableSignError(
                f"slice {record.slice} declares polarity {record.polarity:+d}, "
                f"requiring a psi {_extremum(required)} (sign {required:+d}); "
                "no boundary lies in the grid and the grid curvature at the stored "
                f"axis has eigenvalues {eigenvalues.tolist()}, within the curvature "
                f"tolerance {curvature_floor:.3e} of zero, so the axis type is "
                "undecidable"
            )
        if curvature_sign != required:
            raise ReferenceSignError(
                f"slice {record.slice} declares polarity {record.polarity:+d}, "
                f"requiring a psi {_extremum(required)} (sign {required:+d}); "
                "no boundary lies in the grid and grid curvature at the stored "
                f"axis has eigenvalues {eigenvalues.tolist()} "
                f"(sign {curvature_sign:+d})"
            )
        return {"grid curvature": curvature_sign}
    axis_value = grid_value(
        record.rgrid,
        record.zgrid,
        record.psi,
        (record.axis_r, record.axis_z),
        field=field,
    )
    boundary_value = _boundary_value(
        record.rgrid, record.zgrid, record.psi, point, field=field
    )
    span = axis_value - boundary_value
    width = float(np.ptp(np.asarray(record.psi, dtype=np.float64)))
    signs = {"grid": flux_sign(span, width)}
    _refuse(
        required,
        signs,
        {"grid": span},
        {"grid": sign_tolerance(width)},
        slice_index=record.slice,
        declared=f"polarity {record.polarity:+d}",
    )
    return signs
