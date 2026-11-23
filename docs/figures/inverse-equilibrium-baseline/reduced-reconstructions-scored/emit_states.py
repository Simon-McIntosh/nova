"""Run the three reduced reconstructions and write each as an IMAS equilibrium.

This is the driver behind the ``reduced-reconstructions-scored`` case. It builds
the ITER row-exchange case from the stored record, generating map and pulse
(``case.py`` beside it), runs the moment, harmonic and profile reconstructions
of the same magnetics rows, and writes each result through that reconstruction's
own conversion.

It then writes each column's negative control. The section 3 perturbation --
plasma current, position, flux span and map scale -- is applied to the
*reconstruction*, and the perturbed reconstruction goes through the same
conversion, so a control exercises the conversion rather than copying an
already-written equilibrium.

The states are written under ``NOVA_REDUCED_STATE_ROOT``. Run it with the
``case`` module beside it, from this directory:

    PYTHONPATH=. JAX_PLATFORMS=cpu python emit_states.py
"""

import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from case import build  # noqa: E402

from nova.equilibrium.harmonic import (  # noqa
    ReconstructHarmonic,
    write_harmonic_equilibrium,
)
from nova.equilibrium.moment import (  # noqa: E402
    CurrentCells,
    ReconstructMoment,
    write_moment_equilibrium,
)
from nova.equilibrium import profile as profile_module  # noqa: F401,E402
from nova.jax.config import configure_dtypes  # noqa: E402
import jax  # noqa: E402

configure_dtypes()
assert jax.config.jax_enable_x64 is True

STATE_ROOT = Path(
    os.environ.get(
        "NOVA_REDUCED_STATE_ROOT",
        "/home/ITER/mcintos/.cache/nova-inverse-equilibrium-baseline/"
        "reduced-reconstructions-scored/states",
    )
)
STATE_ROOT.mkdir(parents=True, exist_ok=True)
B0 = 5.3
R0 = 6.2

case = build()
rg, zg, inside = case["rg"], case["zg"], case["inside"]
wall = case["wall"]
record = case["record"]
time = float(record["time"])
axis_seed = tuple(float(v) for v in record["generating"]["axis"])

# --- moment -----------------------------------------------------------------
radius, height = np.meshgrid(rg, zg)
cells = CurrentCells(
    radius.ravel(),
    height.ravel(),
    float(rg[1] - rg[0]),
    float(zg[1] - zg[0]),
    inside.ravel().astype(float),
)
moment = ReconstructMoment(cells, case["magnetics"], case["grid"], major_radius=case["r0"])
moment_result = moment.solve(case["measurement"])
moment_path = write_moment_equilibrium(
    STATE_ROOT / "moment",
    moment_result,
    rg,
    zg,
    inside,
    vacuum_b0=B0,
    reference_radius=R0,
    time=time,
)
print("moment", moment_path, flush=True)

# --- harmonic ---------------------------------------------------------------
from nova.equilibrium.harmonic import HarmonicConfig  # noqa: E402

harmonic = ReconstructHarmonic(
    case["magnetics"],
    HarmonicConfig(pole_r=axis_seed[0], pole_z=axis_seed[1], ip_anchor=True),
)
harmonic_result = harmonic.fit(case["measurement"])
harmonic_path = write_harmonic_equilibrium(
    STATE_ROOT / "harmonic",
    harmonic_result,
    rg,
    zg,
    inside,
    vacuum_flux=case["vacuum_flux"],
    limiter_r=wall[:, 0],
    limiter_z=wall[:, 1],
    plasma_current=float(case["measurement"].plasma_current),
    vacuum_b0=B0,
    reference_radius=R0,
    time=time,
)
print("harmonic", harmonic_path, flush=True)

# --- profile ----------------------------------------------------------------
from nova.equilibrium.profile import (  # noqa: E402
    ProfileDegrees,
    ReconstructProfile,
    write_profile_equilibrium,
)

sources = case["sources"]
source_r = np.array([s[0] for s in sources])
source_z = np.array([s[1] for s in sources])
source_dr = np.array([s[2] for s in sources])
source_dz = np.array([s[3] for s in sources])
source_current = np.array([s[4] for s in sources])
solver = ReconstructProfile.from_geometry(
    grid_r=rg,
    grid_z=zg,
    inside_limiter=inside,
    cell_width=np.asarray(float(rg[1] - rg[0])),
    cell_height=np.asarray(float(zg[1] - zg[0])),
    source_r=source_r,
    source_z=source_z,
    source_width=source_dr,
    source_height=source_dz,
    source_names=[s[5] for s in sources],
    magnetics=case["magnetics"],
    degrees=ProfileDegrees(1, 1),
    axis_seed=axis_seed,
    wall_r=wall[:, 0],
    wall_z=wall[:, 1],
    iterations=8,
)
source = jax.numpy.asarray(source_current)
plasma = jax.numpy.asarray(float(case["measurement"].plasma_current))
initial = solver.initial_flux(source, plasma)
profile_result = solver.solve(
    source,
    plasma,
    jax.numpy.asarray(case["measurement"].measured),
    jax.numpy.asarray(case["measurement"].scale),
    jax.numpy.asarray(case["measurement"].mask),
    initial,
)
profile_path = write_profile_equilibrium(
    STATE_ROOT / "profile",
    profile_result,
    rg,
    zg,
    inside,
    limiter_r=wall[:, 0],
    limiter_z=wall[:, 1],
    vacuum_b0=B0,
    reference_radius=R0,
    time=time,
)
# --- negative controls: the section 3 perturbation through each conversion ----
CONTROL_SHIFT = 0.03
"""Radial shift of a control state [m]."""

CONTROL_CURRENT = 1.06
"""Current scale of a control, as the section 3 perturbation applies it."""

CONTROL_MAP = 1.04
"""Map scale of a control: the flux span grows two percent at each end."""

CONTROL_SPAN = 0.02
"""Fraction of the flux span the control grows at each end."""


def shifted(field, shift):
    """Return the map moved ``shift`` metres outward, held at both edges."""

    field = np.asarray(field, dtype=float)
    rows = np.empty_like(field)
    for index, row in enumerate(field):
        rows[index] = np.interp(rg, rg - shift, row, left=row[0], right=row[-1])
    return rows


def perturbed_moment_state(result):
    """Return the moment reconstruction with the control perturbation applied."""

    span = float(result.boundary_flux) - float(result.axis_flux)
    axis_flux = float(result.axis_flux) - CONTROL_SPAN * span
    flux = shifted(result.flux, CONTROL_SHIFT)
    flux = axis_flux + CONTROL_MAP * (flux - float(result.axis_flux))
    outline = np.asarray(result.ring, dtype=float).copy()
    outline[:, 0] = outline[:, 0] + CONTROL_SHIFT
    return dataclasses.replace(
        result,
        ring=outline,
        flux=flux,
        cell_current=CONTROL_CURRENT * np.asarray(result.cell_current, dtype=float),
        axis_flux=axis_flux,
        boundary_flux=float(result.boundary_flux) + CONTROL_SPAN * span,
    )


def perturbed_profile_state(result):
    """Return the profile reconstruction with the control perturbation applied."""

    grid_flux = np.asarray(result.flux, dtype=float).reshape(zg.size, rg.size)
    axis_r = int(np.argmin(np.abs(rg - float(result.axis[0]))))
    axis_z = int(np.argmin(np.abs(zg - float(result.axis[1]))))
    axis_value = float(grid_flux[axis_z, axis_r])
    moved = shifted(grid_flux, CONTROL_SHIFT)
    flux = axis_value + CONTROL_MAP * (moved - axis_value)
    return result._replace(
        flux=flux.reshape(-1),
        cell_current=CONTROL_CURRENT * np.asarray(result.cell_current, dtype=float),
        axis=np.asarray(result.axis, dtype=float) + np.array([CONTROL_SHIFT, 0.0]),
        boundary_flux=axis_value
        + CONTROL_MAP * (float(result.boundary_flux) - axis_value),
    )


def perturbed_harmonic_state(inversion):
    """Return the harmonic inversion with the control perturbation applied."""

    fields = {
        member.name: getattr(inversion.config, member.name)
        for member in dataclasses.fields(type(inversion.config))
    }
    fields["pole_r"] = float(inversion.config.pole_r) + CONTROL_SHIFT
    return dataclasses.replace(
        inversion,
        config=type(inversion.config)(**fields),
        coefficients=CONTROL_CURRENT
        * np.asarray(inversion.coefficients, dtype=float),
    )


def write_controls():
    """Write each perturbed reconstruction through its own conversion."""

    write_moment_equilibrium(
        STATE_ROOT / "moment-perturbed",
        perturbed_moment_state(moment_result),
        rg,
        zg,
        inside,
        vacuum_b0=B0,
        reference_radius=R0,
        time=time,
    )
    write_harmonic_equilibrium(
        STATE_ROOT / "harmonic-perturbed",
        perturbed_harmonic_state(harmonic_result),
        rg,
        zg,
        inside,
        vacuum_flux=case["vacuum_flux"],
        limiter_r=wall[:, 0],
        limiter_z=wall[:, 1],
        plasma_current=CONTROL_CURRENT * float(case["measurement"].plasma_current),
        vacuum_b0=B0,
        reference_radius=R0,
        time=time,
    )
    write_profile_equilibrium(
        STATE_ROOT / "profile-perturbed",
        perturbed_profile_state(profile_result),
        rg,
        zg,
        inside,
        limiter_r=wall[:, 0],
        limiter_z=wall[:, 1],
        vacuum_b0=B0,
        reference_radius=R0,
        time=time,
    )


write_controls()
print("controls written", flush=True)
print(json.dumps({"states": str(STATE_ROOT)}, indent=1))
