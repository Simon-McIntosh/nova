"""1D resistive current diffusion -- analytic pins plus a reference cross-check.

Correctness is pinned analytically and against reference implementations, with no
data dependency:

* the CIRCULAR large-aspect limit reduces to the classical cylinder equation
  ``dpsi/dt = (eta/mu0)(1/r) d/dr(r dpsi/dr)``: a Bessel eigenmode perturbation
  must decay at its analytic rate, the prescribed-current edge condition must hold
  the enclosed current, and the late-time state must consume flux rigidly at the
  analytic ring voltage;
* the solver FORMULATION -- the time coefficient, the current-condition
  normalisation, the theta-scheme -- matches TORAX on a shared circular case to
  better than 1%;
* the JAX Thomas solve reproduces the torch prototype it replaces to fp64
  round-off across backward-Euler, Crank-Nicolson, ramping-current and
  perturbed-initial-state cases;
* the flux budget decomposes exactly (surface = resistive + inductive);
* the current images round-trip a known coefficient vector through the
  non-negative projection, and the predicted profile integrates back to the
  enclosed current.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nova.biot.greens import MU0
from nova.jax.config import Precision, configure_dtypes
from nova.transport.current_diffusion import (
    CurrentDiffusion,
    EtaProfile,
    FluxSurfaceGeometry,
    basis_projection_images,
    diffuse_psi,
    ejima_coefficient,
    flux_budget,
    poloidal_field_energy_li,
    profile_shapes,
    project_coefficients,
)

DATA = Path(__file__).parent / "data"


def _f64(value):
    """Construct an explicitly selected double-precision JAX value."""
    configure_dtypes()
    import jax.numpy as jnp

    return jnp.asarray(value, dtype=jnp.float64)


#: the geometry fields the frozen torch reference carries, in dataclass order
_GEOMETRY_ARRAYS = (
    "rho_face",
    "rho_cell",
    "psi_face",
    "psi_n_face",
    "psi_n_cell",
    "vpr_face",
    "vpr_cell",
    "g2_face",
    "g3_face",
    "g3_cell",
    "f_face",
    "f_cell",
    "b2_cell",
    "inv_r_cell",
    "q_face",
)
_GEOMETRY_SCALARS = (
    "phi_b",
    "r0",
    "ip_amperes",
    "axis_psi",
    "boundary_psi",
    "volume",
    "flux_sign",
)


def _circular_geometry(
    *, a=0.5, r0=3.0, b0=2.0, ip=5.0e5, n_rho=48
) -> FluxSurfaceGeometry:
    """Exact circular large-aspect metrics -- the analytic verification rig.

    ``V = 2 pi^2 R0 r^2``, ``Phi_tor = pi r^2 B0``, ``rho = r/a``, ``F = R0 B0``,
    ``<1/R^2> = 1/R0^2``, ``g2 = 16 pi^4 r^2``.  The initial flux carries a uniform
    current density: the enclosed current goes as ``rho^2``, integrated through the
    Ampere identity.
    """
    rho_face = np.linspace(0.0, 1.0, n_rho + 1)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    r_face = a * rho_face
    phi_b = np.pi * a * a * b0
    f_face = np.full(n_rho + 1, r0 * b0)
    g2_face = 16.0 * np.pi**4 * r_face**2
    g3_face = np.full(n_rho + 1, 1.0 / r0**2)
    vpr_face = 4.0 * np.pi**2 * r0 * a * a * rho_face
    d_face = np.zeros(n_rho + 1)
    d_face[1:] = g2_face[1:] * g3_face[1:] / rho_face[1:]
    gradient = np.zeros(n_rho + 1)
    gradient[1:] = (
        ip
        * rho_face[1:] ** 2
        * (16.0 * np.pi**3 * MU0 * phi_b)
        / (d_face[1:] * f_face[1:])
    )
    psi_face = np.concatenate(
        [[0.0], np.cumsum(0.5 * (gradient[1:] + gradient[:-1]) * np.diff(rho_face))]
    )
    return FluxSurfaceGeometry(
        rho_face=rho_face,
        rho_cell=rho_cell,
        psi_face=psi_face,
        psi_n_face=rho_face**2,
        psi_n_cell=rho_cell**2,
        vpr_face=vpr_face,
        vpr_cell=0.5 * (vpr_face[:-1] + vpr_face[1:]),
        g2_face=g2_face,
        g3_face=g3_face,
        g3_cell=np.full(n_rho, 1.0 / r0**2),
        f_face=f_face,
        f_cell=np.full(n_rho, r0 * b0),
        b2_cell=np.full(n_rho, b0 * b0),
        inv_r_cell=np.full(n_rho, 1.0 / r0),
        phi_b=phi_b,
        r0=r0,
        ip_amperes=ip,
        axis_psi=0.0,
        boundary_psi=float(psi_face[-1]),
        volume=2.0 * np.pi**2 * r0 * a * a,
        q_face=np.ones(n_rho + 1),
    )


# --- analytic pins ---------------------------------------------------------
def test_diffusion_precision_is_selected_per_solve():
    """General automatic diffusion is fp64 while explicit fp32 stays available."""
    geometry = _circular_geometry(n_rho=8)
    eta = EtaProfile(eta0=1.0e-7, contrast=0.0, shape=1.0)
    times = np.linspace(0.0, 0.001, 3)
    current = np.full(times.size, geometry.ip_amperes)

    automatic = diffuse_psi(geometry, eta, t_grid=times, ip_of_t=current)
    single = diffuse_psi(
        geometry,
        eta,
        t_grid=times,
        ip_of_t=current,
        precision=Precision.SINGLE,
    )

    assert automatic["psi_face"].dtype == np.float64
    assert single["psi_face"].dtype == np.float32


def test_bessel_mode_decays_at_the_analytic_rate():
    """A Neumann eigenmode of the cylinder equation decays at
    ``lam = (eta/mu0)(j'_1/a)^2`` -- the diffusion operator against the classical
    solution."""
    from scipy.special import j0, j1, jn_zeros

    minor_radius = 0.5
    eta0 = 1.0e-6
    geometry = _circular_geometry(a=minor_radius)
    eta = EtaProfile(eta0=eta0, contrast=0.0, shape=1.0)
    root = float(jn_zeros(1, 1)[0])  # first zero of J1: a Neumann mode of J0
    assert abs(j1(root)) < 1e-12
    amplitude = 1.0e-3 * float(np.ptp(geometry.psi_face))
    mode = j0(root * geometry.rho_face)
    rate = (eta0 / MU0) * (root / minor_radius) ** 2

    times = np.linspace(0.0, 0.2 / rate, 400)
    ip_of_t = np.full(times.size, geometry.ip_amperes)
    base = diffuse_psi(geometry, eta, t_grid=times, ip_of_t=ip_of_t)
    perturbed = diffuse_psi(
        geometry,
        eta,
        t_grid=times,
        ip_of_t=ip_of_t,
        psi0_face=geometry.psi_face + amplitude * mode,
    )
    deviation = perturbed["psi_face"] - base["psi_face"]
    # remove the neutral constant mode, conserved under pure Neumann conditions
    deviation = deviation - deviation.mean(axis=1, keepdims=True)
    centred = mode - mode.mean()
    projection = deviation @ centred / (centred @ centred)
    # fit the decay over the window, skipping the first backward-Euler transient
    fitted = -np.polyfit(times[1:], np.log(np.abs(projection[1:])), 1)[0]
    assert abs(fitted - rate) / rate < 0.03


def test_the_current_condition_holds_the_enclosed_current():
    """The edge-gradient condition carries a ramping plasma current: the Ampere
    identity read back from the evolved flux matches the drive."""
    geometry = _circular_geometry()
    eta = EtaProfile(eta0=1.0e-7, contrast=0.0, shape=1.0)
    times = np.linspace(0.0, 0.05, 201)
    ip_of_t = geometry.ip_amperes * (1.0 + 0.5 * times / times[-1])
    step = diffuse_psi(geometry, eta, t_grid=times, ip_of_t=ip_of_t)
    edge = geometry.enclosed_current(step["psi_face"][-1])[-1]
    # a transient skin-layer state carries an O(drho) scheme-consistency gap in
    # the read-back; the condition itself is exact (see the rigid-consumption pin)
    assert abs(edge - ip_of_t[-1]) / ip_of_t[-1] < 2e-2


def test_late_time_consumption_is_rigid_and_ohmic():
    """At constant current with uniform resistivity the state relaxes to rigid
    flux consumption: a spatially uniform loop voltage at the analytic ring value
    ``V = 2 pi R0 eta Ip / (pi a^2)`` -- the resistive channel of the budget."""
    minor_radius, r0 = 0.5, 3.0
    eta0 = 1.0e-6
    geometry = _circular_geometry(a=minor_radius, r0=r0)
    eta = EtaProfile(eta0=eta0, contrast=0.0, shape=1.0)
    resistive_time = MU0 * minor_radius**2 / eta0
    times = np.linspace(0.0, 3.0 * resistive_time, 600)
    step = diffuse_psi(
        geometry,
        eta,
        t_grid=times,
        ip_of_t=np.full(times.size, geometry.ip_amperes),
    )
    v_axis, v_bdry = step["v_axis"][-1], step["v_bdry"][-1]
    ring = 2.0 * np.pi * r0 * eta0 * geometry.ip_amperes / (np.pi * minor_radius**2)
    assert abs(v_axis - v_bdry) < 0.05 * abs(ring)  # rigid consumption
    assert abs(abs(v_bdry) - ring) / ring < 0.05  # ohmic magnitude
    budget = flux_budget(step, geometry)
    assert np.isclose(
        budget["d_psi_bdry"], budget["d_psi_axis"] + budget["d_psi_internal"]
    )
    # constant current with a relaxed profile: consumption is almost all resistive
    assert abs(budget["d_psi_internal"]) < 0.2 * abs(budget["d_psi_axis"])


def test_ejima_coefficient_normalisation():
    assert np.isclose(ejima_coefficient(MU0 * 0.85 * 5.0e5 * 0.45, 5.0e5, 0.85), 0.45)


def test_eta_profile_family_is_bounded_and_monotone():
    eta = EtaProfile(eta0=3.0e-8, contrast=3.0, shape=1.5)
    values = eta(np.linspace(0.0, 1.0, 50))
    assert np.all(np.diff(values) >= 0.0)  # monotone toward the cold edge
    assert np.isclose(values[0], 3.0e-8)
    assert np.isclose(values[-1], 3.0e-8 * np.exp(3.0))
    round_trip = EtaProfile.from_vector(eta.as_vector())
    assert np.isclose(round_trip.eta0, eta.eta0)
    assert np.isclose(round_trip.contrast, eta.contrast)
    _low, high = zip(*EtaProfile.BOUNDS, strict=True)
    clipped = EtaProfile.from_vector(np.array([0.0, 99.0, 99.0]))
    assert clipped.eta0 <= high[0]
    assert clipped.contrast <= high[1]
    assert clipped.shape <= high[2]


def test_internal_inductance_is_positive_and_order_unity():
    geometry = _circular_geometry()
    li = poloidal_field_energy_li(geometry)
    assert 0.1 < li < 5.0


# --- reference cross-checks ------------------------------------------------
def _reference_geometry(stored) -> FluxSurfaceGeometry:
    """Rebuild the frozen reference's geometry so both solvers see one input."""
    fields = {
        name: np.asarray(stored[f"geometry__{name}"], dtype=np.float64)
        for name in _GEOMETRY_ARRAYS
    }
    fields |= {name: float(stored[f"geometry__{name}"]) for name in _GEOMETRY_SCALARS}
    return FluxSurfaceGeometry(**fields)


def test_jax_solve_reproduces_the_torch_prototype():
    """The jitted fp64 Thomas solve replaces a torch dense solve of the same
    tridiagonal system, so it must agree to round-off -- not to a tolerance."""
    with np.load(DATA / "current_diffusion_torch_reference.npz") as stored:
        geometry = _reference_geometry(stored)
        cases = [str(name) for name in stored["cases"]]
        assert cases, "the reference carries no cases"
        for case in cases:
            step = diffuse_psi(
                geometry,
                EtaProfile(*stored[f"{case}__eta"]),
                t_grid=stored[f"{case}__times"],
                ip_of_t=stored[f"{case}__ip_of_t"],
                psi0_face=stored[f"{case}__psi0"],
                theta=float(stored[f"{case}__theta"]),
            )
            expected = stored[f"{case}__psi_face"]
            scale = float(np.abs(expected).max())
            # the reference carries a strided sample of the trajectory, so the
            # comparison spans the whole window rather than just its end state
            sample = stored[f"{case}__sample"]
            np.testing.assert_allclose(
                step["psi_face"][sample], expected, rtol=1e-9, atol=1e-11 * scale
            )
            for trace in ("v_axis", "v_bdry", "psidot_face"):
                np.testing.assert_allclose(
                    step[trace],
                    stored[f"{case}__{trace}"],
                    rtol=1e-8,
                    atol=1e-10 * scale,
                    err_msg=f"{case}/{trace}",
                )


@pytest.mark.slow
def test_solver_formulation_matches_torax_on_the_shared_circular_case():
    """A TORAX run evolved pure ohmic current diffusion on its built-in circular
    geometry with a ramping prescribed current and a Sauter conductivity from
    prescribed profiles.  Here the SAME metrics, conductivity, initial flux and
    current trace drive this solver in chunks (conductivity refreshed per chunk,
    mirroring the frozen-per-interval contract), pinning the FORMULATION -- time
    coefficient, current-condition normalisation, theta-scheme -- against the
    reference implementation.  The metric extraction is pinned separately, so the
    diffusion face coefficient here is TORAX's own array."""
    with np.load(DATA / "torax_circular_psi_reference.npz") as stored:
        rho_face = np.asarray(stored["rho_face_norm"], dtype=np.float64)
        rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
        reference_grid = np.concatenate(
            [[0.0], np.asarray(stored["rho_cell_norm"]), [1.0]]
        )
        times = np.asarray(stored["times"], dtype=np.float64)
        n_rho = rho_face.size - 1
        # re-express g2 so that g2 g3 / rho reproduces the reference's own array
        g2_face = np.zeros_like(rho_face)
        g2_face[1:] = (
            stored["g2g3_over_rhon_face"][1:] * rho_face[1:] / stored["g3_face"][1:]
        )
        vpr_face = np.asarray(stored["vpr_face"], dtype=np.float64)
        f_face = np.asarray(stored["F_face"], dtype=np.float64)
        g3_face = np.asarray(stored["g3_face"], dtype=np.float64)
        r_major = float(stored["R_major"])

        def geometry_at(psi0: np.ndarray) -> FluxSurfaceGeometry:
            return FluxSurfaceGeometry(
                rho_face=rho_face,
                rho_cell=rho_cell,
                psi_face=psi0,
                psi_n_face=rho_face,  # the conductivity table is keyed to rho
                psi_n_cell=rho_cell,
                vpr_face=vpr_face,
                vpr_cell=np.interp(rho_cell, rho_face, vpr_face),
                g2_face=g2_face,
                g3_face=g3_face,
                g3_cell=np.interp(rho_cell, rho_face, g3_face),
                f_face=f_face,
                f_cell=np.interp(rho_cell, rho_face, f_face),
                b2_cell=np.ones(n_rho),
                inv_r_cell=np.full(n_rho, 1.0 / r_major),
                phi_b=float(stored["Phi_b"]),
                r0=r_major,
                ip_amperes=float(stored["ip"][0]),
                axis_psi=float(psi0[0]),
                boundary_psi=float(psi0[-1]),
                volume=float(np.trapezoid(vpr_face, rho_face)),
                q_face=np.ones(rho_face.size),
                flux_sign=1.0,
            )

        psi = np.interp(rho_face, reference_grid, stored["psi"][0])
        psi_start = psi.copy()
        bounds = np.linspace(0, times.size - 1, 16).astype(int)
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            conductivity = np.asarray(stored["sigma_parallel"][start], dtype=np.float64)

            def resistivity(psi_n, table=conductivity):
                return 1.0 / np.interp(np.asarray(psi_n), reference_grid, table)

            psi = diffuse_psi(
                geometry_at(psi),
                resistivity,
                t_grid=times[start : end + 1],
                ip_of_t=np.asarray(stored["ip"][start : end + 1]),
            )["psi_face"][-1]

        ours = psi - psi_start
        theirs = np.interp(rho_face, reference_grid, stored["psi"][-1]) - psi_start
    residual = float(np.sqrt(np.mean((ours - theirs) ** 2)))
    scale = float(np.sqrt(np.mean(theirs**2)))
    assert scale > 0
    assert residual / scale < 0.01


# --- predicted profiles + projection ---------------------------------------
def test_predicted_current_integrates_to_the_plasma_current():
    geometry = _circular_geometry()
    eta = EtaProfile(eta0=1.0e-7, contrast=0.0, shape=1.0)
    times = np.linspace(0.0, 0.02, 101)
    solver = CurrentDiffusion(geometry, eta)
    step = solver.evolve(times, np.full(times.size, geometry.ip_amperes))
    prediction = solver.predict(step)
    surface_per_rho = geometry.vpr_cell * geometry.inv_r_cell / (2.0 * np.pi)
    integrated = float(
        np.sum(prediction["j_tor"] * surface_per_rho * np.diff(geometry.rho_face))
    )
    assert abs(integrated - prediction["i_face"][-1]) / geometry.ip_amperes < 1e-9
    assert (
        abs(prediction["i_face"][-1] - geometry.ip_amperes) / geometry.ip_amperes < 2e-2
    )


def _shaped_geometry() -> FluxSurfaceGeometry:
    """A tight-aspect metric set: the same rig with radially varying metrics.

    On the large-aspect circular limit the two profile families are exactly
    collinear in both current targets (see
    :func:`test_large_aspect_circular_limit_cannot_split_the_families`), so the
    split leverage only exists where the metrics actually vary: concentric
    circular surfaces at a tight aspect ratio give ``<1/R> = 1/sqrt(R0^2 - r^2)``
    and ``<1/R^2> = 1/(R0^2 - r^2)``, a diamagnetic ``F(psi_n)`` breaks the
    remaining proportionality, and ``<B^2>`` carries its POLOIDAL part
    ``(mu0 I(r) / 2 pi r)^2`` -- the term the large-aspect limit drops and the one
    that makes the parallel Ohm's law see the families differently.
    """
    minor_radius, r0, b0, ip = 0.45, 0.9, 0.55, 5.0e5
    base = _circular_geometry(a=minor_radius, r0=r0, b0=b0, ip=ip)
    minor = minor_radius * base.rho_cell
    inv_r = 1.0 / np.sqrt(r0**2 - minor**2)
    inv_r2 = 1.0 / (r0**2 - minor**2)
    f_cell = r0 * b0 * (1.0 + 0.35 * (1.0 - base.psi_n_cell))
    b_poloidal = MU0 * ip * base.rho_cell / (2.0 * np.pi * minor_radius)
    return FluxSurfaceGeometry(
        **{
            **base.__dict__,
            "inv_r_cell": inv_r,
            "g3_cell": inv_r2,
            "f_cell": f_cell,
            "b2_cell": b_poloidal**2 + f_cell**2 * inv_r2,
        }
    )


def test_projection_round_trips_a_known_coefficient_vector():
    """Current profiles built FROM a coefficient vector must project back to it."""
    geometry = _shaped_geometry()
    scale = np.array([2.0e6, 1.5e6, 8.0e5])
    images = basis_projection_images(
        geometry, scale, n_pressure=2, n_diamagnetic=1, nonneg=True
    )
    truth = np.array([0.6, 0.25, 0.4])
    recovered = project_coefficients(
        geometry,
        images,
        images["a_tor"] @ truth,
        images["a_par"] @ truth,
        nonneg=True,
    )
    assert recovered is not None
    # the ridge term biases the recovery at the 1e-3 level; anything looser would
    # mean the two families are not actually separated by these metrics
    np.testing.assert_allclose(recovered, truth, rtol=1e-3, atol=1e-6)


def test_the_two_current_targets_weight_the_families_differently():
    """The parallel Ohm's law carries split information the toroidal current does
    not: per unit coefficient the two targets weight the pressure-gradient and
    diamagnetic families by different, radially varying factors."""
    geometry = _shaped_geometry()
    images = basis_projection_images(
        geometry, np.ones(3), n_pressure=2, n_diamagnetic=1, nonneg=True
    )
    ratio = images["a_par"] / images["a_tor"]
    # the pressure family's weight is F/R0 and the diamagnetic family's is
    # <B^2> R0/F: neither is constant, and they do not coincide
    assert not np.allclose(ratio[:, 0], ratio[0, 0], rtol=1e-3)
    assert not np.allclose(ratio[:, 0], ratio[:, 2], rtol=1e-3)


def test_large_aspect_circular_limit_cannot_split_the_families():
    """In the large-aspect circular limit ``F/R0`` and ``<B^2> R0/F`` are both
    ``B0`` and the toroidal weights are both one, so a pressure-gradient term and
    a diamagnetic term of the same shape are indistinguishable: only their
    current-weighted SUM is identified.  This is the degeneracy the varying
    metrics of a real equilibrium break."""
    geometry = _circular_geometry()
    scale = np.array([2.0e6, 8.0e5])
    images = basis_projection_images(
        geometry, scale, n_pressure=1, n_diamagnetic=1, nonneg=True
    )
    # the two columns are the same shape up to their coefficient scales, in BOTH
    # targets: the pair spans a one-dimensional space
    np.testing.assert_allclose(
        images["a_tor"][:, 0] / scale[0], images["a_tor"][:, 1] / scale[1]
    )
    np.testing.assert_allclose(
        images["a_par"][:, 0] / scale[0], images["a_par"][:, 1] / scale[1]
    )
    truth = np.array([0.6, 0.4])
    recovered = project_coefficients(
        geometry,
        images,
        images["a_tor"] @ truth,
        images["a_par"] @ truth,
        nonneg=True,
    )
    assert recovered is not None
    assert not np.allclose(recovered, truth, rtol=1e-2)
    np.testing.assert_allclose(recovered @ scale, truth @ scale, rtol=1e-4)


def test_profile_shapes_vanish_at_the_edge_in_both_arms():
    psi_n = np.linspace(0.0, 1.0, 21)
    monomial = profile_shapes(psi_n, 3, nonneg=True)
    legendre = profile_shapes(psi_n, 3, nonneg=False)
    assert monomial.shape == (21, 3) and legendre.shape == (21, 3)
    np.testing.assert_allclose(monomial[-1], 0.0)
    np.testing.assert_allclose(legendre[-1], 0.0)
    assert np.all(monomial[:-1] > 0.0)  # the non-negative arm is sign-definite


@pytest.mark.parametrize("nonneg", [True, False])
@pytest.mark.parametrize("n_terms", [0, 1, 2, 3, 4, 5])
def test_the_two_profile_ladders_are_one_family(nonneg, n_terms):
    """The host and traced ladders must not drift apart.

    The two are written differently on purpose -- the host arm calls
    ``numpy.polynomial.legendre.legval`` for each degree, the traced arm walks
    Bonnet's recurrence so it stays a fixed-shape differentiable reduction -- so
    nothing but a test holds them to the same family. Coefficients fitted
    against one and evaluated against the other would otherwise be silently
    wrong by the difference.
    """
    pytest.importorskip("jax.numpy")
    from nova.transport.current_diffusion import _traced_profile_shapes

    psi_n = np.linspace(0.0, 1.0, 33)
    host = profile_shapes(psi_n, n_terms, nonneg=nonneg)
    traced = np.asarray(
        _traced_profile_shapes(_f64(psi_n), n_terms, nonnegative=nonneg)
    )
    assert host.shape == traced.shape == (psi_n.size, n_terms)
    np.testing.assert_allclose(traced, host, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("nonneg", [True, False])
def test_both_ladders_clip_outside_the_unit_interval(nonneg):
    """Off-interval flux is clipped identically, not extrapolated."""
    pytest.importorskip("jax.numpy")
    from nova.transport.current_diffusion import _traced_profile_shapes

    psi_n = np.array([-0.5, -0.1, 0.0, 0.5, 1.0, 1.4])
    host = profile_shapes(psi_n, 4, nonneg=nonneg)
    traced = np.asarray(_traced_profile_shapes(_f64(psi_n), 4, nonnegative=nonneg))
    np.testing.assert_allclose(traced, host, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(host[0], host[2])  # clipped to the axis value
    np.testing.assert_allclose(host[-1], host[-2])  # clipped to the edge value


def test_a_degenerate_projection_returns_none_rather_than_a_guess():
    geometry = _circular_geometry()
    images = basis_projection_images(
        geometry, np.ones(2), n_pressure=1, n_diamagnetic=1, nonneg=True
    )
    nan_target = np.full(geometry.rho_cell.size, np.nan)
    assert project_coefficients(geometry, images, nan_target, nan_target) is None
