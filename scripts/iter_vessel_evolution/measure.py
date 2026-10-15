"""Measure the evolved vessel currents against DINA and run the controls."""

from __future__ import annotations

import numpy as np

from nova.circuit.propagate import integrate_eddy_ode


def positive_control(
    inductance: float = 1.0,
    resistance: float = 1.0,
    linkage: float = 0.5,
    rate: float = 1.0,
    ramp: float = 0.1,
    duration: float = 1.0,
    samples: int = 4001,
) -> dict:
    """A single passive ring under a coil-current ramp vs the analytic L/R solution.

    The ring is one circuit with inductance ``L`` and resistance ``R``, linked to
    the coil by ``M``.  The analytic ring current for the piecewise-linear drive
    is compared with the exact-ZOH integration of the same drive.
    """
    times = np.linspace(0.0, duration, samples)
    coil = np.clip(rate * times, 0.0, rate * ramp)
    psi = linkage * coil
    tau = np.array([inductance / resistance])
    a, _ = integrate_eddy_ode(tau, times, psi[:, None])
    numeric = a[:, 0]

    decaying = np.exp(-times / tau[0])
    ramp_mask = times <= ramp
    analytic_ramp = -linkage * rate * tau[0] * (1.0 - decaying)
    peak = -linkage * rate * tau[0] * (1.0 - np.exp(-ramp / tau[0]))
    analytic_hold = peak * np.exp(-(times - ramp) / tau[0])
    analytic = np.where(ramp_mask, analytic_ramp, analytic_hold)
    error = np.max(np.abs(numeric - analytic)) / max(abs(peak), 1e-30)
    return {
        "tau_s": float(tau[0]),
        "peak_current_a": float(peak),
        "relative_error": float(error),
        "resolved": bool(error < 1e-6),
    }


def plasma_flux_history(equilibrium, measure, points, n_slices: int):
    """Prescribed plasma flux [Wb] at ``points`` over subsampled slice times."""
    eligible = [
        index
        for index, slice_ in enumerate(equilibrium.time_slice)
        if abs(float(slice_.global_quantities.ip)) >= 1e5
    ]
    if not eligible:
        raise ValueError("no plasma-bearing equilibrium slices")
    selected = np.unique(np.linspace(eligible[0], eligible[-1], n_slices, dtype=int))
    slice_times = np.array(
        [float(equilibrium.time_slice[index].time) for index in selected]
    )
    flux = np.zeros((slice_times.size, points.shape[0]))
    for row, index in enumerate(selected):
        current = measure.prescribe_slice(equilibrium.time_slice[index])
        from nova.imas.prescribed_plasma import plasma_flux_at_points

        flux[row] = plasma_flux_at_points(current, points[:, 0], points[:, 1])
    return slice_times, flux


def dina_loop_polygons(passive) -> list:
    """One shapely polygon per DINA pf_passive loop, the union of its elements."""
    from shapely.ops import unary_union

    from nova.imas.machine import CrossSection

    polygons = []
    for loop in passive.loop:
        parts = [
            CrossSection(element.geometry).data.poly for element in loop.element
        ]
        polygons.append(unary_union(parts) if parts else None)
    return polygons


def assign_by_containment(centroids: np.ndarray, polygons: list) -> np.ndarray:
    """Map each circuit to the DINA loop whose cross-section contains it.

    ``owner == -1`` means inside no DINA loop and ``-2`` means inside more than
    one; both are unassigned rather than forced onto a nearest loop.
    """
    from shapely import contains_xy

    owner = np.full(centroids.shape[0], -1, dtype=np.int64)
    ambiguous = 0
    for index, polygon in enumerate(polygons):
        if polygon is None:
            continue
        inside = contains_xy(polygon, centroids[:, 0], centroids[:, 1])
        hit = np.flatnonzero(inside)
        multiple = hit[owner[hit] != -1]
        owner[multiple] = -2
        owner[hit[owner[hit] == -1]] = index
        ambiguous += multiple.size
    return owner, int(np.count_nonzero(owner == -1)), ambiguous


def group_comparison(
    times: np.ndarray,
    i_passive: np.ndarray,
    circuit_centroids: np.ndarray,
    loop_polygons: list,
    loop_names: list[str],
    dina_times: np.ndarray,
    dina_currents: np.ndarray,
    *,
    absolute_a: float,
    relative: float,
) -> dict:
    """Sum evolved current per DINA loop (by containment) and compare with DINA's."""
    owner, unassigned, ambiguous = assign_by_containment(
        circuit_centroids, loop_polygons
    )
    assigned = owner >= 0
    magnitude = np.abs(i_passive)
    total_magnitude = float(magnitude.sum())
    unassigned_magnitude = float(magnitude[:, ~assigned].sum())
    rows = []
    for index, name in enumerate(loop_names):
        members = np.flatnonzero(owner == index)
        dina = dina_currents[:, index]
        if members.size == 0:
            rows.append({"name": name, "members": 0, "state": "no nova circuit inside"})
            continue
        nova = i_passive[:, members].sum(axis=1)
        reference = np.interp(times, dina_times, dina)
        residual = nova - reference
        scale = max(absolute_a, relative * float(np.max(np.abs(reference))))
        rows.append(
            {
                "name": name,
                "members": int(members.size),
                "dina_peak_a": float(np.max(np.abs(reference))),
                "nova_peak_a": float(np.max(np.abs(nova))),
                "residual_rms_a": float(np.sqrt(np.mean(residual**2))),
                "residual_peak_a": float(np.max(np.abs(residual))),
                "tolerance_a": float(scale),
                "within_tolerance": bool(np.max(np.abs(residual)) <= scale),
            }
        )
    compared = [row for row in rows if "residual_peak_a" in row]
    if not compared:
        raise AssertionError("no DINA loop received a nova circuit")
    failed = [row for row in compared if not row["within_tolerance"]]
    return {
        "tolerance_rule": "peak residual <= max(abs A, rel * peak |DINA I|)",
        "absolute_a": absolute_a,
        "relative": relative,
        "loops_total": len(loop_names),
        "loops_compared": len(compared),
        "loops_within_tolerance": len(compared) - len(failed),
        "circuits_unassigned": unassigned,
        "circuits_ambiguous": ambiguous,
        "circuits_assigned": int(np.count_nonzero(assigned)),
        "unassigned_current_fraction": (
            unassigned_magnitude / total_magnitude if total_magnitude else float("nan")
        ),
        "worst": max(compared, key=lambda row: row["residual_peak_a"]),
        "rows": rows,
    }


def passive_flux_at_points(conductors, circuits, i_passive, times, points):
    """Flux [Wb] at ``points`` from the evolved passive circuit currents."""
    from nova.biot.greens import greens_psi

    rows = conductors.rows(circuits)
    owner = np.concatenate(
        [
            np.full(group.size, index, dtype=np.int64)
            for index, group in enumerate(rows)
        ]
    )
    filament = np.concatenate(rows)
    source_r = conductors.r[filament]
    source_z = conductors.z[filament]
    share = conductors.current_share[filament]
    membership = np.zeros((len(circuits), share.size))
    np.add.at(membership, (owner, np.arange(share.size)), share)
    kernel = greens_psi(
        points[:, 0], points[:, 1], source_r[:, None], source_z[:, None]
    )
    coupling = membership @ kernel
    return i_passive @ coupling


def vs3_residual(
    times: np.ndarray,
    section2_residual: np.ndarray,
    evolved_flux: np.ndarray,
    dina_flux: np.ndarray,
    *,
    tolerance_wb: float,
) -> dict:
    """Whether the evolved passive flux at the VS3 sections closes the psi-map residual.

    ``section2_residual``, ``evolved_flux`` and ``dina_flux`` are all
    ``(n_times, n_points)`` at the VS3 element centroids.  The psi-map check left
    its conductor source driven by DINA's stored ``pf_passive`` currents; the
    evolved passive currents replace them, so the residual it left open moves by
    the flux change ``evolved - dina`` at each point.  The section-2 residual is
    ``predicted - inferred`` with ``inferred = psi_map - source - gauge``, so a
    rise of that flux amount *reduces* the residual.
    """
    delta = evolved_flux - dina_flux
    after = section2_residual - delta
    per_point = []
    for index in range(section2_residual.shape[1]):
        before = float(np.max(np.abs(section2_residual[:, index])))
        after_peak = float(np.max(np.abs(after[:, index])))
        per_point.append(
            {
                "point": index,
                "section2_peak_wb": before,
                "evolved_passive_flux_peak_wb": float(
                    np.max(np.abs(evolved_flux[:, index]))
                ),
                "dina_passive_flux_peak_wb": float(
                    np.max(np.abs(dina_flux[:, index]))
                ),
                "delta_peak_wb": float(np.max(np.abs(delta[:, index]))),
                "residual_after_peak_wb": after_peak,
                "within_tolerance": bool(after_peak <= tolerance_wb),
            }
        )
    before_peak = max(row["section2_peak_wb"] for row in per_point)
    after_peak = max(row["residual_after_peak_wb"] for row in per_point)
    return {
        "points": per_point,
        "section2_peak_wb": before_peak,
        "residual_after_peak_wb": after_peak,
        "tolerance_wb": tolerance_wb,
        "residual_removed_fraction": (
            1.0 - after_peak / before_peak if before_peak else float("nan")
        ),
        "closes": bool(after_peak <= tolerance_wb and after_peak < before_peak),
    }