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
    # ramp branch: a = -M rate tau (1 - exp(-t/tau))
    analytic_ramp = -linkage * rate * tau[0] * (1.0 - decaying)
    # hold branch: decay from the ramp-end amplitude
    peak_time = ramp
    peak = -linkage * rate * tau[0] * (1.0 - np.exp(-peak_time / tau[0]))
    analytic_hold = peak * np.exp(-(times - peak_time) / tau[0])
    analytic = np.where(ramp_mask, analytic_ramp, analytic_hold)
    error = np.max(np.abs(numeric - analytic)) / max(abs(peak), 1e-30)
    return {
        "tau_s": float(tau[0]),
        "peak_current_a": float(peak),
        "analytic_peak_a": float(peak),
        "relative_error": float(error),
        "resolved": bool(error < 1e-6),
    }


def dina_loop_centroids(passive) -> tuple[np.ndarray, list[str]]:
    """One (r, z) centroid and name per DINA pf_passive loop.

    The element geometry is read through nova's own section reader, the single
    owner of the IMAS geometry conventions.
    """
    from nova.imas.machine import CrossSection

    centroids = []
    names = []
    for loop in passive.loop:
        points = []
        for element in loop.element:
            centroid = CrossSection(element.geometry).data.poly.centroid
            points.append([float(centroid.x), float(centroid.y)])
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            centroids.append([np.nan, np.nan])
        else:
            centroids.append(points.mean(axis=0))
        names.append(str(loop.name).strip())
    return np.asarray(centroids, dtype=np.float64), names


def assign_to_loops(
    circuit_centroids: np.ndarray, loop_centroids: np.ndarray
) -> np.ndarray:
    """Assign each passive circuit to the nearest DINA loop centroid."""
    d2 = (circuit_centroids[:, 0][:, None] - loop_centroids[:, 0][None, :]) ** 2 + (
        circuit_centroids[:, 1][:, None] - loop_centroids[:, 1][None, :]
    ) ** 2
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    return np.argmin(d2, axis=1)


def group_comparison(
    times: np.ndarray,
    i_passive: np.ndarray,
    circuit_centroids: np.ndarray,
    loop_centroids: np.ndarray,
    loop_names: list[str],
    dina_times: np.ndarray,
    dina_currents: np.ndarray,
    *,
    absolute_a: float,
    relative: float,
) -> dict:
    """Sum evolved circuit currents per DINA loop and compare with DINA's own."""
    owner = assign_to_loops(circuit_centroids, loop_centroids)
    rows = []
    for index, name in enumerate(loop_names):
        members = np.flatnonzero(owner == index)
        dina = dina_currents[:, index]
        if members.size == 0:
            rows.append(
                {"name": name, "members": 0, "state": "no nova circuit assigned"}
            )
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
        "worst": max(compared, key=lambda row: row["residual_peak_a"]),
        "rows": rows,
    }
