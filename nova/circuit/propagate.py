"""Exact zero-order-hold propagation of circuit eigenmode states.

In the L-orthonormal eigencoordinates of a passive circuit system every mode is
a first-order lag with its own decay time,

    da/dt + a/tau = -dPsi/dt + v,

with ``Psi`` the external flux the mode links [Wb] and ``v`` a voltage-type
drive [V] that is NOT the derivative of a linked flux (the resistive term of a
conductor wired across a winding).  Taking the linked flux piecewise-linear
between samples and the voltage piecewise-constant at the step midpoint makes
the per-step update EXACT rather than a discretisation:

    a_t = exp(-dt/tau) a_{t-1} - (tau/dt)(1 - exp(-dt/tau)) dPsi_t
                               + tau (1 - exp(-dt/tau)) vbar_t

Three evaluations of the same recurrence live here, and they agree to round-off
by construction (pinned by test):

* :func:`integrate_eddy_ode` -- arbitrary sample times, the reference form;
* :func:`zoh_mode_response` -- uniform cadence, evaluated as a linear filter so
  the per-sample Python loop disappears (a resistance calibration integrates
  ~1e5 samples x ~1e2 modes per objective evaluation);
* :func:`scan_eddy_modes` -- the same recurrence as a fixed-shape
  ``jax.lax.scan``, ``jit`` / ``vmap`` / ``grad``-safe, for batched device runs.
"""

from __future__ import annotations

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.jax.config import enable_x64

    enable_x64()

#: floor on a step interval [s] -- a repeated sample must not divide by zero
MIN_STEP = 1e-6


def integrate_eddy_ode(
    tau: np.ndarray,
    times: np.ndarray,
    psi_mode: np.ndarray,
    initial: np.ndarray | None = None,
    voltage_mode: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-ZOH integration of the mode ODE along a series of sample times.

    ``psi_mode`` ``(n_t, n_modes)`` is the external flux each mode links [Wb],
    taken piecewise-linear between samples; ``voltage_mode`` the optional
    voltage-type drive [V], taken piecewise-constant at the step midpoint.

    Returns ``(a, u)`` -- the mode state and the per-step flux swing ``-dPsi``
    (the drive feature a downstream consumer standardises).  ``a[0]`` is
    ``initial`` (zeros when omitted): the machine is quiescent before the drives
    start, so integrating from the raw stream start needs no other condition.
    """
    tau = np.asarray(tau, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    psi_mode = np.asarray(psi_mode, dtype=np.float64)
    n_t = psi_mode.shape[0]
    state = np.zeros_like(psi_mode)
    swing = np.zeros_like(psi_mode)
    if initial is not None:
        state[0] = np.asarray(initial, dtype=np.float64)
    if voltage_mode is not None:
        voltage_mode = np.asarray(voltage_mode, dtype=np.float64)
    for step in range(1, n_t):
        interval = max(float(times[step] - times[step - 1]), MIN_STEP)
        decay = np.exp(-interval / tau)
        coefficient = tau / interval * (1.0 - decay)
        swing[step] = -(psi_mode[step] - psi_mode[step - 1])
        state[step] = decay * state[step - 1] + coefficient * swing[step]
        if voltage_mode is not None:
            mid = 0.5 * (voltage_mode[step] + voltage_mode[step - 1])
            state[step] = state[step] + tau * (1.0 - decay) * mid
    return state, swing


def zoh_mode_response(
    tau: np.ndarray,
    interval: float,
    psi_mode: np.ndarray,
    voltage_mode: np.ndarray | None = None,
) -> np.ndarray:
    """Exact-ZOH mode response on a UNIFORM time grid, vectorised.

    The same recurrence as :func:`integrate_eddy_ode` with ``a[0] = 0``, but a
    linear constant-coefficient recurrence per mode once the step is uniform, so
    it evaluates as a first-order IIR filter instead of a Python loop.
    """
    from scipy.signal import lfilter

    tau = np.asarray(tau, dtype=np.float64)
    psi_mode = np.asarray(psi_mode, dtype=np.float64)
    interval = float(interval)
    swing = np.zeros_like(psi_mode)
    swing[1:] = -(psi_mode[1:] - psi_mode[:-1])
    decay = np.exp(-interval / tau)
    coefficient = tau / interval * (1.0 - decay)
    state = np.empty_like(psi_mode)
    for mode in range(psi_mode.shape[1]):
        state[:, mode] = lfilter(
            [coefficient[mode]], [1.0, -decay[mode]], swing[:, mode]
        )
    if voltage_mode is not None:
        voltage_mode = np.asarray(voltage_mode, dtype=np.float64)
        mid = np.zeros_like(voltage_mode)
        mid[1:] = 0.5 * (voltage_mode[1:] + voltage_mode[:-1])
        voltage_coefficient = tau * (1.0 - decay)
        for mode in range(psi_mode.shape[1]):
            state[:, mode] += lfilter(
                [voltage_coefficient[mode]], [1.0, -decay[mode]], mid[:, mode]
            )
    return state


def scan_eddy_modes(tau, times, psi_mode, initial=None, voltage_mode=None):
    """Exact-ZOH mode integration as a fixed-shape ``jax.lax.scan``.

    Identical recurrence and identical output to :func:`integrate_eddy_ode`, with
    every shape fixed at trace time: ``tau`` ``(n_modes,)``, ``times``
    ``(n_t,)``, ``psi_mode`` ``(n_t, n_modes)``.  ``jit`` / ``vmap`` / ``grad``
    -safe, so a batch of drive histories (or of candidate resistance models,
    through ``tau``) propagates in one device call.

    Returns ``(a, u)`` as arrays of the input dtype -- run with x64 enabled (this
    module does so on import) to match the host integrator to round-off.
    """
    tau = jnp.asarray(tau)
    times = jnp.asarray(times)
    psi_mode = jnp.asarray(psi_mode)
    start = jnp.zeros_like(psi_mode[0]) if initial is None else jnp.asarray(initial)
    intervals = jnp.maximum(jnp.diff(times), MIN_STEP)
    swing = -jnp.diff(psi_mode, axis=0)
    if voltage_mode is None:
        mid = jnp.zeros_like(swing)
    else:
        voltage_mode = jnp.asarray(voltage_mode)
        mid = 0.5 * (voltage_mode[1:] + voltage_mode[:-1])

    def step(state, carry):
        interval, step_swing, step_mid = carry
        decay = jnp.exp(-interval / tau)
        state = (
            decay * state
            + tau / interval * (1.0 - decay) * step_swing
            + tau * (1.0 - decay) * step_mid
        )
        return state, state

    _final, history = jax.lax.scan(step, start, (intervals, swing, mid))
    return (
        jnp.concatenate([start[jnp.newaxis, :], history]),
        jnp.concatenate([jnp.zeros_like(start)[jnp.newaxis, :], swing]),
    )


__all__ = [
    "MIN_STEP",
    "integrate_eddy_ode",
    "scan_eddy_modes",
    "zoh_mode_response",
]
