"""The fit inside the equilibrium iteration, on an algebraic instrument.

The instrument is a finite-dimensional stand-in for the held-map pass: cell
currents are linear in the coefficients with a flux-dependent weight, the flux
map is one fixed coupling of that current to the node set, and the measured rows
are a fixed sensor coupling plus optional conductor columns. It states the
orchestration claim only — one least-squares solve per pass lays down the
coefficients and currents the rows choose, one flux update follows from that
current, and the passes repeat until the map stops changing — and makes no
physics claim.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.flux_iteration import FluxIterationFit
from nova.equilibrium.measurement import Magnetics, SliceMeasurement
from nova.equilibrium.source import (
    DomainProfile,
    ForwardSource,
    PolynomialFluxFunction,
)
from nova.jax.config import configure_dtypes

CELLS = 3
SENSORS = 4
TRUTH = np.array([1.0, -0.4])
TRUTH_CURRENTS = np.array([2.0, -1.5])
CONDUCTOR_NAMES = ("coil_a", "coil_b")
COUPLING = np.array([[0.20, -0.10, 0.20], [-0.05, 0.25, 0.02], [0.01, 0.03, 0.30]])
PROFILE_WEIGHT = np.array([[1.0, 0.0], [0.5, -0.3], [0.0, 1.0]])
SENSOR = np.array(
    [[1.0, 0.5, -1.0], [0.2, 1.5, 0.7], [-0.4, 0.3, 2.0], [0.6, -0.2, 0.9]]
)
CONDUCTOR_RESPONSE = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-1.0, 0.7]])
EXTERNAL = np.array([1.0, -0.5, 0.25])
SCALE = np.array([0.5, 2.0, 1.0, 1.0])
LOOPS = np.array([True, False, True, False])


class AxisLost(ValueError):
    """The refusal a topology read raises when the map carries no axis."""


FLUX_WEIGHT = 0.10


def _profile_weight(psi, weight=FLUX_WEIGHT):
    return (1.0 + weight * np.asarray(psi))[:, None] * PROFILE_WEIGHT


class _Instrument:
    """Own the pass counters and the laid-down currents across rebuilds."""

    def __init__(self, flux_weight=FLUX_WEIGHT):
        self.cells_remaining = None
        self.currents = np.zeros(2)
        self.flux_weight = flux_weight

    def cell_current(self, source, psi):
        if self.cells_remaining is not None:
            self.cells_remaining -= 1
            if self.cells_remaining < 0:
                raise AxisLost("no qualified axis in the trial map")
        coefficients = np.r_[
            np.asarray(source.core.p_prime.coefficients),
            np.asarray(source.core.ff_prime.coefficients),
        ]
        return _profile_weight(psi, self.flux_weight) @ coefficients


class _Operator:
    def __init__(self, instrument, source):
        self.instrument = instrument
        self.source = source

    def cell_current(self, psi, requested_class=None, target_current=None):
        return jnp.asarray(self.instrument.cell_current(self.source, psi))

    def __call__(self, psi, current=None, requested_class=None, target_current=None):
        if current is not None:
            self.instrument.currents = np.asarray(current, dtype=np.float64)
        return jnp.asarray(
            EXTERNAL + COUPLING @ self.instrument.cell_current(self.source, psi)
        )

    def read(self, psi):
        return None, SimpleNamespace(flux_span=1.0)


class _Profile:
    def __init__(self, source, flux_weight=FLUX_WEIGHT):
        self.instrument = _Instrument(flux_weight)
        self.operator = _Operator(self.instrument, source)

    @property
    def source(self):
        return self.operator.source

    def _with_source(self, source):
        active = object.__new__(type(self))
        active.instrument = self.instrument
        active.operator = _Operator(self.instrument, source)
        return active


def _source(amplitudes):
    return ForwardSource(
        core=DomainProfile(
            p_prime=PolynomialFluxFunction(jnp.asarray([amplitudes[0]])),
            ff_prime=PolynomialFluxFunction(jnp.asarray([amplitudes[1]])),
        )
    )


def _fixed_point(amplitudes, flux_weight=FLUX_WEIGHT):
    """Solve the instrument's own map for the state those profiles support."""
    flux = np.zeros(CELLS)
    source = _source(amplitudes)
    instrument = _Instrument(flux_weight)
    for _ in range(400):
        flux = 0.5 * flux + 0.5 * (
            EXTERNAL + COUPLING @ instrument.cell_current(source, flux)
        )
    return flux


def _instrument(initial=None, conductor=False, flux_weight=FLUX_WEIGHT):
    flux = _fixed_point(TRUTH, flux_weight)
    cells = _profile_weight(flux, flux_weight) @ TRUTH
    measured = SENSOR @ cells
    geometry = Magnetics(
        r=np.arange(1.0, SENSORS + 1.0),
        z=np.zeros(SENSORS),
        angle=np.zeros(SENSORS),
        flux_loop=LOOPS,
    )

    def sensor_rows(currents):
        return SENSOR @ np.asarray(currents)

    source = _source(TRUTH if initial is None else initial)
    kwargs = {}
    if conductor:
        measured = measured + CONDUCTOR_RESPONSE @ TRUTH_CURRENTS
    measurement = SliceMeasurement(
        measured=measured,
        vacuum=np.zeros(SENSORS),
        mask=np.ones(SENSORS, dtype=bool),
        scale=SCALE,
        plasma_current=float(np.sum(cells)),
        conductor_current=TRUTH_CURRENTS if conductor else None,
        conductor_scale=np.array([0.4, 0.4]) if conductor else None,
    )
    if conductor:
        kwargs = dict(
            conductor_response=CONDUCTOR_RESPONSE,
            conductor_names=CONDUCTOR_NAMES,
        )
    fitter = FluxIterationFit(
        _Profile(source, flux_weight), geometry, sensor_rows, **kwargs
    )
    return fitter, measurement, flux, cells


@pytest.fixture(autouse=True)
def _precision():
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True


def test_the_iteration_recovers_the_state_the_rows_came_from():
    fitter, measurement, flux, cells = _instrument(initial=TRUTH * 1.05)
    receipt = fitter.fit(
        measurement,
        np.zeros(CELLS),
        plasma_current_scale=1.0,
        flux_tolerance=1.0e-12,
    )
    assert receipt.converged
    assert receipt.axis_lost_pass is None
    assert receipt.stopped_improving_pass is None
    np.testing.assert_allclose(receipt.coefficients, TRUTH, atol=1.0e-9)
    np.testing.assert_allclose(receipt.terminal_state.cell_current, cells, rtol=1e-8)
    assert receipt.terminal_misfit <= 1.0e-9
    assert receipt.conductor_currents.size == 0
    assert receipt.singular_values.shape == (TRUTH.size,)
    assert receipt.undetermined_coefficients == ()
    assert np.all(np.isfinite(receipt.terminal_state.flux))


def test_each_pass_costs_one_flux_map_per_coefficient_plus_the_residual():
    fitter, measurement, _, _ = _instrument(initial=TRUTH * 1.05)
    receipt = fitter.fit(
        measurement,
        np.zeros(CELLS),
        plasma_current_scale=1.0,
        flux_tolerance=1.0e-12,
    )
    assert receipt.flux_evaluations == len(receipt.passes)
    assert receipt.sensor_evaluations == len(receipt.passes) * (TRUTH.size + 1) + 1
    assert [item.index for item in receipt.passes] == list(range(len(receipt.passes)))
    assert receipt.last_flux_update == receipt.passes[-1].flux_update
    assert receipt.last_flux_update <= 1.0e-12
    assert receipt.passes[-1].misfit <= 1.0e-9


def test_the_fitted_conductor_currents_take_their_measured_rows():
    # A map that does not read the flux it is handed leaves the rows one exact
    # linear system in the coefficients and the fitted currents, so recovery is
    # the row ordering's claim and not a convergence claim.
    fitter, measurement, _, _ = _instrument(
        initial=TRUTH * 1.05, conductor=True, flux_weight=0.0
    )
    receipt = fitter.fit(
        measurement,
        np.zeros(CELLS),
        plasma_current_scale=1.0,
        flux_tolerance=1.0e-12,
    )
    assert receipt.converged
    np.testing.assert_allclose(receipt.coefficients[: TRUTH.size], TRUTH, atol=1e-8)
    np.testing.assert_allclose(receipt.conductor_currents, TRUTH_CURRENTS, atol=1e-8)
    assert receipt.coefficient_names[-2:] == CONDUCTOR_NAMES
    assert np.allclose(fitter.profile.instrument.currents, TRUTH_CURRENTS, atol=1e-8)


def test_a_flux_map_that_loses_its_axis_is_recorded_at_that_pass():
    fitter, measurement, _, _ = _instrument(initial=TRUTH * 1.05)
    # one residual read, one response read per coefficient and one map
    # application, which reads the held map's current through the same reader
    fitter.profile.instrument.cells_remaining = 2 * (TRUTH.size + 2)
    receipt = fitter.fit(
        measurement,
        np.zeros(CELLS),
        plasma_current_scale=1.0,
        flux_tolerance=1.0e-12,
        max_passes=8,
    )
    assert not receipt.converged
    assert receipt.axis_lost_pass == 2
    assert len(receipt.passes) == 2


def test_a_scaled_input_leaves_the_plasma_current_band():
    fitter, measurement, _, _ = _instrument(initial=TRUTH * 1.05)
    scaled = SliceMeasurement(
        measured=measurement.measured * 1.10,
        vacuum=measurement.vacuum,
        mask=measurement.mask,
        scale=measurement.scale * 1.10,
        plasma_current=measurement.plasma_current * 1.10,
    )
    receipt = fitter.fit(
        scaled,
        np.zeros(CELLS),
        plasma_current_scale=1.10,
        flux_tolerance=1.0e-12,
    )
    current = abs(float(np.sum(receipt.terminal_state.cell_current)))
    reference = abs(float(measurement.plasma_current))
    assert abs(current - reference) / reference > 0.015
