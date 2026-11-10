"""Regression tests for the reduced playable receipt history."""

from types import SimpleNamespace

import numpy as np

from apps.playable.production import ProductionSolver


class StubProfile:
    """Capture the history passed to the profile receipt builder."""

    def __init__(self):
        self.history = None

    def _receipt(self, state, history, *args, **kwargs):
        del state, args, kwargs
        self.history = history
        return history


def test_reduced_receipt_preserves_per_trip_support_cells():
    support_cells = np.asarray([17, 23, 31], dtype=np.int32)
    residuals = np.asarray([1.0e-2, 3.0e-3, 7.0e-4])
    result = SimpleNamespace(
        state=np.asarray([1.0]),
        active_set_residuals=residuals,
        active_set_support_cells=support_cells,
        terminal_residual=residuals[-1],
        converged=True,
        termination_reason=0,
        active_set_iterations=3,
        active_set_mask_differences=np.asarray([0, 1, 0], dtype=np.int32),
        prescribed_current=None,
        constraints=(),
    )
    profile = StubProfile()

    ProductionSolver._reduced_receipt(profile, result)

    captured = np.asarray(profile.history.active_set_support_cells)
    np.testing.assert_array_equal(captured, support_cells)
    assert captured.size == np.asarray(profile.history.active_set_residuals).size
    assert np.all(captured > 0)
