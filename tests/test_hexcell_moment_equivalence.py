"""Equivalence of exact cell moments and a refined plasma-element mesh."""

import pytest

from benchmarks.hex_cell_moment_equivalence import run_equivalence


@pytest.fixture(scope="module")
def equivalence():
    """Measure the shared hexagonal source through both representations."""
    return run_equivalence(refinements=(48, 120, 300, 720))


@pytest.mark.parametrize("case", ["uniform", "linear"])
@pytest.mark.parametrize("region", ["near", "far"])
@pytest.mark.parametrize("quantity", ["psi", "B_R", "B_Z"])
def test_refined_plasma_elements_converge_to_cell_moments(
    equivalence, case, region, quantity
):
    """Every field component converges in both target regimes."""
    metric = equivalence["cases"][case]["regions"][region][quantity]
    assert metric["fitted_order"] > 1.35
    assert metric["finest_fraction"] < 8.0e-3
    assert metric["fractions"][-1] < metric["fractions"][0]


@pytest.mark.parametrize("case", ["uniform", "linear"])
def test_refined_plasma_elements_preserve_total_current(equivalence, case):
    """The partition carries the analytic cell current at round-off."""
    residuals = equivalence["cases"][case]["current_relative_residuals"]
    assert max(residuals) < 5.0e-14


def test_measurement_records_all_nine_moment_blocks(equivalence):
    """The exact route explicitly exercises three blocks per quantity."""
    assert equivalence["moment_blocks"] == {
        "psi": ["uniform", "radial", "vertical"],
        "B_R": ["uniform", "radial", "vertical"],
        "B_Z": ["uniform", "radial", "vertical"],
    }
