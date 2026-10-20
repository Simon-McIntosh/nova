"""The boundary event lies inside independently certified escape brackets."""

import jax
import pytest

from nova.jax.config import configure_dtypes
from scripts.prototypes.efit_field_swap.crosscheck import read_trace_record
from scripts.prototypes.efit_maximin_certificate.adapter_probe import decision
from scripts.prototypes.efit_maximin_certificate.run import ROWS


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("spec", "lower", "upper", "boundary_class"),
    [
        (ROWS[0], 0.0196524250845, 0.019652442526100, 1),
        (ROWS[1], 0.014214693, 0.014214735554751, 0),
        (ROWS[2], 0.012577131, 0.012577168750081, 0),
    ],
    ids=[row["case"] for row in ROWS],
)
def test_boundary_matches_certified_escape(spec, lower, upper, boundary_class):
    """Attaining paths and continuous event bounds enclose the escape level.

    The synthetic upper bounds enclose the whole left wall in Bernstein
    pieces of length 1e-5 with a 1e-12 arithmetic pad. The specified endpoint
    at Z=0.2 is below the nearby wall tangency and cannot bound that event.
    """
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    reading = decision(read_trace_record(spec["source"]))
    level = reading["boundary_flux"]
    assert lower <= level <= upper, (
        f"{spec['case']}: boundary {level:.17g} is outside the "
        f"certified escape interval [{lower:.17g}, {upper:.17g}]"
    )
    assert reading["boundary_class"] == boundary_class
