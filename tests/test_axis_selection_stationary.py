"""Collect the archived stationary-axis coordinate-binding guard on CPU."""

import jax

from nova.jax.config import configure_dtypes
from scripts.prototypes.diiid_axis_selection.check_field_orientation import (
    field_orientation_receipt,
)


def test_native_axis_binding_preserves_stationary_maximum():
    configure_dtypes()
    assert jax.config.jax_enable_x64
    with jax.default_device(jax.devices("cpu")[0]):
        receipt = field_orientation_receipt()

    checks = receipt["checks"]
    assert checks["native_shift_below_lattice_spacing"]
    assert checks["native_hessian_negative"]
    assert checks["transposed_shift_above_0_1_m"]
