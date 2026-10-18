"""The recorded-axis check must expose a nonstationary spline rebuild."""

from __future__ import annotations

import unittest

import numpy as np

from analyse import REPORT
from field import FastField
from page import require_rebuild
from records import parse_record


class RebuildCheckTest(unittest.TestCase):
    def test_synthetic_recorded_axes_are_refused(self):
        for side in ("a", "b"):
            with self.subTest(side=side):
                record = parse_record(REPORT / "crossing" / f"synthetic_crossing_{side}.txt")
                field = FastField(record.rgrid, record.zgrid, record.psi)
                value, gradient = field.value_gradient(
                    np.array([record.axis_r]), np.array([record.axis_z])
                )
                row = {
                    "case": f"synthetic-crossing-{side.upper()}",
                    "rebuild": {
                        "axis_psi_abs_diff": abs(float(value[0]) - record.axis_psi),
                        "grad_norm_at_axis": float(np.linalg.norm(gradient[0])),
                    },
                }
                with self.assertRaisesRegex(AssertionError, "spline rebuild failed"):
                    require_rebuild(row)


if __name__ == "__main__":
    unittest.main()
