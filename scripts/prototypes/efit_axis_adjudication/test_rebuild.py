"""The recorded-axis check must expose a nonstationary spline rebuild."""

from __future__ import annotations

import unittest
import importlib.util
import os
from pathlib import Path

import numpy as np

from analyse import REPORT
from field import FastField
from records import parse_record


page_path = Path(
    os.environ.get("ADJUDICATION_PAGE_PATH", Path(__file__).with_name("page.py"))
)
page_spec = importlib.util.spec_from_file_location("adjudication_page", page_path)
assert page_spec is not None and page_spec.loader is not None
page = importlib.util.module_from_spec(page_spec)
page_spec.loader.exec_module(page)
print(f"measurement_module {page.__file__}")
require_rebuild = page.require_rebuild


class RebuildCheckTest(unittest.TestCase):
    def test_synthetic_recorded_axes_are_refused(self):
        for side in ("a", "b"):
            with self.subTest(side=side):
                record = parse_record(
                    REPORT / "crossing" / f"synthetic_crossing_{side}.txt"
                )
                field = FastField(record.rgrid, record.zgrid, record.psi)
                value, gradient = field.value_gradient(
                    np.array([record.axis_r]), np.array([record.axis_z])
                )
                row = {
                    "case": f"synthetic-crossing-{side.upper()}",
                    "polarity": record.polarity,
                    "rebuild": {
                        "axis_psi_reported": record.axis_psi,
                        "axis_psi_spline": float(value[0]),
                        "axis_psi_abs_diff": abs(float(value[0]) - record.axis_psi),
                        "grad_norm_at_axis": float(np.linalg.norm(gradient[0])),
                        "status": "rebuild failed",
                    },
                }
                with self.assertRaisesRegex(AssertionError, "spline rebuild failed"):
                    require_rebuild(row)
                self.assertIn(
                    "<strong>rebuild failed</strong>", page.rebuild_table([row])
                )


if __name__ == "__main__":
    unittest.main()
