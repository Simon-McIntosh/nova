"""The maximin certificate must reproduce each side's escape level and fire its cut.

Three rows of the escape-level comparison start from the same axis on both codes,
so each tests the boundary rule.  The certificate fixes the maximin of
polarity*psi from the axis to a target by a descending union-find on the record's
own scipy spline, and the blocking cut is the interface the axis component
leaves when the target cannot be reached at a level.  The first test pins the
measured maximin to imas-efit's escape level; the second is the negative control
and requires the cut detector to fire on a level the target cannot attain.
"""

from __future__ import annotations

import pathlib
import unittest

import numpy as np

from scripts.prototypes.efit_maximin_certificate.certificate import (
    Discrete,
    blocking_cut,
    certify,
    load_record,
    maximin,
    sample_field,
    spline_of,
    vessel_ring,
)

REPORT = pathlib.Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-efit/"
    "continuous-boundary-level/cbl-saddle-admission"
)
MASTU_B = REPORT / "traces/mastu-fullPulse/boundary_slice_1_iter_5.txt"

MASTU_CONTACT = (0.3327971398830414, 1.3030896186828613)
MASTU_AXIS = (0.8473058496500641, -0.1920736910170366)
MASTU_NOVA_LEVEL = 0.020530973894263525
MASTU_EFIT_LEVEL = 0.019652442525099203


def _discrete(path, n):
    record = load_record(path)
    spline = spline_of(record)
    ring = vessel_ring(record)
    pts, values, inside, n = sample_field(spline, ring, int(record.polarity), n)
    return Discrete(pts, values, inside, n)


class MaximinCertificateTest(unittest.TestCase):
    def test_mastu_contact_maximin_is_below_nova_level(self):
        """The maximin to nova's contact is imas-efit's level, below nova's."""
        row = certify(
            "mastu-fullPulse-s1-4-5-B",
            str(MASTU_B),
            MASTU_B,
            MASTU_AXIS,
            MASTU_CONTACT,
            MASTU_NOVA_LEVEL,
            MASTU_EFIT_LEVEL,
            "limited",
            "diverted",
            n=400,
        )
        self.assertAlmostEqual(row.maximin_axis_to_target, MASTU_EFIT_LEVEL, places=3)
        self.assertLess(row.maximin_axis_to_target, MASTU_NOVA_LEVEL)
        # the bottle is at the X-point region, not at the contact
        self.assertLess(abs(row.path_min_at[0] - 0.575), 0.03)
        self.assertLess(abs(row.path_min_at[1] - 1.123), 0.03)
        # nova's own attaining path cannot reach the contact at nova's level
        self.assertLess(row.path_min, MASTU_NOVA_LEVEL - 1e-4)

    def test_cut_detector_fires_without_errors(self):
        """Negative control: at a level above the maximin the cut must appear."""
        discrete = _discrete(MASTU_B, 400)
        ax = discrete.nearest(MASTU_AXIS)
        tg = discrete.nearest(MASTU_CONTACT)
        level = maximin(discrete, ax, tg)
        self.assertTrue(np.isfinite(level))
        exists, points = blocking_cut(discrete, ax, tg, level + 1e-3)
        self.assertTrue(exists)
        self.assertGreater(len(points), 0)
        exists_low, _ = blocking_cut(discrete, ax, tg, level - 1e-3)
        self.assertFalse(exists_low)


if __name__ == "__main__":
    unittest.main()