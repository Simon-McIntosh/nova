"""The reference label follows the stored outline, even with distant X-points."""

import imas
import numpy as np

from scripts.prototypes.tcv_liuqe_known_answer import measure


class _Field:
    def value(self, point):
        return np.asarray(point)[0]


def test_stored_xpoints_outside_limited_boundary_do_not_make_it_diverted():
    db = imas.DBEntry(
        f"imas:hdf5?path={measure.SOURCE_LIUQE_STORE}", "r", dd_version="3.41.0"
    )
    equilibrium = db.get("equilibrium")
    assert equilibrium.ids_properties.homogeneous_time == 1
    view = measure.slice_view(equilibrium, 0)
    db.close()
    assert len(view["xpoints"]) == 2
    view["wall_rings"] = measure.read_wall(
        "/home/ITER/mcintos/Code/imas-efit/tests/data/imas/TCV/77549"
    )

    reference = measure.liuqe_reference(view, _Field())

    assert reference["liuqe_label"] == "limited"
    assert reference["liuqe_wall_distance"] < 0.002
    assert reference["liuqe_xpoint_distance"] > 0.02
