"""Fixed-shape finite-arc driver and tile registry contract."""

import numpy as np
import pytest

from nova.biot.polygon import pad_batch
from nova.biot.polygonarc import packed_arc_greens, polygon_arc_greens
from nova.biot.tiledassembly import TilePlan, tile_evaluator


def rectangle():
    return np.array([[5.9, -0.1], [6.1, -0.1], [6.1, 0.1], [5.9, 0.1]])


def diamond():
    return np.array([[6.0, -0.12], [6.12, 0.0], [6.0, 0.12], [5.88, 0.0]])


def test_the_packed_arc_matches_the_shortcut_host_driver():
    sections = [rectangle(), diamond()]
    edge, weight, norm = pad_batch(sections)
    target_r = np.array([5.7, 6.3, 5.8, 6.2])
    target_z = np.array([0.2, -0.2, 0.0, 0.1])
    target_phi = np.array([0.2, 1.3, 2.1, -0.4])
    source = np.array([0, 0, 1, 1])
    start = np.array([0.4, 0.4, -0.2, -0.2])
    end = np.array([2.1, 2.1, 1.4, 1.4])
    got = np.stack(
        packed_arc_greens(
            np,
            target_r,
            target_z,
            target_phi,
            edge[:, :, source],
            weight[:, source],
            norm[source],
            start,
            end,
        )
    )
    expected = np.column_stack(
        [
            polygon_arc_greens(
                target_r[index],
                target_z[index],
                target_phi[index],
                sections[source[index]],
                start[index],
                end[index],
            )
            for index in range(len(target_r))
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=3e-11, atol=5e-19)


def test_the_registry_rejects_incompatible_arc_routes_before_compiling():
    plan = TilePlan(2, 2, 1, 1, 1)
    with pytest.raises(ValueError, match="only the closed"):
        tile_evaluator(plan, geometry="arc", kernel="quadrature")
    with pytest.raises(ValueError, match="require batched"):
        tile_evaluator(plan, geometry="arc", kernel="closed", batched=False, devices=2)
    with pytest.raises(ValueError, match="unknown geometry"):
        tile_evaluator(plan, geometry="helix")
