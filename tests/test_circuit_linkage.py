"""Two-section flux linkage between conductor circuits.

The load-bearing contracts:

* the observer sub-gridding rule is MACHINE-AGNOSTIC -- scaling every conductor
  section scales the rule by the same factor, so nothing metre-level is baked in;
* a small section stays centroid-linked, a large one is subdivided area-faithfully;
* far apart, section averaging changes nothing (the flux is uniform across the
  element); close up on a large element it corrects the centroid link, and the
  two-section matrix stays symmetric (reciprocity);
* a winding pack enclosed by a thin-wall case couples strongly but below unity;
* a thin shell's ring resistance uses its TRUE cross-section, never the
  flux-integration floor;
* a same-channel merge that averages on the source side averages on the observer
  side too.
"""

from __future__ import annotations

import numpy as np

from nova.biot.greens import hybrid_greens
from nova.circuit.conductor import ConductorSet
from nova.circuit.linkage import (
    SECTION_FLOOR,
    circuit_linkage_matrix,
    drive_linkage,
    linked_flux_columns,
    median_section_scale,
    ring_resistance,
    section_grid,
    section_points,
)
from nova.circuit.passive import NOMINAL_STEEL_RESISTIVITY


def _conductors(rows) -> ConductorSet:
    """Build a conductor set from ``(circuit, r, z, dr, dz, share)`` rows."""
    columns = np.asarray(rows, dtype=np.float64)
    return ConductorSet(
        circuit=columns[:, 0].astype(np.int64),
        r=columns[:, 1],
        z=columns[:, 2],
        dr=columns[:, 3],
        dz=columns[:, 4],
        current_share=columns[:, 5],
    )


def test_section_points_small_element_stays_at_the_centroid():
    point_r, point_z, weight = section_points(1.2, -0.3, 0.03, 0.04, 0.05, 6)
    assert point_r.size == 1
    np.testing.assert_allclose([point_r[0], point_z[0], weight[0]], [1.2, -0.3, 1.0])


def test_section_points_large_element_grid_is_area_faithful():
    point_r, point_z, weight = section_points(1.0, 0.5, 0.24, 0.42, 0.05, 6)
    assert point_r.size == 5 * 6  # ceil(0.24/0.05) = 5; ceil(0.42/0.05) capped at 6
    np.testing.assert_allclose(weight.sum(), 1.0)
    np.testing.assert_allclose([point_r.mean(), point_z.mean()], [1.0, 0.5])
    assert np.all(np.abs(point_r - 1.0) < 0.12)
    assert np.all(np.abs(point_z - 0.5) < 0.21)


def test_median_section_scale_is_machine_intrinsic():
    dr = np.array([0.04, 0.06])
    dz = np.array([0.05, 0.06])
    scale = median_section_scale(dr, dz)
    assert np.sqrt(0.04 * 0.05) <= scale <= 0.06
    # scaling every section by two scales the rule by two: no metre-level lock-in
    np.testing.assert_allclose(
        median_section_scale(2.0 * dr, 2.0 * dz), 2.0 * scale, rtol=1e-12
    )


def test_section_averaged_linkage_matches_centroid_far_field_and_is_symmetric():
    far = _conductors(
        [
            (0, 0.6, -1.4, 0.03, 0.03, 1.0),
            (1, 1.6, 1.5, 0.03, 0.03, 1.0),
        ]
    )
    groups = far.rows([0, 1])
    point_r, point_z, weight, owner = section_grid(far, groups, 0.05, 6)
    linked = linked_flux_columns(far, groups[1], point_r, point_z, weight, owner, 2)[0]
    centroid = hybrid_greens(np.array([0.6]), np.array([-1.4]), 1.6, 1.5, 0.03, 0.03)[0]
    np.testing.assert_allclose(linked, float(centroid[0]), rtol=1e-10)

    near = _conductors(
        [
            (0, 0.9, 0.0, 0.235, 0.416, 1.0),
            (1, 1.05, 0.15, 0.05, 0.05, 1.0),
        ]
    )
    groups = near.rows([0, 1])
    point_r, point_z, weight, owner = section_grid(near, groups, 0.05, 6)
    averaged = linked_flux_columns(near, groups[1], point_r, point_z, weight, owner, 2)[
        0
    ]
    centroid = hybrid_greens(np.array([0.9]), np.array([0.0]), 1.05, 0.15, 0.05, 0.05)[
        0
    ]
    assert abs(averaged - float(centroid[0])) > 1e-3 * abs(averaged)
    # reciprocity: swapping source and observer agrees to quadrature accuracy
    swapped = linked_flux_columns(near, groups[0], point_r, point_z, weight, owner, 2)[
        1
    ]
    np.testing.assert_allclose(averaged, swapped, rtol=2e-3)


def test_winding_enclosed_by_a_thin_case_couples_strongly_below_unity():
    """A winding pack boxed by four thin plates must couple strongly -- the
    shielding physics -- with a symmetric mutual and a coupling coefficient
    strictly below one.  The finite-area kernel handles the enclosing-observer
    configuration with no special casing: the plates sit OUTSIDE the winding
    section and the kernel is smooth everywhere, conductor interiors included."""
    conductors = _conductors(
        [
            (0, 1.500, 1.104, 0.159, 0.158, 1.0),
            (1, 1.4064, 1.0985, 0.0030, 0.1870, 0.25),
            (1, 1.4984, 1.1935, 0.1870, 0.0030, 0.25),
            (1, 1.5934, 1.1015, 0.0030, 0.1870, 0.25),
            (1, 1.5014, 1.0065, 0.1870, 0.0030, 0.25),
        ]
    )
    groups = conductors.rows([0, 1])
    point_r, point_z, weight, owner = section_grid(conductors, groups, 0.03, 6)
    l_winding, mutual = linked_flux_columns(
        conductors, groups[0], point_r, point_z, weight, owner, 2
    )
    mutual_swapped, l_case = linked_flux_columns(
        conductors, groups[1], point_r, point_z, weight, owner, 2
    )
    np.testing.assert_allclose(mutual, mutual_swapped, rtol=5e-3)
    coupling = mutual / np.sqrt(l_winding * l_case)
    assert 0.7 < coupling < 1.0, f"winding-case coupling {coupling:.3f}"
    assert l_winding > 0 and l_case > 0


def test_thin_shell_resistance_uses_the_true_cross_section():
    """The section floor guards the flux integration only: a 3 mm wall must carry
    the resistance of its TRUE section, over three times the floored value."""
    conductors = _conductors([(0, 1.5, 0.0, 0.003, 0.187, 0.25)])
    resistance = ring_resistance(conductors, [0], NOMINAL_STEEL_RESISTIVITY)[0]
    true = 2.0 * np.pi * 1.5 * NOMINAL_STEEL_RESISTIVITY / (0.003 * 0.187) * 0.25**2
    floored = (
        2.0
        * np.pi
        * 1.5
        * NOMINAL_STEEL_RESISTIVITY
        / (SECTION_FLOOR * 0.187)
        * 0.25**2
    )
    np.testing.assert_allclose(resistance, true, rtol=1e-12)
    assert true / floored > 3.0


def test_linkage_matrix_is_symmetric_and_positive():
    conductors = _conductors(
        [
            (0, 0.4, 0.0, 0.02, 1.8, 1.0),
            (1, 1.8, 0.0, 0.02, 1.8, 1.0),
            (2, 1.1, 0.9, 1.4, 0.02, 1.0),
        ]
    )
    lmat = circuit_linkage_matrix(conductors, [0, 1, 2])
    np.testing.assert_allclose(lmat, lmat.T, rtol=1e-12)
    assert np.all(np.diag(lmat) > 0)
    assert np.all(np.linalg.eigvalsh(lmat) > 0)


def test_drive_linkage_averages_a_merged_channel_on_both_sides():
    """A channel wired to two redundant circuits is averaged on the source side,
    so the observer side must average too -- otherwise the merged channel's self
    linkage doubles."""
    conductors = _conductors(
        [
            (0, 1.0, 0.5, 0.05, 0.05, 1.0),
            (1, 1.0, -0.5, 0.05, 0.05, 1.0),
            (2, 1.6, 0.0, 0.05, 0.05, 1.0),
        ]
    )
    single = drive_linkage(conductors, {"a_current": [0], "c_current": [2]})[1]
    merged_channels, merged = drive_linkage(
        conductors, {"a_current": [0, 1], "c_current": [2]}
    )
    assert merged_channels == ["a_current", "c_current"]
    # the merged self term is the mean of the two circuits' own linkages, which
    # for the symmetric pair sits between the single-circuit self and mutual
    assert merged[0, 0] < single[0, 0]
    # the untouched channel's self term is unchanged by another channel's merge
    np.testing.assert_allclose(merged[1, 1], single[1, 1], rtol=1e-12)
