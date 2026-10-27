"""Lattice vertex groups refuse one over-wide link, never the whole group."""

from __future__ import annotations

import numpy as np

from nova.equilibrium.stencil_mesh import lattice_vertex_groups


def groups_of(points) -> list[list[int]]:
    """Return which slots share a lattice vertex, as sorted slot lists."""
    polygons = [np.asarray([point], dtype=np.float64) for point in points]
    groups, _ = lattice_vertex_groups(polygons)
    members: dict[int, list[int]] = {}
    for slot, group in enumerate(groups):
        members.setdefault(int(group), []).append(slot)
    return sorted(sorted(group) for group in members.values())


def test_one_over_wide_link_is_refused_without_dissolving_the_group():
    """A close pair stays merged when the link onwards would widen it too far.

    The merge width is one billionth of the lattice scale, and the scale is one
    metre here, so the width is ``1e-9``. Slot 1 sits six tenths of a width
    from slot 0 and slot 2 sits six tenths from slot 1, while slot 2 is one and
    a fifth widths from slot 0. The two close links are taken; the link that
    would carry the group's spread to the merge width is refused on its own, so
    slots 0 and 1 keep sharing a vertex and slot 2 stands alone.
    """
    width = 1.0e-9
    points = [(0.0, 0.0), (0.6 * width, 0.0), (1.2 * width, 0.0)]

    assert groups_of(points) == [[0, 1], [2]]


def test_a_chain_of_close_links_still_merges_every_member():
    """Every link inside the merge width joins, so the whole chain is one vertex."""
    width = 1.0e-9
    points = [(0.0, 0.0), (0.4 * width, 0.0), (0.8 * width, 0.0)]

    assert groups_of(points) == [[0, 1, 2]]


def test_reversing_the_cell_order_keeps_the_merge():
    """The refused link is chosen from the coordinates, not from the listing."""
    width = 1.0e-9
    points = [(0.0, 0.0), (0.6 * width, 0.0), (1.2 * width, 0.0)]
    forward = groups_of(points)
    backward = groups_of(list(reversed(points)))
    remap = {slot: len(points) - 1 - slot for slot in range(len(points))}
    remapped = sorted(sorted(remap[slot] for slot in group) for group in forward)
    assert remapped == backward
