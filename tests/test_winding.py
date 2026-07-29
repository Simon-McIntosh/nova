import tempfile

import numpy as np
import pytest

from nova.frame.coilset import CoilSet
from nova.graphics.plot import Plot


def test_rect_volume_poloidal_plane():
    coilset = CoilSet(delta=0)
    theta = np.linspace(0, 2 * np.pi, 50)
    coilset.winding.insert(
        5 * np.c_[np.cos(theta), np.zeros_like(theta), np.sin(theta)],
        {"rect": [0, 0, 0.3, 0.7]},
        quadrant_segments=16,
        align="axes",
    )
    volume = 0.3 * 0.7 * 2 * np.pi * 5
    assert np.isclose(coilset.frame.volume.iloc[0], volume, 1e-2)
    assert np.isclose(coilset.subframe.volume.sum(), volume, 1e-2)


def test_rect_volume_toroidal_plane():
    coilset = CoilSet(dwinding=0)
    theta = np.linspace(0, 2 * np.pi, 50)
    coilset.winding.insert(
        5 * np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)],
        {"rect": [0, 0, 0.3, 0.7]},
        quadrant_segments=16,
        align="axes",
    )
    volume = 0.3 * 0.7 * 2 * np.pi * 5
    assert np.isclose(coilset.frame.volume.iloc[0], volume, 1e-2)
    assert np.isclose(coilset.subframe.volume.sum(), volume, 1e-2)


def test_polyplot_subframe():
    coilset = CoilSet(delta=0)
    theta = np.linspace(0, 2 * np.pi, 10)
    coilset.winding.insert(
        5 * np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)],
        {"rect": [0, 0, 0.3, 0.7]},
    )
    with Plot().test_plot():
        coilset.subframe.polyplot()


def arc_winding(cross_section, **kwargs):
    """Return a coilset carrying one thickened winding swept about the vertical."""
    coilset = CoilSet()
    angle = np.array([0.3, 1.1, 1.9])
    coilset.winding.insert(
        np.stack(
            [3 * np.cos(angle), 3 * np.sin(angle), np.full_like(angle, 0.2)], axis=-1
        ),
        cross_section,
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        filament=False,
        ifttt=False,
        **kwargs,
    )
    return coilset


@pytest.mark.parametrize(
    "cross_section,section,element",
    [
        ({"rect": (0, 0, 0.06, 0.04)}, "rectangle", "bow"),
        ({"hex": (0, 0, 0.06, 0.04)}, "hexagon", "polybow"),
        ({"disc": (0, 0, 0.06, 0.06)}, "disc", "polybow"),
    ],
)
def test_a_thickened_arc_routes_to_the_element_that_can_evaluate_its_section(
    cross_section, section, element
):
    """Every section, with no segment named at the call site.

    ``Bow`` integrates the box its width and height bound while normalising by the
    section's own area, so it is EXACT for a rectangle -- which fills its own box --
    and 4/3 out on a hexagon and 4/pi out on a disc, and cannot express a section
    that is not a rectangle at all.  So the rectangle keeps the cheap exact element
    and everything else takes the corner-by-corner one, which is the dispatch this
    asserts: the element is a function of the PROFILE, not of the segment kind alone.
    A filament winding carries no section and stays an ``arc``.
    """
    coilset = arc_winding(cross_section)
    assert np.asarray(coilset.subframe["segment"]).tolist() == [element]
    assert np.asarray(coilset.subframe["section"]).tolist() == [section]


@pytest.mark.parametrize(
    "cross_section",
    [
        {"rect": (0, 0, 0.06, 0.04)},
        {"hex": (0, 0, 0.06, 0.04)},
        {"disc": (0, 0, 0.06, 0.06)},
    ],
)
def test_the_poly_column_carries_the_area_the_section_encloses(cross_section):
    """The frame's area is the area of the polygon it actually carries.

    Every element that spreads a current over a section divides by this column, so
    it has to be the area of the shape being integrated rather than of the curve
    that shape approximates.  A generated disc is a 64-gon and under-fills its
    circle by 1.6e-03: taking the analytic circle instead would spread 1.0016
    amperes of the frame's one ampere over the polygon, a first-order error in place
    of the polygon's own second-order one.
    """
    coilset = arc_winding(cross_section)
    poly = np.asarray(coilset.subframe["poly"])[0]
    area = float(np.asarray(coilset.subframe["area"])[0])
    assert np.isclose(area, poly.poly.area, rtol=1e-12)


def test_a_hollow_winding_carries_its_core_as_a_linked_negative():
    """An annulus is superposition, not a special section.

    The outer boundary goes in as a solid section at current density ``+j`` and the
    interior boundary as a core at ``-j``, both of them shapes every thickened
    element already evaluates.  ``j`` is the annulus's, which is why both rows carry
    the annulus as their ``area``; the core's ``-1`` factor is the frame's own link
    machinery.  The member currents then sum to the conductor's.
    """
    coilset = arc_winding({"box": (0, 0, 0.06, 0.2)})
    assert len(coilset.subframe) == 2
    assert len(coilset.frame) == 1
    area = np.asarray(coilset.subframe["area"], dtype=float)
    factor = np.asarray(coilset.subframe["factor"], dtype=float)
    assert factor.tolist() == [1.0, -1.0]
    assert np.allclose(area, 0.06**2 - 0.048**2, rtol=1e-12)
    outer, core = (poly.poly.area for poly in np.asarray(coilset.subframe["poly"]))
    assert np.isclose(outer, 0.06**2, rtol=1e-12)
    assert np.isclose(core, 0.048**2, rtol=1e-12)
    density = 1.0 / area[0]  # the winding's own Ic over the annulus
    assert np.isclose(density * outer - density * core, 1.0, rtol=1e-12)


def test_a_swept_skin_keeps_its_circular_boundaries():
    """A skin is a disc with a disc removed, not a square annulus.

    Computing its area from a box would make it 4/pi too large and hand every
    element a square where the section is round. Both boundaries come from the
    section itself, so the two members are 64-gons.
    """
    coilset = arc_winding({"sk": (0, 0, 0.06, 0.2)})
    assert np.asarray(coilset.subframe["section"]).tolist() == ["skin", "skin"]
    outer, core = (poly.poly.area for poly in np.asarray(coilset.subframe["poly"]))
    exact = np.pi / 4 * 0.06**2
    assert np.isclose(outer, exact, rtol=2e-03)
    assert np.isclose(core, exact * 0.8**2, rtol=2e-03)
    area = float(np.asarray(coilset.subframe["area"])[0])
    assert np.isclose(area, outer - core, rtol=1e-12)
    assert np.isclose(area, exact * (1 - 0.8**2), rtol=2e-03)


def test_store_load():
    coilset = CoilSet(dwinding=0)
    theta = np.linspace(0, 2 * np.pi, 20)
    coilset.winding.insert(
        5 * np.c_[np.cos(theta), np.sin(theta), np.zeros_like(theta)],
        {"rect": [0, 0, 0.3, 0.7]},
    )
    with tempfile.NamedTemporaryFile() as tmp:
        coilset.filepath = tmp.name
        coilset.store()
        new_coilset = CoilSet()
        new_coilset.filepath = tmp.name
        new_coilset.load()
        coilset._clear()


if __name__ == "__main__":
    pytest.main([__file__, "-W error::PendingDeprecationWarning", "-vvv"])
