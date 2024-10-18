import itertools
import numpy as np
import pytest
import scipy.spatial.transform

from nova.frame.coilset import CoilSet


def arc(radius, theta, angles=(0, 0, 0), number=500):
    """Return arc in the x-y plane."""
    theta = np.linspace(0, theta, number)
    points = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=-1
    )
    if angles == (0, 0, 0):
        return points
    rotation = scipy.spatial.transform.Rotation.from_euler("xyz", angles)
    return rotation.apply(points)


def points(radius, theta, angle_rng=0, number=250):
    """Return transformed arc multi-line."""
    if angle_rng == 0:
        angles = (0, 0, 0)
    else:
        rng = np.random.default_rng(angle_rng)
        angles = tuple(np.pi * (2 * rng.random(3) - 1))
    return arc(radius, theta, angles, number)


def coilset(segment, radius, theta, angle_rng=0, Ic=1e3):
    """Return coilset segment: {circle, cylinder, line, arc}."""
    coilset = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay", "Az"])
    match segment:
        case "circle" | "cylinder":
            assert angle_rng == 0
            assert np.isclose(theta, 2 * np.pi)
            coilset.coil.insert(radius, 0, 0.05, 0.05, segment=segment, Ic=Ic)
        case "arc" | "line":
            minimum_arc_nodes = number = {"arc": 3, "line": 500}[segment]
            if segment == "line":
                minimum_arc_nodes += 1
            coilset.winding.insert(
                points(radius, theta, angle_rng, number),
                {"c": (0, 0, 0.05)},
                minimum_arc_nodes=minimum_arc_nodes,
                Ic=Ic,
                rdp_eps=1e-6,
            )
    coilset.grid.solve(20, limit=[0.1 * radius, 0.85 * radius, -1, 1])
    return coilset


def _allclose_field_attrs(coilset_a, coilset_b, plot=False):
    for attr in coilset_a.field_attrs:
        _attr = attr.lower()
        if plot:
            coilset_a.grid.set_axes("2d")
            levels = coilset_a.grid.plot(_attr, colors="C0")
            coilset_b.grid.plot(_attr, levels=levels, colors="C3", linestyles="--")
            coilset_a.grid.axes.set_title(attr)
        assert np.allclose(
            getattr(coilset_a.grid, _attr),
            getattr(coilset_b.grid, _attr),
            atol=1e-4,
            rtol=1e-4,
        )


@pytest.mark.parametrize(
    "segments, radius",
    itertools.product(
        itertools.combinations(["arc", "line", "circle", "cylinder"], 2),
        [3.1, 5.3],
    ),
)
def test_segment_field_attrs_axisymmetric(segments, radius, plot=False):
    coilset_a = coilset(segments[0], radius, 2 * np.pi)
    coilset_b = coilset(segments[1], radius, 2 * np.pi)
    _allclose_field_attrs(coilset_a, coilset_b, plot)


@pytest.mark.parametrize(
    "theta, angle_rng",
    itertools.product(
        np.pi * np.array([0.1, 0.5, 3 / 4]),
        range(1, 10),
    ),
)
def test_segment_field_attrs_non_axisymmetric(theta, angle_rng, plot=False):
    coilset_a = coilset("arc", 9.2, theta, angle_rng)
    coilset_b = coilset("line", 9.2, theta, angle_rng)
    _allclose_field_attrs(coilset_a, coilset_b, plot)


if __name__ == "__main__":
    # pytest.main([__file__])

    test_segment_field_attrs_axisymmetric(("arc", "line"), 3.1)

    """
    import pprofile

    profiler = pprofile.Profile()
    with profiler:
        test_segment_field_attrs_non_axisymmetric(0.1 * np.pi, 3)
    # Process profile content: generate a cachegrind file and send it to user.

    # You can also write the result to the console:
    profiler.print_stats()

    # test_segment_field_attrs_axisymmetric(("circle", "line"), 3.1, False)
    """

"""
def vtk(coilset):
    import pyvista as pv
    import vedo

    points = np.stack(
        [
            coilset.grid.data.x2d,
            np.zeros_like(coilset.grid.data.x2d),
            coilset.grid.data.z2d,
        ],
        axis=-1,
    ).reshape(-1, 3)

    mesh = pv.PolyData(points).delaunay_2d()
    contours = mesh.contour(isosurfaces=71, scalars=coilset.grid.bx.reshape(-1))

    coilset.frame.vtkplot()
    vedo.Mesh(contours, c="black").show(new=False)
"""
