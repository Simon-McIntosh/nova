"""A wall contact must never acquire the stationary-saddle painter or label."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import analyse
import plot as adjudication_plot


def _record():
    rs = np.linspace(-2.0, 2.0, 9)
    zs = np.linspace(-2.0, 2.0, 9)
    return SimpleNamespace(
        rgrid=rs,
        zgrid=zs,
        psi=rs[:, None] ** 2 - zs[None, :] ** 2,
        wall_r=np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]]),
        wall_z=np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]),
    )


def _adapter(r, z):
    return {"nova_saddle_r": r, "nova_saddle_z": z, "nova_saddle_psi": r * r - z * z}


@pytest.mark.parametrize(
    "position,expected",
    [
        ((0.0, 0.0), "stationary saddle"),
        ((1.0, 0.0), "wall contact"),
        ((0.5, 0.0), "neither"),
    ],
)
def test_hyperbolic_hessian_requires_stationarity(position, expected):
    point = analyse.classify_boundary_point(_record(), _adapter(*position))
    assert point["eig"][0] < 0 < point["eig"][1]
    assert point["class"] == expected
    assert point["relative_gradient"] == pytest.approx(
        point["grad_norm"] * point["length_scale"] / point["flux_span"]
    )


def test_stationary_extremum_is_not_a_saddle():
    record = _record()
    record.psi = record.rgrid[:, None] ** 2 + record.zgrid[None, :] ** 2
    point = analyse.classify_boundary_point(record, _adapter(0.0, 0.0))
    assert point["relative_gradient"] < point["gradient_tolerance"]
    assert point["class"] == "neither"


def test_classification_is_flux_scale_and_gauge_independent():
    record = _record()
    original = analyse.classify_boundary_point(record, _adapter(1.0, 0.0))
    record.psi = -3.0 * record.psi + 17.0
    transformed = analyse.classify_boundary_point(record, _adapter(1.0, 0.0))
    assert transformed["class"] == original["class"] == "wall contact"
    assert transformed["relative_gradient"] == pytest.approx(
        original["relative_gradient"]
    )


def test_wall_contact_is_refused_by_saddle_painter():
    point = {"r": 1.0, "z": 0.0, "class": "wall contact"}
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="only a stationary saddle"):
            adjudication_plot._marked_saddle(ax, point, color="brown", marker="X")
        assert not ax.lines
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "kind,marker,label",
    [
        ("wall contact", "s", "nova wall contact"),
        ("stationary saddle", "X", "nova stationary saddle"),
        ("neither", "o", "nova other boundary point"),
    ],
)
def test_receipt_class_controls_marker_and_legend(kind, marker, label):
    fig, ax = plt.subplots()
    try:
        point = {"r": 1.0, "z": 0.0, "class": kind}
        drawn = adjudication_plot._draw_boundary_point(ax, point)
        assert drawn == label
        assert ax.lines[-1].get_marker() == marker
        handles = adjudication_plot._legend_handles({drawn})
        assert [handle.get_label() for handle in handles] == [label]
        assert handles[0].get_marker() == marker
    finally:
        plt.close(fig)


def test_trace_marks_include_both_contacts_and_stationary_saddles():
    from collections import Counter

    adapters = analyse.load_adapter_a()
    classes = Counter()
    for case, path, _ in analyse.build_cases():
        if case.startswith("jt60sa"):
            continue
        adapter = adapters.get(case.lower())
        point = analyse.classify_boundary_point(analyse.parse_record(path), adapter)
        if point:
            classes[point["class"]] += 1
    assert classes == {"stationary saddle": 6, "wall contact": 8}
