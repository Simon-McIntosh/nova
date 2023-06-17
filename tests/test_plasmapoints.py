from dataclasses import dataclass, field
import pytest

import numpy as np
import xarray

from nova.geometry.plasmapoints import ControlPoints, PlasmaPoints, Points


DATA = {
    "elongation": 2.1,
    "triangularity": 0.3,
    "triangularity_upper": 0.2,
    "triangularity_lower": 0.4,
    "triangularity_inner": 0.4,
    "triangularity_outer": 0.6,
    "squareness": 0.1,
    "geometric_axis": [6, 0],
    "minor_radius": 3.2,
}


@dataclass
class DataClass:
    """Test class with a dict-like interface."""

    data: dict = field(default_factory=dict)

    def __getitem__(self, attr):
        """Return item from data."""
        return self.data[attr]

    def __setitem__(self, attr, value):
        """Update item in data."""
        self.data[attr] = value

    def __iter__(self):
        return iter(self.data)

    def copy(self):
        """Retun new class instance."""
        return DataClass(self.data.copy())


@pytest.fixture(params=["dict", "dataset", "class"])
def data(request):
    """Return point data."""
    match request.param:
        case "dict":
            return DATA.copy()
        case "dataset":
            data = DATA.copy()
            data["geometric_axis"] = ("point", data["geometric_axis"])
            return xarray.Dataset(data)
        case "class":
            return DataClass(DATA.copy())


def test_elongation_name(data):
    elongation = Points("elongation", data)
    assert elongation.name == "elongation"


def test_elongation_attrs(data):
    elongation = Points("elongation", data)
    assert elongation.attrs == ["elongation"]


def test_elongation_mean_item(data):
    elongation = Points("elongation", data)
    assert elongation.mean == data["elongation"]


def test_elongation_item_error(data):
    elongation = Points("elongation", data)
    with pytest.raises(KeyError):
        elongation["upper"]


def test_elongation_attr(data):
    elongation = Points("elongation", data)
    assert elongation.mean == data["elongation"]


def test_elongation_attr_error(data):
    elongation = Points("elongation", data)
    with pytest.raises(AttributeError):
        elongation.upper


def test_triangularity_major_attrs(data):
    triangularity = Points("triangularity", data, "major")
    assert triangularity.attrs == [
        "triangularity_upper",
        "triangularity_lower",
    ]


def test_triangularity_major_mean(data):
    triangularity = Points("triangularity", data, "major")
    assert np.isclose(triangularity.mean, data["triangularity"])


def test_triangularity_upper(data):
    triangularity = Points("triangularity", data, "major")
    assert triangularity["upper"] == data["triangularity_upper"]
    assert triangularity.upper == data["triangularity_upper"]


def test_triangularity_minor_attrs(data):
    triangularity = Points("triangularity", data, "minor")
    assert triangularity.attrs == [
        "triangularity_inner",
        "triangularity_outer",
    ]


def test_triangularity_minor_mean(data):
    triangularity = Points("triangularity", data, "minor")
    assert np.isclose(
        triangularity.mean,
        np.mean(
            [data[attr] for attr in ["triangularity_inner", "triangularity_outer"]]
        ),
    )


def test_triangularity_minor_mean_input_adjust(data):
    data["triangularity_inner"] = 0
    triangularity = Points("triangularity", data, "minor")
    assert np.isclose(triangularity.mean, 0.3)


def test_triangularity_outer(data):
    triangularity = Points("triangularity", data, "minor")
    assert triangularity["outer"] == data["triangularity_outer"]
    assert triangularity.outer == data["triangularity_outer"]


def test_triangularity_dict_update(data):
    triangularity = Points("triangularity", data, "major")
    data["triangularity_upper"] = -0.3
    assert np.isclose(triangularity.upper, -0.3)


def test_triangularity_mean_update_major(data):
    triangularity = Points("triangularity", data, "major")
    kwargs = {"deep": True} if isinstance(data, xarray.Dataset) else {}
    old_data = data.copy(**kwargs)
    value = 0.5
    factor = value / triangularity.mean
    triangularity.mean = value
    assert np.isclose(triangularity.upper, factor * old_data["triangularity_upper"])
    assert np.isclose(triangularity.lower, factor * old_data["triangularity_lower"])


def test_triangularity_attr_update(data):
    triangularity = Points("triangularity", data, "major")
    triangularity["upper"] = -0.3
    assert np.isclose(triangularity.upper, -0.3)
    assert np.isclose(data["triangularity_upper"], -0.3)


def test_triangularity_class_update(data):
    triangularity = Points("triangularity", data, "major")
    data["triangularity_upper"] = -0.3
    assert np.isclose(triangularity.upper, -0.3)


def test_triangularity_class_attr_update(data):
    triangularity = Points("triangularity", data, "major")
    triangularity["upper"] = -0.3
    assert np.isclose(data["triangularity_upper"], -0.3)


def test_plasmapoints_getitems(data):
    plasmapoints = PlasmaPoints(data)
    assert np.isclose(plasmapoints.elongation, data["elongation"])
    assert np.isclose(plasmapoints.triangularity.upper, data["triangularity_upper"])


def test_plasmapoints_adjust_major(data):
    plasmapoints = PlasmaPoints(data)
    kwargs = {"deep": True} if isinstance(data, xarray.Dataset) else {}
    old_data = data.copy(**kwargs)
    factor = 1.5
    plasmapoints.triangularity_major.mean *= factor
    assert np.isclose(
        plasmapoints.triangularity_major.upper,
        factor * old_data["triangularity_upper"],
    )
    assert np.isclose(
        plasmapoints.triangularity.upper,
        factor * old_data["triangularity_upper"],
    )
    assert np.isclose(
        data["triangularity_upper"],
        factor * old_data["triangularity_upper"],
    )


def test_squareness_missing_attr(data):
    squareness = Points("squareness", data, "square")
    assert squareness.upper_outer == data["squareness"]


def test_control_points_data(data):
    points = ControlPoints(data, square=True, strike=True)
    assert np.allclose(points.axis, data["geometric_axis"])
    for attr in ["elongation", "minor_radius"]:
        assert getattr(points, attr) == data[attr]
    assert np.isclose(
        points.triangularity_major.mean,
        np.mean([data[f"triangularity_{attr}"] for attr in ["upper", "lower"]]),
    )
    assert np.isclose(points.squareness.mean, data["squareness"])


def test_elongation():
    plasmapoints = PlasmaPoints({"elongation": 2.3})
    assert plasmapoints.elongation == 2.3


def test_upper_triangularity_lower():
    plasmapoints = PlasmaPoints({"triangularity_upper": 3, "triangularity_lower": 2})
    assert plasmapoints.triangularity.mean == 2.5


def test_triangularity_upper_attr():
    plasmapoints = PlasmaPoints({"triangularity_upper": 3, "triangularity_lower": 2})
    assert np.isclose(plasmapoints.triangularity.upper, 3)


def test_triangularity_upper_from_lower():
    plasmapoints = PlasmaPoints({"triangularity_lower": 2.5})
    assert np.isclose(plasmapoints.triangularity.upper, 2.5)


def test_triangularity_lower_attr():
    plasmapoints = PlasmaPoints({"triangularity_upper": 3, "triangularity_lower": 1.4})
    assert np.isclose(plasmapoints.triangularity.lower, 1.4)


def test_triangularity_lower_from_upper():
    plasmapoints = PlasmaPoints({"triangularity_upper": 2.4})
    assert np.isclose(plasmapoints.triangularity.lower, 2.4)


def test_triangularity_lower_from_upper_major():
    plasmapoints = PlasmaPoints(
        {"triangularity_upper": 2.4, "triangularity_outer": 0.1}
    )
    assert np.isclose(plasmapoints.triangularity_major.lower, 2.4)


def test_triangularity_over_constraint_dual():
    plasmapoints = PlasmaPoints(
        {
            "triangularity_dual": 2.4,
            "triangularity_upper": 3,
            "triangularity_lower": 1.4,
        }
    )
    assert np.isclose(plasmapoints.triangularity.upper, 3)
    assert np.isclose(plasmapoints.triangularity.lower, 1.4)
    assert np.isclose(plasmapoints.triangularity.mean, 2.4)


def test_triangularity_over_constraint_major():
    plasmapoints = PlasmaPoints(
        {"triangularity": 2.4, "triangularity_upper": 3, "triangularity_lower": 1.4}
    )
    assert np.isclose(plasmapoints.triangularity_major.upper, 3)
    assert np.isclose(plasmapoints.triangularity_major.lower, 1.4)
    assert np.isclose(plasmapoints.triangularity_major.mean, 2.2)


def test_triangularty_over_constraint():
    plasmapoints = PlasmaPoints(
        {"triangularity": 2.5, "triangularity_upper": 3, "triangularity_lower": 2}
    )
    assert np.isclose(plasmapoints.triangularity.upper, 3)
    assert np.isclose(plasmapoints.triangularity.lower, 2)
    assert np.isclose(plasmapoints.triangularity.mean, 2.5)


def test_triangularity_mean():
    plasmapoints = PlasmaPoints({"triangularity_upper": 3, "triangularity_dual": 2})
    assert np.isclose(plasmapoints.triangularity.mean, 2)


if __name__ == "__main__":
    pytest.main([__file__])
