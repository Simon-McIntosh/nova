from dataclasses import dataclass, field
import pytest

import numpy as np

from nova.geometry.plasmapoints import PlasmaPoints, Points


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


@pytest.fixture
def data_dict() -> dict:
    """Return test plasma geometry dataset."""
    return {
        "elongation": 2.1,
        "triangularity": 0.3,
        "triangularity_upper": 0.2,
        "triangularity_lower": 0.4,
        "triangularity_inner": 0.4,
        "triangularity_outer": 0.6,
        "squareness": 0.1,
    }


@pytest.fixture
def data_class():
    """Return data class."""
    return DataClass(
        {
            "elongation": 2.1,
            "triangularity": 0.3,
            "triangularity_upper": 0.2,
            "triangularity_lower": 0.4,
            "triangularity_inner": 0.4,
            "triangularity_outer": 0.6,
            "squareness": -0.1,
        }
    )


def test_elongation_name(data_dict):
    elongation = Points("elongation", data_dict)
    assert elongation.name == "elongation"


def test_elongation_attrs(data_dict):
    elongation = Points("elongation", data_dict)
    assert elongation.attrs == ["elongation"]


def test_elongation_mean_item(data_dict):
    elongation = Points("elongation", data_dict)
    assert elongation.mean == data_dict["elongation"]


def test_elongation_item_error(data_dict):
    elongation = Points("elongation", data_dict)
    with pytest.raises(KeyError):
        elongation["upper"]


def test_elongation_attr(data_dict):
    elongation = Points("elongation", data_dict)
    assert elongation.mean == data_dict["elongation"]


def test_elongation_attr_error(data_dict):
    elongation = Points("elongation", data_dict)
    with pytest.raises(AttributeError):
        elongation.upper


def test_triangularity_major_attrs(data_dict):
    triangularity = Points("triangularity", data_dict, "major")
    assert triangularity.attrs == [
        "triangularity_upper",
        "triangularity_lower",
    ]


def test_triangularity_major_mean(data_dict):
    triangularity = Points("triangularity", data_dict, "major")
    assert np.isclose(triangularity.mean, data_dict["triangularity"])


def test_triangularity_upper(data_dict):
    triangularity = Points("triangularity", data_dict, "major")
    assert triangularity["upper"] == data_dict["triangularity_upper"]
    assert triangularity.upper == data_dict["triangularity_upper"]


def test_triangularity_minor_attrs(data_dict):
    triangularity = Points("triangularity", data_dict, "minor")
    assert triangularity.attrs == [
        "triangularity_inner",
        "triangularity_outer",
    ]


def test_triangularity_minor_mean(data_dict):
    triangularity = Points("triangularity", data_dict, "minor")
    assert np.isclose(
        triangularity.mean,
        np.mean(
            [data_dict[attr] for attr in ["triangularity_inner", "triangularity_outer"]]
        ),
    )


def test_triangularity_minor_mean_input_adjust(data_dict):
    data_dict["triangularity_inner"] = 0
    triangularity = Points("triangularity", data_dict, "minor")
    assert np.isclose(triangularity.mean, 0.3)


def test_triangularity_outer(data_dict):
    triangularity = Points("triangularity", data_dict, "minor")
    assert triangularity["outer"] == data_dict["triangularity_outer"]
    assert triangularity.outer == data_dict["triangularity_outer"]


def test_triangularity_dict_update(data_dict):
    triangularity = Points("triangularity", data_dict, "major")
    data_dict["triangularity_upper"] = -0.3
    assert np.isclose(triangularity.upper, -0.3)


def test_triangularity_mean_update_major(data_dict):
    triangularity = Points("triangularity", data_dict, "major")
    old_data_dict = data_dict.copy()
    value = 0.5
    factor = value / triangularity.mean
    triangularity.mean = value
    assert np.isclose(
        triangularity.upper, factor * old_data_dict["triangularity_upper"]
    )
    assert np.isclose(
        triangularity.lower, factor * old_data_dict["triangularity_lower"]
    )


def test_triangularity_attr_update(data_dict):
    triangularity = Points("triangularity", data_dict, "major")
    triangularity["upper"] = -0.3
    assert np.isclose(triangularity.upper, -0.3)
    assert np.isclose(data_dict["triangularity_upper"], -0.3)


def test_triangularity_class_update(data_class):
    triangularity = Points("triangularity", data_class, "major")
    data_class["triangularity_upper"] = -0.3
    assert np.isclose(triangularity.upper, -0.3)


def test_triangularity_class_attr_update(data_class):
    triangularity = Points("triangularity", data_class, "major")
    triangularity["upper"] = -0.3
    assert np.isclose(data_class["triangularity_upper"], -0.3)


def test_plasmapoints_getitems(data_class):
    plasmapoints = PlasmaPoints(data_class)
    assert np.isclose(plasmapoints.elongation, data_class["elongation"])
    assert np.isclose(
        plasmapoints.triangularity.upper, data_class["triangularity_upper"]
    )


def test_plasmapoints_adjust_major(data_class):
    plasmapoints = PlasmaPoints(data_class)
    old_data = data_class.data.copy()
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
        data_class["triangularity_upper"],
        factor * old_data["triangularity_upper"],
    )


def test_squareness_missing_attr(data_dict):
    squareness = Points("squareness", data_dict, "square")
    assert squareness.upper_outer == data_dict["squareness"]


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


def test_triangularity_over_constraint_error():
    plasmapoints = PlasmaPoints(
        {"triangularity": 2.4, "triangularity_upper": 3, "triangularity_lower": 1.4}
    )
    assert np.isclose(plasmapoints.triangularity.upper, 3)
    assert np.isclose(plasmapoints.triangularity.lower, 1.4)
    assert np.isclose(plasmapoints.triangularity.mean, 2.2)


def test_triangularty_over_constraint():
    plasmapoints = PlasmaPoints(
        {"triangularity": 2.5, "triangularity_upper": 3, "triangularity_lower": 2}
    )
    assert np.isclose(plasmapoints.triangularity.upper, 3)
    assert np.isclose(plasmapoints.triangularity.lower, 2)
    assert np.isclose(plasmapoints.triangularity.mean, 2.5)


def test_triangularity_lower():
    plasmapoints = PlasmaPoints({"triangularity_upper": 3})
    assert np.isclose(plasmapoints.triangularity.lower, 3)


if __name__ == "__main__":
    pytest.main([__file__])
