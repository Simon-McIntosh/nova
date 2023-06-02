import pytest

from nova.frame.select import Label


def test_default():
    label = Label(None, None, None)
    assert label.include == []
    assert label.exclude == []


def test_preclude():
    label = Label(None, None, "plasma")
    assert label.exclude == ["plasma"]


def test_input_exclude_unique():
    label = Label("active", ["active", "passive"], None)
    assert label.include == ["active"]
    assert label.exclude == ["passive"]


def test_plasma():
    label = Label("plasma", None, ["feedback", "plasma"])
    assert label.include == ["plasma"]
    assert label.exclude == ["feedback"]


def test_required():
    label = Label(
        "active", ["active", "passive", "feedback"], ["plasma", "feedback", "control"]
    )
    assert label.require == ["active", "passive", "feedback", "plasma", "control"]


def test_to_dict():
    label = Label("active", ["active", "passive"], None)
    assert label.to_dict() == {"include": ["active"], "exclude": ["passive"]}


if __name__ == "__main__":
    pytest.main([__file__])
