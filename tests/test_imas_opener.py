"""The IMAS data entry opener keeps the dictionary version and the backend policy."""

from types import SimpleNamespace

from packaging.version import Version

import nova.imas.dataset as dataset_module


class Recorder:
    """Stand in for imas.DBEntry and record how it was called."""

    def __init__(self, record):
        self.record = record

    def __call__(self, uri, *args, **kwargs):
        self.record["uri"] = uri
        self.record["args"] = args
        self.record["kwargs"] = kwargs
        return SimpleNamespace(get=lambda *a, **k: None, close=lambda: None)


def opener():
    """Return the module-level opener, failing clearly where it is absent."""
    return dataset_module.open_db_entry


def test_opener_is_read_only_by_default(monkeypatch):
    record = {}
    monkeypatch.setattr(dataset_module.imas, "DBEntry", Recorder(record))

    opener()("imas:hdf5?path=/tmp/entry")

    assert record["kwargs"]["mode"] == "r"


def test_opener_passes_dd_version_through(monkeypatch):
    record = {}
    monkeypatch.setattr(dataset_module.imas, "DBEntry", Recorder(record))

    opener()("imas:hdf5?path=/tmp/entry", "r", dd_version=Version("3.39.0"))

    assert record["kwargs"]["dd_version"] == "3.39.0"


def test_opener_adds_no_dd_version_it_was_not_given(monkeypatch):
    record = {}
    monkeypatch.setattr(dataset_module.imas, "DBEntry", Recorder(record))

    opener()("imas:hdf5?path=/tmp/entry", "r")

    assert "dd_version" not in record["kwargs"]


def test_opener_routes_a_named_backend_into_the_uri(monkeypatch):
    record = {}
    monkeypatch.setattr(dataset_module.imas, "DBEntry", Recorder(record))

    opener()("/tmp/pulse.nc", "r", backend="hdf5")

    assert record["uri"] == "imas:hdf5?path=/tmp/pulse.nc"


def test_opener_leaves_a_uri_that_names_its_backend(monkeypatch):
    record = {}
    monkeypatch.setattr(dataset_module.imas, "DBEntry", Recorder(record))

    opener()("imas:hdf5?path=/tmp/entry", "r", backend="hdf5")

    assert record["uri"] == "imas:hdf5?path=/tmp/entry"


def test_is_valid_opens_through_the_opener(monkeypatch):
    record = {}
    monkeypatch.setattr(dataset_module, "open_db_entry", Recorder(record))
    dataset = SimpleNamespace(
        uri="imas:hdf5?path=/tmp/entry", name="equilibrium", occurrence=0
    )

    assert dataset_module.Dataset.is_valid.fget(dataset) is True
    assert record["uri"] == "imas:hdf5?path=/tmp/entry"
    assert record["args"][0] == "r"
    assert len(record["args"]) == 1
    assert record["kwargs"] == {}


def test_db_entry_property_passes_the_dataset_dd_version(monkeypatch):
    record = {}
    monkeypatch.setattr(dataset_module, "open_db_entry", Recorder(record))
    dataset = SimpleNamespace(
        uri="imas:hdf5?path=/tmp/entry",
        mode="r",
        dd_version=Version("3.39.0"),
        backend="hdf5",
    )

    dataset_module.Dataset.db_entry.func(dataset)

    assert record["args"][1] == Version("3.39.0")
    assert record["args"][0] == "r"