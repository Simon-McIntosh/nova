"""The reader opens the wall at the Data Dictionary version of its own pulse.

The equilibrium reader reads the wall from the same pulse as the equilibrium,
so the wall must be opened at the version that pulse was written in. The URI
the reader carries names only the major (``version=3``), and a wall left to its
default resolves the latest published major, so a pulse at an earlier version
is refused as an unrequested major conversion. The reader therefore passes the
full pulse version beside the URI, and these tests fix that argument without
opening a database.

The reader itself needs a pulse to construct, which is why the guard is driven
through the wall property's own body with the two inputs it reads — the URI and
the pulse's version — supplied directly.
"""

from __future__ import annotations

import packaging.version
import pytest

import nova.imas.equilibrium as equilibrium_module
from nova.imas.equilibrium import EquilibriumData

#: a pulse written at a Data Dictionary version the published default is not.
PULSE_VERSION = "3.38.1"
URI = "imas:hdf5?path=/archive/iterdb/3/105052/1,equilibrium:0"


class Recorder:
    """Record the arguments the reader hands the wall constructor."""

    calls: list[dict] = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)


@pytest.fixture
def reader(monkeypatch):
    """Return a reader carrying one pulse's URI and written DD version.

    The wall property is invoked on an instance built without the reader's
    dataclass initialiser: that initialiser opens the pulse, and the pulse is
    what these tests exist to avoid needing.
    """
    Recorder.calls = []
    monkeypatch.setattr(equilibrium_module, "Wall", Recorder)
    monkeypatch.setattr(EquilibriumData, "uri", property(lambda self: URI))
    instance = object.__new__(EquilibriumData)
    instance.__dict__["ids_dd_version"] = packaging.version.Version(PULSE_VERSION)
    return instance


def test_the_wall_read_names_the_pulse_version(reader):
    """The wall constructor receives the pulse's version beside its URI."""
    EquilibriumData.wall.__get__(reader)
    assert len(Recorder.calls) == 1
    kwargs = Recorder.calls[0]
    assert kwargs["uri"] == URI
    assert kwargs["dd_version"] == PULSE_VERSION


def test_the_major_is_carried_with_the_pulse_version(reader):
    """The version passed is the pulse's full one, so its major travels with it.

    Dropping the major — opening the wall at the URI's own ``version=`` alone,
    or at a latest-published default — is what the pass-through prevents, and
    either leaves this argument absent or unrelated to the pulse.
    """
    EquilibriumData.wall.__get__(reader)
    kwargs = Recorder.calls[0]
    assert "dd_version" in kwargs, "the wall read names no version of its own"
    version = packaging.version.Version(kwargs["dd_version"])
    assert version == packaging.version.Version(PULSE_VERSION)
    assert str(version.major) == PULSE_VERSION.split(".")[0]
    assert kwargs["dd_version"] != str(version.major), (
        "the wall read passed a major alone, dropping the pulse's version"
    )
