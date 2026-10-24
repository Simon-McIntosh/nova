"""Contracts for the ids metadata a write sets.

The provenance structure sits on ``ids_properties`` at both Data Dictionary
majors, but the node it holds is laid out differently: a DD3 node carries a
list of sources, a DD4 node carries a structured reference and no source field.
A write therefore has to ask the node what it carries instead of assuming one
layout.
"""

from __future__ import annotations

import pytest

from nova.imas.metadata import Metadata

imas = pytest.importorskip("imas")

PROVENANCE = ["130506,6,equilibrium,0,iter,public,hdf5"]

#: the IDS names the review found sharing the DD4 layout.
IDS_NAMES = ["equilibrium", "pf_active", "pulse_schedule", "wall"]


def ids_at(dd_version: str, name: str = "equilibrium"):
    """Return a new, empty ids at the version named."""
    return imas.IDSFactory(dd_version).new(name)


def test_a_dd3_write_records_the_provenance_sources():
    """DD3 carries a source list, so the list is written there."""
    ids = ids_at("3.39.0")
    Metadata(ids).put_properties("test", homogeneous_time=1, provenance=PROVENANCE)
    node = ids.ids_properties.provenance.node[0]
    assert list(node.sources) == PROVENANCE


@pytest.mark.parametrize("dd_version", ["4.0.0", "4.1.1"])
@pytest.mark.parametrize("name", IDS_NAMES)
def test_a_dd4_write_with_provenance_does_not_raise(dd_version, name):
    """A DD4 node carries no source field, so the list is left unwritten.

    The write must still succeed: the metadata a DD4 entry can hold is what
    gets written, and a field the version does not define is skipped rather
    than refused one line earlier on the node itself.
    """
    ids = ids_at(dd_version, name)
    Metadata(ids).put_properties("test", homogeneous_time=1, provenance=PROVENANCE)
    node = ids.ids_properties.provenance.node[0]
    assert not hasattr(node, "sources"), "a DD4 provenance node carries no sources"
    assert hasattr(node, "reference"), "a DD4 provenance node carries a reference"


def test_a_write_without_provenance_leaves_the_node_alone():
    """An empty source list is not a source list of length zero."""
    ids = ids_at("4.1.1")
    Metadata(ids).put_properties("test", homogeneous_time=1)
    assert len(ids.ids_properties.provenance.node) == 0
