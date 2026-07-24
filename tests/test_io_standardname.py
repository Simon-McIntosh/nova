"""Standard-name resolution against the managed ISN/ISNC vocabulary."""

import pytest

from nova.utilities.importmanager import mark_import

with mark_import("imas_standard_names") as mark_isn:
    import imas_standard_names  # noqa: F401

    from nova.io.standardname import (
        NameSource,
        StandardNameResolver,
        UnknownStandardName,
    )


@pytest.fixture(scope="module")
def resolver():
    """Return a resolver backed by the installed catalog/grammar."""
    return StandardNameResolver()


@mark_isn
def test_catalog_hit_carries_unit(resolver):
    resolution = resolver.resolve("plasma_current")
    assert resolution.source is NameSource.CATALOG
    assert not resolution.provisional
    assert resolution.unit is not None


@mark_isn
def test_catalog_hit_known(resolver):
    assert resolver.known("current_density")


@mark_isn
def test_grammar_valid_name_is_provisional(resolver):
    # A grammar-valid name absent from the bundled catalog resolves as a
    # provisional candidate rather than raising -- these are the catalog-fork
    # contribution candidates.
    resolution = resolver.resolve("radial_current_density")
    if resolution.source is NameSource.CATALOG:
        pytest.skip("name is present in the installed catalog")
    assert resolution.provisional
    assert resolution.status == "provisional"
    assert resolution.unit is None
    assert not resolver.known("radial_current_density")


@mark_isn
def test_unparseable_name_raises(resolver):
    with pytest.raises(UnknownStandardName):
        resolver.resolve("definitely_not_a_standard_name_zzz")
