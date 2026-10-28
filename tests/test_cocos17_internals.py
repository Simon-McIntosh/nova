"""Contracts for nova's internal consumers running in COCOS 17.

Each test fixes a known COCOS 17 state and reads the consumer's sign-sensitive
output back, so a site that re-derives the convention instead of reading it
from :mod:`nova.equilibrium.convention` or from the cache-key owner
:meth:`nova.io.signalstore.SignalStore.cache_key` fails on its own assertion.
"""

from __future__ import annotations

from math import tau

import numpy as np
import pytest

from nova.biot import contour as contour_module
from nova.biot.contour import Surface
from nova.equilibrium.convention import (
    COCOS,
    TOTAL_FLUX_FACTOR,
    toroidal_current_density,
)
from nova.io.signalstore import StaleConventionError

from tests.test_signalstore_convention_key import (
    DD3,
    DD4,
    build_equilibrium,
    make_store,
    tensorize,
)


def surface(points: int = 40) -> Surface:
    """Return a closed contour surface on a small flux ring."""
    theta = np.linspace(0.0, tau, points)
    radius = 6.2 + 0.5 * np.cos(theta)
    height = 0.4 * np.sin(theta)
    return Surface(np.c_[radius, height], np.full(points, 79), 1.0)


def test_the_contour_fit_recovers_the_flux_functions_through_the_owner():
    """A fitted surface's coefficients reproduce the current density it was fit to."""
    surface_ = surface()
    p_prime = -3.0e3
    ff_prime = 5.0
    surface_.fit(
        lambda radius, height: toroidal_current_density(radius, p_prime, ff_prime)
    )

    assert np.allclose(surface_.coefficients, [p_prime, ff_prime], rtol=1e-9)
    reconstructed = toroidal_current_density(
        surface_.radius, surface_.coefficients[0], surface_.coefficients[1]
    )
    assert np.allclose(reconstructed, surface_.variable, rtol=1e-9)


def test_the_contour_fit_takes_its_factor_from_the_owner(monkeypatch):
    """Changing the owner's factor moves the fitted coefficients by the same factor."""
    reference = surface()
    reference.fit(lambda radius, height: toroidal_current_density(radius, 2.0e3, -7.0))

    monkeypatch.setattr(contour_module, "TOTAL_FLUX_FACTOR", 1.0)
    rescaled = surface()
    rescaled.fit(lambda radius, height: toroidal_current_density(radius, 2.0e3, -7.0))

    assert TOTAL_FLUX_FACTOR == tau
    assert np.allclose(rescaled.coefficients, tau * reference.coefficients, rtol=1e-9)


def test_the_cache_key_carries_the_convention_through_the_key_owner(tmp_path):
    """The store's key names the convention as well as the source identity."""
    store = make_store(tmp_path, DD3)
    assert store.cocos == 11
    assert store.cache_key == store.hash_attrs(
        {"uri": store.uri, "dd_version": DD3, "cocos": store.cocos}
    )
    assert store.cache_key != store.legacy_cache_key


def test_a_second_convention_of_one_source_gets_its_own_group(tmp_path):
    """Two conventions of one source never share a group."""
    dd3 = make_store(tmp_path, DD3)
    rekeyed = make_store(tmp_path, DD3, cocos=17)
    dd4 = make_store(tmp_path, DD4)

    assert dd3.cache_key != rekeyed.cache_key
    assert dd3.cache_key != dd4.cache_key
    assert dd4.cocos == COCOS


def test_an_entry_written_before_the_change_is_refused_not_served(tmp_path):
    """A group keyed without the convention is refused rather than served."""
    dataset = tensorize(tmp_path, DD3)
    stale = make_store(tmp_path, DD3)
    stale.group = stale.legacy_cache_key
    stale.write(dataset)

    fresh = make_store(tmp_path, DD3)
    with pytest.raises(StaleConventionError):
        fresh.read()
    assert not fresh.cached()


def test_the_inverse_boundary_constraint_carries_the_readers_flux(monkeypatch):
    """The inverse constraint takes psi_boundary as the reader delivered it."""
    from nova.equilibrium import inverse as module

    recorded: dict[str, float] = {}
    host = next(
        klass
        for klass in module.InverseDesign.__mro__[1:]
        if "update_constraints" in klass.__dict__
    )

    def record(_self, constraint, *args, **kwargs):
        recorded["constraint"] = constraint

    monkeypatch.setattr(host, "update_constraints", record)
    monkeypatch.setattr(module.InverseDesign, "__getitem__", lambda _self, key: -123.5)
    instance = object.__new__(module.InverseDesign)
    module.InverseDesign.update_constraints(instance)

    assert recorded["constraint"] == -123.5


def test_the_synthetic_equilibrium_builder_round_trips():
    """Guard the fixture itself: a builder that returns nothing would pass silently."""
    ids = build_equilibrium(DD3)
    assert len(ids.time) > 0
