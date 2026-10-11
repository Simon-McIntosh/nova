"""Contracts for the device-batched hex-carrier coupling assembly."""

from contextlib import contextmanager
import os

import numpy as np
import pytest

from nova.biot.polygonanalytic import polygon_analytic_flux_moments_batched
from nova.database.zarrstore import ZarrStore
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import coupling_build, measure

# The uniform flux row is the packed kernel's primary channel and reproduces the
# host reduction to round-off.  The two moment companions accumulate a difference
# of two large partial reductions before the displacement correction, so the same
# expression graph lands a few hundred times further from the host value; the
# bound below is the measured round-off of that channel, not a modelling slack.
UNIFORM_BOUND = 1.0e-10
MOMENT_BOUND = 1.0e-7


def _hexagon(centre_r: float, centre_z: float, radius: float) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    return np.c_[centre_r + radius * np.cos(angle), centre_z + radius * np.sin(angle)]


def _case(count: int = 9):
    configure_dtypes()
    polygons = tuple(
        _hexagon(1.0 + 0.03 * index, 0.02 * (index % 3) - 0.02, 0.011)
        for index in range(count)
    )
    centres = np.asarray(
        [
            [float(np.mean(polygon[:, 0])), float(np.mean(polygon[:, 1]))]
            for polygon in polygons
        ]
    )
    targets = np.c_[
        1.0 + 0.4 * np.linspace(-1.0, 1.0, count), 0.3 * np.linspace(-1.0, 1.0, count)
    ]
    return targets, polygons, centres


def _identity_gap(blocks, reference) -> list[float]:
    """Return the per-row difference relative to each row's peak magnitude."""
    gap = []
    for row in range(3):
        peak = float(np.max(np.abs(reference[row])))
        difference = float(np.max(np.abs(blocks[row] - reference[row])))
        gap.append(difference / peak if peak else float("inf"))
    return gap


def test_device_blocks_match_host_builder_at_small_count():
    targets, polygons, centres = _case()
    reference = polygon_analytic_flux_moments_batched(
        targets[:, 0], targets[:, 1], polygons, expansion_points=centres, workers=1
    )
    blocks = coupling_build.build_flux_blocks(targets, polygons, centres)
    uniform, radial, vertical = _identity_gap(blocks, reference)
    assert uniform <= UNIFORM_BOUND
    assert radial <= MOMENT_BOUND
    assert vertical <= MOMENT_BOUND


def test_identity_check_catches_perturbed_block_by_test_id():
    targets, polygons, centres = _case()
    reference = polygon_analytic_flux_moments_batched(
        targets[:, 0], targets[:, 1], polygons, expansion_points=centres, workers=1
    )
    blocks = coupling_build.build_flux_blocks(targets, polygons, centres)
    perturbed = tuple(np.array(block, copy=True) for block in blocks)
    perturbed[0][0, 0] *= 1.0 + 1.0e-9
    with pytest.raises(AssertionError):
        _assert_identity(perturbed, reference)


def _assert_identity(blocks, reference):
    """Fail with the offending rows when the blocks leave the accepted round-off."""
    uniform, radial, vertical = _identity_gap(blocks, reference)
    assert uniform <= UNIFORM_BOUND, f"uniform row gap {uniform:.3e}"
    assert radial <= MOMENT_BOUND, f"radial moment row gap {radial:.3e}"
    assert vertical <= MOMENT_BOUND, f"vertical moment row gap {vertical:.3e}"


@contextmanager
def _builder_environment(value: str | None):
    """Set or clear the assembly-path variable for the duration of a block."""
    previous = os.environ.get(measure.BUILDER_ENVIRONMENT)
    if value is None:
        os.environ.pop(measure.BUILDER_ENVIRONMENT, None)
    else:
        os.environ[measure.BUILDER_ENVIRONMENT] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(measure.BUILDER_ENVIRONMENT, None)
        else:
            os.environ[measure.BUILDER_ENVIRONMENT] = previous


def _identity_group(store: ZarrStore, builder: str | None = None) -> tuple[dict, str]:
    """Return the semantic identity and its cache group for one assembly path."""
    identity = measure.cache_identity(
        measure.analytic_case(),
        requested_cells=-9,
        wall_nodes=measure.WALL_POINT_COUNT,
        builder=builder,
    )
    return identity, str(store.hash_attrs(identity))


def test_the_default_assembly_path_is_the_host_builder():
    """No existing caller changes behaviour: the host builder stays the default."""
    with _builder_environment(None):
        assert measure.resolve_builder() == measure.BUILDER_HOST
        identity = measure.cache_identity(
            measure.analytic_case(),
            requested_cells=-9,
            wall_nodes=measure.WALL_POINT_COUNT,
        )
        assert identity["builder"] == measure.BUILDER_HOST
    assert measure.resolve_builder(None) == measure.BUILDER_HOST
    assert measure.resolve_builder(measure.BUILDER_HOST) == measure.BUILDER_HOST


def test_the_environment_selects_the_device_path_for_a_scale_rung():
    """A scale rung opts in without editing the fixture."""
    with _builder_environment(measure.BUILDER_DEVICE):
        assert measure.resolve_builder() == measure.BUILDER_DEVICE
        identity = measure.cache_identity(
            measure.analytic_case(),
            requested_cells=-9,
            wall_nodes=measure.WALL_POINT_COUNT,
        )
        assert identity["builder"] == measure.BUILDER_DEVICE
    with pytest.raises(ValueError):
        measure.resolve_builder("reflected")


def test_the_device_path_publishes_a_distinct_cache_key(tmp_path):
    """A device-built carrier can never be read where a host-built one is expected."""
    store = ZarrStore(filename=measure.CACHE_FILENAME, dirname=str(tmp_path))
    host_identity, host_group = _identity_group(store, measure.BUILDER_HOST)
    device_identity, device_group = _identity_group(store, measure.BUILDER_DEVICE)
    assert device_identity["builder"] == measure.BUILDER_DEVICE
    assert host_group != device_group
    assert device_identity["builder_revision"] == host_identity["builder_revision"], (
        "the builder path alone must separate the two groups"
    )
    assert device_identity["discretisation"] == host_identity["discretisation"]


def test_naming_no_assembly_path_reaches_the_host_builder():
    """A caller that names no path keeps working and gets the host builder."""
    targets, polygons, centres = _case()
    with _builder_environment(None):
        blocks = measure._flux_blocks(targets, polygons, centres)
    analytic = polygon_analytic_flux_moments_batched(
        targets[:, 0], targets[:, 1], polygons, expansion_points=centres, workers=1
    )
    _assert_identity(blocks, analytic)


def test_each_assembly_path_reaches_its_own_builder():
    """The device path is the traced assembly; the host path is the analytic one."""
    targets, polygons, centres = _case()
    device = measure._flux_blocks(
        targets,
        polygons,
        centres,
        builder=measure.BUILDER_DEVICE,
    )
    reference = coupling_build.build_flux_blocks(targets, polygons, centres)
    for row in range(3):
        np.testing.assert_array_equal(device[row], reference[row])
    host = measure._flux_blocks(
        targets,
        polygons,
        centres,
        builder=measure.BUILDER_HOST,
    )
    analytic = polygon_analytic_flux_moments_batched(
        targets[:, 0], targets[:, 1], polygons, expansion_points=centres, workers=1
    )
    _assert_identity(host, analytic)
