"""Closed-form topology reads across the analytic refinement ladder."""

from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nova.equilibrium import topology  # noqa: E402
from nova.equilibrium.solve_request import TopologyPolicy  # noqa: E402
from tests.equilibrium import test_topology_read as fixture  # noqa: E402


GOLDENS = (
    Path(__file__).resolve().parents[1] / "scripts/prototypes/ladder/compatibility"
)
RADIUS = {"limited": 6.2, "diverted": 1.7}
COEFFICIENT = {"limited": 0.274714, "diverted": 2.1805365680219544}


@pytest.mark.slow
@pytest.mark.parametrize("kind", ("limited", "diverted"))
@pytest.mark.parametrize("cells", (550, 2000, 5000, 10000))
def test_closed_form_read_is_accurate_and_compatible(kind, cells):
    assert jax.config.jax_enable_x64 is True
    backend = jax.default_backend()
    if backend == "cpu" and cells > 550:
        pytest.skip("larger analytic carriers use the device lane")
    oracle, field, wall, axis, saddle = fixture._analytic_inputs(kind)
    geometry = fixture._realised_hex_geometry(wall, cells)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0,
        normal_form_pitch_floor=1.5,
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    result = jax.jit(topology.read)(field, geometry, convention, policy)
    jax.block_until_ready(result)
    assert bool(result.valid), f"reason={int(result.reason)}"
    assert bool(result.qualified)
    assert np.linalg.norm(np.asarray(result.axis) - axis) <= pitch
    x_points = np.asarray(result.x_points)[np.asarray(result.x_point_valid)]
    if saddle is None:
        assert len(x_points) == 0
    else:
        assert len(x_points) == 1
        assert np.linalg.norm(x_points[0] - saddle) <= pitch

    reference, uncertainty = fixture._reference_cell_fractions(kind, oracle, geometry)
    area = np.asarray(geometry.full_area)
    error = np.abs(np.asarray(result.membership) - reference) * area / np.median(area)
    bound = COEFFICIENT[kind] * (pitch / RADIUS[kind]) ** 2
    assert np.max(error) <= bound + np.max(uncertainty)

    if cells < 10000:
        suffix = "-cpu" if backend == "cpu" else ""
        with np.load(GOLDENS / f"{kind}-{cells}{suffix}.npz") as saved:
            assert bool(saved["valid"])
            np.testing.assert_allclose(
                result.membership, saved["membership"], rtol=0, atol=1e-13
            )
            np.testing.assert_allclose(result.axis, saved["axis"], rtol=0, atol=1e-13)
            np.testing.assert_array_equal(result.x_point_valid, saved["x_point_valid"])
            np.testing.assert_allclose(
                result.x_points, saved["x_points"], rtol=0, atol=1e-13
            )
            np.testing.assert_allclose(
                result.boundary_flux, saved["boundary_flux"], rtol=0, atol=1e-13
            )
