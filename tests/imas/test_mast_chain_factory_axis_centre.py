"""Boundary rings retain the centre used by the connectivity ray cast."""

from types import SimpleNamespace

import numpy as np

from nova.equilibrium.labels import LCFS_ANGLES
from nova.imas import mast_chain_factory


def test_reported_ring_uses_the_connectivity_ray_centre(monkeypatch):
    """A refined read cannot relocate radii cast about the extracted centre."""

    initial_axis = np.array([0.8, 0.1])
    refined_axis = np.array([0.81, 0.11])
    calls = []
    radial_samples = []

    def boundary_read(_flux, _grid, axis, **options):
        calls.append(np.asarray(axis, dtype=float))
        radial_samples.append(options["n_ray"])
        if len(calls) == 1:
            return SimpleNamespace(axis=initial_axis)
        return SimpleNamespace(
            axis=refined_axis,
            radii=np.ones(len(LCFS_ANGLES)),
            psi_bnd=1.0,
            psi_axis=0.0,
            xset=np.full((2, 2), np.nan),
            is_diverted=False,
        )

    monkeypatch.setattr(mast_chain_factory, "host_boundary_read", boundary_read)
    grid = SimpleNamespace(
        rg=np.array([0.5, 1.0]),
        zg=np.array([-0.5, 0.5]),
        inside_limiter=np.ones((2, 2), dtype=bool),
    )
    labeler = mast_chain_factory.MastTopologyLabeler(grid, (0.75, 0.0))

    topology = labeler(np.arange(4.0)[None])

    assert len(calls) == 2
    np.testing.assert_allclose(calls[1], initial_axis)
    np.testing.assert_allclose(np.mean(topology.lcfs_m[0], axis=0), calls[1])
    np.testing.assert_allclose(topology.magnetic_axis_m[0], refined_axis)
    assert radial_samples == [mast_chain_factory.BOUNDARY_RADIAL_SAMPLES] * 2
    assert mast_chain_factory.BOUNDARY_RADIAL_SAMPLES > len(LCFS_ANGLES)
