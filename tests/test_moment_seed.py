"""Flux-functions-only moment prediction and constrained seed contracts."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from benchmarks.efit_forward_parity_slice import GRID_STRIDE
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.moment import (
        CurrentCells,
        CurrentIntegralSupport,
        MomentConfig,
        MomentOrder,
        ReconstructMoment,
    )
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.imas.mast_solve_inputs import SHOT_STORE
    from nova.jax.config import configure_dtypes
    from scripts.moment_prediction_confidence.measure import _mast_frames


BANK = Path("scripts/moment_prediction_confidence/moment-prediction-confidence.tsv")


def _constant_profile() -> DomainProfile:
    """Return a smooth prescribed source with a non-zero current integral."""

    return DomainProfile(
        p_prime=lambda value: -jnp.ones_like(value),
        ff_prime=lambda value: jnp.zeros_like(value),
    )


def _rectangular_machine() -> tuple[ForwardProfile, np.ndarray]:
    """Return a minimal public forward surface for seed construction."""

    configure_dtypes()
    radius = np.linspace(0.5, 1.5, 31)
    height = np.linspace(-0.5, 0.5, 31)
    lattice = FluxLattice(radius, height)
    coordinate = lattice.coordinate
    operator = SimpleNamespace(
        grid=SimpleNamespace(coordinate=coordinate),
        wall=SimpleNamespace(
            coordinate=np.asarray(
                [[0.45, -0.55], [1.55, -0.55], [1.55, 0.55], [0.45, 0.55]]
            )
        ),
        inside_material=np.ones(lattice.node_count),
        source=ForwardSource(core=_constant_profile()),
    )
    operator.external = lambda _current=None: jnp.zeros(lattice.node_count)
    operator.coupling_current_moments = lambda moments: moments
    operator.current_moment_image = lambda moments: moments.cell_current
    profile = object.__new__(ForwardProfile)
    profile.operator = operator
    profile.lattice = lattice
    boundary = np.asarray([[0.62, -0.40], [1.42, -0.32], [1.38, 0.42], [0.58, 0.35]])
    return profile, boundary


def test_seed_integrates_to_its_declared_current_and_centroid() -> None:
    profile, boundary = _rectangular_machine()

    seed = profile.moment_seed(boundary, 120_000.0, radius_fraction=0.35)
    current = np.asarray(seed.cell_current)
    coordinate = np.asarray(profile.lattice.coordinate)
    integrated = float(np.sum(current))
    centroid = np.sum(coordinate * current[:, None], axis=0) / integrated

    assert integrated == pytest.approx(seed.moments.plasma_current, abs=1.0e-9)
    np.testing.assert_allclose(centroid, seed.moments.centroid, atol=1.0e-12)
    assert seed.support is CurrentIntegralSupport.COMPACT_CENTROID_DISC
    assert seed.moments.current_support is (
        CurrentIntegralSupport.BOUNDARY_HYPOTHESIS_ALL_DOMAIN
    )
    assert seed.moments.centroid_support is (
        CurrentIntegralSupport.BOUNDARY_HYPOTHESIS_ALL_DOMAIN
    )
    assert seed.supported_cells == np.count_nonzero(current)


def test_seed_calls_the_public_constrained_seam() -> None:
    profile, boundary = _rectangular_machine()
    seed = profile.moment_seed(boundary, 120_000.0, radius_fraction=0.35)
    captured = {}

    def solve(initial_flux, **options):
        captured["initial_flux"] = initial_flux
        captured.update(options)
        return "solved"

    result = seed.solve(SimpleNamespace(solve=solve), route="newton_krylov")

    assert result == "solved"
    np.testing.assert_array_equal(captured["initial_flux"], seed.flux)
    assert captured["target_current"] == seed.moments.plasma_current
    assert captured["route"] == "newton_krylov"


def test_profile_predictor_matches_the_banked_reference_rows() -> None:
    configure_dtypes()
    with BANK.open(newline="") as stream:
        expected = {
            row["identity"]: row
            for row in csv.DictReader(stream, delimiter="\t")
            if row["machine"] == "MAST" and float(row["boundary_scale"]) == 1.0
        }

    assert GRID_STRIDE == 2
    assert len(expected) == 6
    for frame in _mast_frames(SHOT_STORE):
        lattice = FluxLattice(frame.radius, frame.height)
        reconstruction = ReconstructMoment(
            CurrentCells(
                lattice.coordinate[:, 0],
                lattice.coordinate[:, 1],
                dr=lattice.radial_step,
                dz=lattice.vertical_step,
            ),
            config=MomentConfig(order=MomentOrder.CENTROID),
        )
        prediction = reconstruction.predict_profile_moments(
            frame.profile,
            frame.boundary,
            frame.target_current_a,
            cell_area=lattice.cell_area,
        )
        row = expected[frame.identity]
        assert prediction.plasma_current == pytest.approx(
            float(row["predicted_current_a"]), rel=2.0e-12
        )
        assert prediction.raw_source_current == pytest.approx(
            float(row["raw_predicted_source_current_a"]), rel=2.0e-12
        )
        assert prediction.centroid_r == pytest.approx(
            float(row["predicted_centroid_r_m"]), abs=2.0e-12
        )
        assert prediction.centroid_z == pytest.approx(
            float(row["predicted_centroid_z_m"]), abs=2.0e-12
        )
