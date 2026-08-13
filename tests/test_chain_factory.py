"""Production MAST reconstruction-chain factory contracts."""

from pathlib import Path

import pytest

from nova.equilibrium.moment import ReconstructMoment
from nova.equilibrium.profile import ReconstructProfile
from nova.imas.mast_chain_factory import (
    MastCurrentDiffusion,
    MastTopologyLabeler,
    build_mast_parity_chain,
)
from nova.transport.current_diffusion import CurrentDiffusion


SHOT = 21978
SHOT_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")
ARTIFACT_CACHE = Path("/run/user/39486/imas-ambix-machine-artifact")


def _artifact_digest() -> str:
    objects = sorted((ARTIFACT_CACHE / "sha256").glob("[0-9a-f]" * 64))
    if not objects:
        pytest.skip("content-addressed MAST artifact is not mounted")
    return f"sha256:{objects[0].name}"


def test_factory_returns_production_solver_types():
    """The factory cannot regress to the chain's compact test doubles."""

    if not (SHOT_STORE / f"{SHOT}.zarr").exists():
        pytest.skip("MAST corrected-read store is not mounted")
    components = build_mast_parity_chain(
        SHOT,
        artifact_cache=ARTIFACT_CACHE,
        artifact_digest=_artifact_digest(),
        store=SHOT_STORE,
    )

    assert isinstance(components.moment_solver, ReconstructMoment)
    assert isinstance(components.profile_solver, ReconstructProfile)
    assert isinstance(components.topology_labeler, MastTopologyLabeler)
    assert isinstance(components.temporal_scorer, MastCurrentDiffusion)
    assert isinstance(components.temporal_scorer, CurrentDiffusion)
