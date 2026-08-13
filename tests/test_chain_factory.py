"""Production MAST reconstruction-chain factory contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from nova.equilibrium.moment import ReconstructMoment
from nova.equilibrium.profile import ReconstructProfile
from nova.imas.mast_chain_factory import (
    DEFAULT_RADIAL_POINTS,
    DEFAULT_VERTICAL_POINTS,
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
    assert components.topology_labeler.grid.rg.size == DEFAULT_RADIAL_POINTS
    assert components.topology_labeler.grid.zg.size == DEFAULT_VERTICAL_POINTS


@pytest.mark.parametrize(
    ("read", "reason"),
    [
        (SimpleNamespace(found=False), "no closed axis-connected surface"),
        (
            SimpleNamespace(found=True, is_diverted=True, boundary_resolved=True),
            "x-point saddle bound the separatrix",
        ),
        (
            SimpleNamespace(found=True, is_diverted=True, boundary_resolved=False),
            "x-point saddle remained unresolved",
        ),
        (
            SimpleNamespace(
                found=True,
                is_diverted=False,
                x_candidate_count=0,
            ),
            "wall tangency with no saddle candidate",
        ),
    ],
)
def test_boundary_termination_reason(read, reason):
    from benchmarks.topology_disagreement import boundary_termination_reason

    assert boundary_termination_reason(read) == reason
