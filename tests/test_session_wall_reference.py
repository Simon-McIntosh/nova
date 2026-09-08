"""Machine-description wall-reference contracts for steering sessions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nova.equilibrium.steering_frames import WallReference, dereference_wall
from nova.equilibrium.wall_mask import pack_wall_units, wall_units_from_ids
from nova.imas.mast_solve_input_ids import open_description


ARTIFACT_CACHE = Path("/run/user/39486/imas-ambix-machine-artifact")


def _mast_artifact() -> tuple[Path, str]:
    """Return one locally mounted MAST machine artifact, if present."""

    objects = ARTIFACT_CACHE / "sha256"
    if not objects.is_dir():
        pytest.skip("content-addressed MAST artifact is not mounted")
    for directory in sorted(objects.glob("[0-9a-f]" * 64)):
        manifest = directory / "manifest.json"
        if '"schema":"nova-mast-machine-artifact"' in manifest.read_text():
            return directory, f"sha256:{directory.name}"
    pytest.skip("no MAST machine artifact is mounted")


def test_mast_artifact_reference_dereferences_operator_wall_to_roundoff() -> None:
    """A persisted MAST artifact reproduces the operator wall from its reference."""

    directory, digest = _mast_artifact()
    opened = open_description(ARTIFACT_CACHE, digest)
    units = wall_units_from_ids(opened.ids["wall"])
    reference = WallReference.from_units(
        machine="MAST",
        source=str(directory),
        source_kind="artifact",
        dd_version=opened.dd_version,
        units=units,
    )

    dereferenced = dereference_wall(reference)
    expected, expected_offsets = pack_wall_units(units)
    actual, actual_offsets = pack_wall_units(dereferenced)

    np.testing.assert_array_equal(actual_offsets, expected_offsets)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)
