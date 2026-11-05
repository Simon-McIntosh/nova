import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from nova.database.filepath import operand_cache_root


MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "diiid_forward_gs_match.py"
SPEC = importlib.util.spec_from_file_location(
    "diiid_forward_gs_match_operands", MODULE_PATH
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _synthetic_pair(root: Path, rows: int = 12, columns: int = 4):
    """Write one frame artifact beside a response matrix, in the frame layout."""

    rng = np.random.default_rng(20261105)
    response = rng.normal(size=(rows, columns))
    current = rng.normal(size=columns)
    radius = np.linspace(1.0, 2.0, 4)
    height = np.linspace(-0.5, 0.5, 3)
    points = np.stack(np.meshgrid(radius, height, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    index = np.arange(points.shape[0], dtype=np.intp)
    conductor = response[index] @ current
    plasma = rng.normal(size=points.shape[0])
    artifact = root / "frame-display-synthetic.npz"
    np.savez_compressed(
        artifact,
        state=rng.normal(size=6),
        display_points=points,
        display_radius=radius,
        display_height=height,
        flux=(conductor + plasma).reshape(radius.size, height.size),
        conductor_flux=conductor,
        plasma_flux=plasma,
        conductor_current=current,
        response_row_index=index,
    )
    matrix = root / "display-conductor-response-synthetic.npy"
    np.save(matrix, response)
    return artifact, matrix


def test_rebuild_reproduces_the_persisted_field_by_the_matrix_product(tmp_path):
    """A persisted frame's field is its conductor product plus its plasma term."""

    artifact, matrix = _synthetic_pair(tmp_path)
    operands = gate.load_frame_operands(artifact)
    field, conductor = gate.rebuild_display_flux(operands, np.load(matrix))
    reference = operands["flux"].ravel()
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(reference))))
    assert np.allclose(conductor, operands["conductor_flux"], rtol=0.0, atol=tolerance)
    assert np.allclose(field, reference, rtol=0.0, atol=tolerance)


def test_persisted_bank_frame_rebuilds_to_one_em12():
    """The newest persisted bank frame rebuilds from its own matrix to 1e-12."""

    root = operand_cache_root() / "diiid-forward-gs"
    artifacts = sorted(
        root.glob("frame-display-*.npz"), key=lambda path: path.stat().st_mtime
    )
    if not artifacts:
        pytest.skip(
            "no persisted frame artifact yet: the bank predates the persistence"
        )
    artifact = artifacts[-1]
    operands = gate.load_frame_operands(artifact)
    candidates = sorted(
        root.glob("display-conductor-response-*.npy"),
        key=lambda path: path.stat().st_mtime,
    )
    target = int(operands["response_row_index"].ravel()[-1]) + 1
    matrix = None
    for candidate in candidates:
        loaded = np.load(candidate)
        if (
            loaded.shape[1] == operands["conductor_current"].size
            and loaded.shape[0] >= target
        ):
            matrix = loaded
            break
    if matrix is None:
        pytest.skip("no response matrix covers this frame's display points")
    field, conductor = gate.rebuild_display_flux(operands, matrix)
    reference = operands["flux"].ravel()
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(reference))))
    assert np.allclose(conductor, operands["conductor_flux"], rtol=0.0, atol=tolerance)
    assert np.allclose(field, reference, rtol=0.0, atol=tolerance)
