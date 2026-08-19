from __future__ import annotations

import numpy as np
import pytest

from nova.imas import diiid_description as diiid


def _row() -> dict[str, object]:
    names = list(diiid.F_COILS) + ["ECOILA"]
    count = len(names)
    return {
        "coil_name": names,
        "coil_input_column": [f"magnetics_{name}" for name in names],
        "coil_R": np.linspace(0.8, 2.6, count),
        "coil_Z": np.linspace(-1.5, 1.5, count),
        "coil_width": np.full(count, 0.08),
        "coil_height": np.full(count, 0.12),
        "coil_angle1": [45.0 if name == "F5A" else 0.0 for name in names],
        "coil_angle2": [-92.4 if name == "F6B" else 0.0 for name in names],
        "efit_grid_R": np.asarray([1.0, 1.1]),
        "efit_grid_Z": np.asarray([-0.1, 0.1]),
        "efit_times": np.asarray([0.25, 0.75]),
        "magnetics_time": np.asarray([0.0, 0.5, 1.0]),
        **{
            f"magnetics_{name}": np.asarray([1.0, 2.0, 3.0])
            for name in names + ["bcoil"]
        },
    }


def test_registry_selection_has_complete_element_receipts() -> None:
    row = _row()
    registry = diiid.DiiidDescriptionRegistry()
    description = registry.ingest(row, source_row="d3d_shot_example.parquet")

    assert registry.select(row) is description
    assert description.provenance_complete
    assert {conductor.name for conductor in description.conductors} == set(
        diiid.ALL_CONDUCTORS
    )
    assert all(conductor.receipts for conductor in description.conductors)
    assert next(c for c in description.conductors if c.name == "bcoil").vertices is None


def test_skewed_sections_are_parallelograms_with_mirrored_shear() -> None:
    upper = diiid.section_vertices(1.0, 0.5, 0.2, 0.1, 0.0, 108.06)
    lower = diiid.section_vertices(1.0, -0.5, 0.2, 0.1, 0.0, -108.06)

    np.testing.assert_allclose(upper[1] - upper[0], upper[2] - upper[3])
    np.testing.assert_allclose(upper[3] - upper[0], upper[2] - upper[1])
    np.testing.assert_allclose(
        np.sort(upper[:, 0] - 1.0), np.sort(-(lower[:, 0] - 1.0))
    )


def test_every_poloidal_section_uses_polygon_kernel(monkeypatch) -> None:
    row = _row()
    description = diiid.DiiidDescriptionRegistry().ingest(
        row, source_row="d3d_shot_example.parquet"
    )
    calls: list[np.ndarray] = []

    def fake_greens(r, z, vertices):
        calls.append(np.asarray(vertices))
        value = np.full(np.asarray(r).shape, float(len(calls)))
        return value, value, value

    monkeypatch.setattr(diiid, "polygon_greens", fake_greens)
    response = diiid.vacuum_response(
        description, row["efit_grid_R"], row["efit_grid_Z"]
    )
    psi = diiid.vacuum_psi(row, description, response)

    assert len(calls) == 19
    np.testing.assert_allclose(response[1][0], 1.0)
    assert psi.shape == (2, 2, 2)
    assert np.isfinite(psi).all()


def test_registry_rejects_incomplete_geometry() -> None:
    row = _row()
    row["coil_name"] = list(row["coil_name"])[1:]
    with pytest.raises(diiid.DiiidDescriptionError, match="different lengths"):
        diiid.DiiidDescriptionRegistry().ingest(row, source_row="bad.parquet")


def test_starter_kit_bar_is_registered_exactly() -> None:
    assert diiid.STARTER_KIT_VACUUM_R2_BAR == 0.94
    assert "fusion-equilibrium-challenge-starter" in diiid.STARTER_KIT_VACUUM_BAR_SOURCE
