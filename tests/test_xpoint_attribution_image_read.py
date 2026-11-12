"""The saved-field attribution must locate the physical image's own saddle."""

import json

import numpy as np
import pytest

from scripts.prototypes.diiid_xpoint_attribution.attribute import split_operator


def bank(tmp_path):
    radius = np.linspace(1.0, 2.0, 21)
    height = np.linspace(-0.6, 0.4, 21)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    reference = (rr - 1.5) ** 2 - 2 * (zz + 0.1) ** 2
    physical = (rr - 1.52) ** 2 - 2 * (zz + 0.07) ** 2
    shadow = (abs(rr - 1.5) < 0.13) & (abs(zz + 0.1) < 0.13)
    held = np.where(shadow, 0.015 * (zz + 0.1), 0)
    solved = physical + held
    native = tmp_path / "native"
    saved = tmp_path / "saved"
    native.mkdir()
    saved.mkdir()
    np.savez(saved / "terminal.npz", shadow=shadow.ravel(), state=solved.ravel())
    np.savez(saved / "request-seed.npz", state=solved.ravel())
    np.savez(
        native / "fields-137.npz",
        radius=radius,
        height=height,
        reference=reference,
        solved=solved,
        vacuum=np.zeros_like(reference),
        reference_plasma=reference,
    )
    record = dict(
        folder=str(saved),
        current=dict(prescribed_a=1.0, reference_tare_a=1.0),
        solved_saddle_m=[1.52, -0.07],
        reference_saddle_m=[1.5, -0.1],
        attribution={},
        measured_distance_cm=9.0,
    )
    (native / "result-137.json").write_text(json.dumps(record))
    for name, field in (("reference", reference), ("solved", physical)):
        np.savez(
            tmp_path / f"operator-137-{name}.npz",
            external=np.zeros(rr.size),
            source_image=field.ravel(),
            source_current=np.ones(rr.size),
        )
    return physical, held, shadow


def test_split_operator_reads_image_saddle_and_retains_shadow_difference(tmp_path):
    physical, held, shadow = bank(tmp_path)
    result = split_operator(tmp_path, 137)
    np.testing.assert_allclose(
        result["physical_image_saddle_m"], [1.52, -0.07], atol=1e-10
    )
    np.testing.assert_allclose(
        result["physical_image_offset_cm"], [2.0, 3.0], atol=1e-8
    )
    assert result["physical_image_distance_cm"] == pytest.approx(np.sqrt(13), abs=1e-8)
    assert result["shadow_vs_request_seed_sup_wb"] == 0
    assert result["unshadowed_closure_sup_wb"] < 1e-14
    assert result["shadow_sup_wb"] > 0
    with np.load(tmp_path / "native/fields-137.npz") as fields:
        np.testing.assert_allclose(fields["physical_image"], physical)
        np.testing.assert_allclose(fields["boundary_shadow"], held, atol=1e-14)
        np.testing.assert_array_equal(fields["shadow"], shadow)


def test_split_operator_refuses_unshadowed_reconstruction_error(tmp_path):
    bank(tmp_path)
    path = tmp_path / "operator-137-solved.npz"
    data = dict(np.load(path))
    data["source_image"] = data["source_image"] + 0.01
    np.savez(path, **data)
    with pytest.raises(ValueError, match="physical operator does not close"):
        split_operator(tmp_path, 137)
