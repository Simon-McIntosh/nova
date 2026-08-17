"""Diagnose DIII-D vacuum-forward sensitivity without fitting the cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.diiid_vacuum_gate import _exterior, _r2, _row
from nova.biot.polygon import polygon_greens
from nova.imas.diiid_description import DiiidDescriptionRegistry, section_vertices


DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
TURN_VALUES = (1, 12, 24, 48, 72, 96)
SKEW_SIGNS = (
    ("shipped", 1.0, 1.0),
    ("angle1_flipped", -1.0, 1.0),
    ("angle2_flipped", 1.0, -1.0),
    ("both_flipped", -1.0, -1.0),
)


def _response(
    row: dict[str, Any], *, angle1_sign: float = 1.0, angle2_sign: float = 1.0
) -> tuple[tuple[str, ...], np.ndarray]:
    radius = np.asarray(row["efit_grid_R"], dtype=float)
    height = np.asarray(row["efit_grid_Z"], dtype=float)
    target_r, target_z = np.meshgrid(radius, height)
    names: list[str] = []
    responses: list[np.ndarray] = []
    for values in zip(
        row["coil_name"],
        row["coil_R"],
        row["coil_Z"],
        row["coil_width"],
        row["coil_height"],
        row["coil_angle1"],
        row["coil_angle2"],
        strict=True,
    ):
        name, coil_r, coil_z, width, extent, angle1, angle2 = values
        vertices = section_vertices(
            coil_r,
            coil_z,
            width,
            extent,
            angle1_sign * angle1,
            angle2_sign * angle2,
        )
        total_flux = polygon_greens(target_r.ravel(), target_z.ravel(), vertices)[0]
        names.append(name)
        responses.append(total_flux.reshape(target_r.shape) / (2.0 * np.pi))
    return tuple(names), np.stack(responses)


def _currents(row: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    target_time = np.asarray(row["efit_times"], dtype=float)
    source_time = np.asarray(row["magnetics_time"], dtype=float)
    columns = []
    for name in names:
        values = np.asarray(row[f"magnetics_{name}"], dtype=float)
        valid = np.isfinite(source_time) & np.isfinite(values)
        columns.append(
            1000.0 * np.interp(target_time, source_time[valid], values[valid])
        )
    return np.column_stack(columns)


def _prediction_components(
    current_matrices: list[np.ndarray],
    response: tuple[tuple[str, ...], np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    names, matrix = response
    ecoila_index = names.index("ECOILA")
    shaping = matrix.copy()
    shaping[ecoila_index] = 0.0
    return [
        (
            np.einsum("tc,czr->tzr", currents, shaping, optimize=True),
            currents[:, ecoila_index, None, None] * matrix[ecoila_index],
        )
        for currents in current_matrices
    ]


def _predictions(
    components: list[tuple[np.ndarray, np.ndarray]],
    *,
    ecoila_turns: float,
    ecoilb_turns: float,
) -> list[np.ndarray]:
    effective_turns = ecoila_turns + ecoilb_turns
    return [shaping + effective_turns * ecoila for shaping, ecoila in components]


def _prepare_exterior(
    rows: list[dict[str, Any]],
) -> list[tuple[list[np.ndarray], list[np.ndarray], np.ndarray]]:
    prepared = []
    for row in rows:
        truth = np.asarray(row["efit_psirz"], dtype=float)
        masks = []
        actual_frames = []
        for frame in range(len(truth)):
            mask = _exterior(row, frame) & np.isfinite(truth[frame])
            actual = truth[frame][mask]
            masks.append(mask)
            actual_frames.append(actual)
        prepared.append((masks, actual_frames, np.concatenate(actual_frames)))
    return prepared


def _pooled_score(
    prepared: list[tuple[list[np.ndarray], list[np.ndarray], np.ndarray]],
    predictions: list[np.ndarray],
) -> dict[str, Any]:
    signed = {}
    for sign in (1.0, -1.0):
        actual_parts = []
        predicted_parts = []
        for (masks, actual_frames, actual), prediction in zip(
            prepared, predictions, strict=True
        ):
            aligned = []
            for frame, (mask, actual_frame) in enumerate(
                zip(masks, actual_frames, strict=True)
            ):
                values = sign * prediction[frame][mask]
                aligned.append(values + np.mean(actual_frame - values))
            actual_parts.append(actual)
            predicted_parts.append(np.concatenate(aligned))
        signed[sign] = (np.concatenate(actual_parts), np.concatenate(predicted_parts))
    chosen_sign = max(signed, key=lambda sign: _r2(*signed[sign]))
    return {"pooled_r2": _r2(*signed[chosen_sign]), "global_sign": int(chosen_sign)}


def sweep(paths: list[Path]) -> dict[str, Any]:
    rows = [_row(path) for path in paths]
    prepared = _prepare_exterior(rows)
    registry = DiiidDescriptionRegistry()
    descriptions = [
        registry.ingest(row, source_row=path.name)
        for row, path in zip(rows, paths, strict=True)
    ]
    digests = {description.physical_digest for description in descriptions}
    if len(digests) != 1:
        raise RuntimeError("the selected cohort does not share one geometry")

    shipped_response = _response(rows[0])
    current_matrices = [_currents(row, shipped_response[0]) for row in rows]
    shipped_components = _prediction_components(current_matrices, shipped_response)
    turn_sweep = []
    for turns in TURN_VALUES:
        score = _pooled_score(
            prepared,
            _predictions(
                shipped_components,
                ecoila_turns=turns,
                ecoilb_turns=0.0,
            ),
        )
        turn_sweep.append({"ecoila_turns": turns, **score})

    ecoilb_sweep = []
    for present, ecoilb_turns in ((False, 0.0), (True, 48.0)):
        score = _pooled_score(
            prepared,
            _predictions(
                shipped_components,
                ecoila_turns=48.0,
                ecoilb_turns=ecoilb_turns,
            ),
        )
        ecoilb_sweep.append(
            {
                "ecoila_turns": 48,
                "ecoilb_present": present,
                "ecoilb_turns": ecoilb_turns,
                **score,
            }
        )

    skew_sweep = []
    for convention, angle1_sign, angle2_sign in SKEW_SIGNS:
        response = _response(rows[0], angle1_sign=angle1_sign, angle2_sign=angle2_sign)
        components = _prediction_components(current_matrices, response)
        score = _pooled_score(
            prepared,
            _predictions(
                components,
                ecoila_turns=1.0,
                ecoilb_turns=0.0,
            ),
        )
        skew_sweep.append(
            {
                "convention": convention,
                "angle1_sign": angle1_sign,
                "angle2_sign": angle2_sign,
                **score,
            }
        )

    names = list(rows[0]["coil_name"])
    f5a = names.index("F5A")
    f5a_vertices = section_vertices(
        rows[0]["coil_R"][f5a],
        rows[0]["coil_Z"][f5a],
        rows[0]["coil_width"][f5a],
        rows[0]["coil_height"][f5a],
        rows[0]["coil_angle1"][f5a],
        rows[0]["coil_angle2"][f5a],
    )
    return {
        "shots": len(paths),
        "frames": sum(len(row["efit_times"]) for row in rows),
        "geometry_digest": next(iter(digests)),
        "turn_sweep": turn_sweep,
        "ecoilb_sweep": ecoilb_sweep,
        "skew_sweep": skew_sweep,
        "f5a_geometry": {
            "center_m": [rows[0]["coil_R"][f5a], rows[0]["coil_Z"][f5a]],
            "extent_m": [rows[0]["coil_width"][f5a], rows[0]["coil_height"][f5a]],
            "angles_deg": [rows[0]["coil_angle1"][f5a], rows[0]["coil_angle2"][f5a]],
            "vertices_m": f5a_vertices.tolist(),
        },
        "assumptions": {
            "ecoilb": (
                "co-located with ECOILA, 48 turns, driven by the recorded "
                "ECOILA current"
            ),
            "skew": (
                "one-factor sign conventions at the shipped one-turn ECOILA baseline"
            ),
            "fitting": (
                "only one additive gauge per frame and one global sign; no "
                "amplitude or current fit"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--shots", type=int, default=20)
    args = parser.parse_args()
    paths = sorted(args.data.glob("*.parquet"))[: args.shots]
    if len(paths) != args.shots:
        raise SystemExit(f"requested {args.shots} shots, found {len(paths)}")
    print(json.dumps(sweep(paths), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
