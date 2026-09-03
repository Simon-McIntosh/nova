"""Reproduce the solenoid inclusion ladder's all-conductors vacuum field
from the persisted DIII-D machine description.

The ladder (``benchmarks/diiid_solenoid_inclusion_ladder.py``) builds its
24-conductor response from two independent geometry reads: the eighteen
F-coils plus ECOILA from the row-derived challenge description, and the five
omitted ohmic channels (ECOILB, E567UP, E567DN, E89UP, E89DN) from the raw
source netCDF (200000.nc, DD 3.41.0).  This driver instead reads every one of
the 24 poloidal coils from the single persisted machine description
(``diiid_machine_description.nc``, DD 4.1.1) through
``active_coil_response_from_imas``, drives both constructions with the
identical all-conductors current vector the ladder used, and measures whether
the resulting poloidal flux and field agree to roundoff.  No plasma is
present; only conductor Green's-function response is compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.diiid_diverted_root_full_currents import _omitted_vertices
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _read,
    build_profile,
)
from nova.biot.polygon import polygon_greens
from nova.imas.diiid_description import (
    POLOIDAL_CONDUCTORS,
    dataset_machine_description,
)
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/diiid-vertical-force-balance")
RECEIPT_NAME = "vacuum-field-reproduction.json"
FIGURE_NAME = "vacuum-field-reproduction.png"
PERSISTED_ENTRY = Path(
    "docs/figures/diiid-forward-onboarding/ids-set/diiid_machine_description.nc"
)
PERSISTED_DD_VERSION = "4.1.1"
ROUNDOFF_RELATIVE_BOUND = 1e-12
PSEUDO_WALL_EXPANSION = None

# Reproduced from benchmarks.diiid_solenoid_inclusion_ladder rather than
# imported: that module's own import chain currently raises ImportError
# (diiid_forward_gs_match no longer exports ``_separatrix``), a pre-existing
# defect outside this driver's write scope.  These values are copied
# unchanged from the ladder's all-conductors rung.
ECOILA_INDEX = POLOIDAL_CONDUCTORS.index("ECOILA")
MISSING_CONDUCTOR_ORDER = ("ECOILB", "E567UP", "E567DN", "E89UP", "E89DN")
CURRENT_SCALE = {
    "ECOILB": 1.0172,
    "E567UP": 0.9929,
    "E567DN": 0.9823,
    "E89UP": 0.9806,
    "E89DN": 1.0165,
}
ALL_CONDUCTOR_NAMES = tuple(POLOIDAL_CONDUCTORS) + MISSING_CONDUCTOR_ORDER

FRAMES = (
    {"shot": "d3d_shot_0003ff34e7.parquet", "frame": 89},
    {"shot": "d3d_shot_00000c4a7b.parquet", "frame": 102},
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _persisted_response(
    entry_path: Path,
    dd_version: str,
    coil_names: tuple[str, ...],
    target_r: np.ndarray,
    target_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read every persisted pf_active coil element and return (psi, Br, Bz).

    Mirrors ``nova.imas.diiid_description.active_coil_response_from_imas``
    but keeps all three ``polygon_greens`` outputs instead of flux alone, so
    the field can be compared as well as the flux.
    """

    import imas

    shape = target_r.shape
    psi = np.zeros((len(coil_names), *shape), dtype=float)
    br = np.zeros((len(coil_names), *shape), dtype=float)
    bz = np.zeros((len(coil_names), *shape), dtype=float)
    records = []
    with imas.DBEntry(Path(entry_path), "r", dd_version=dd_version) as entry:
        active = entry.get("pf_active", autoconvert=False)
        written_dd = str(active.ids_properties.version_put.data_dictionary)
        if written_dd != dd_version:
            raise RuntimeError(f"expected DD {dd_version}, read {written_dd}")
        coils = {str(coil.name): coil for coil in active.coil}
        missing = [name for name in coil_names if name not in coils]
        if missing:
            raise RuntimeError(f"persisted description is missing coils: {missing}")
        for index, name in enumerate(coil_names):
            coil = coils[name]
            turn_sum = 0.0
            for element in coil.element:
                geometry = element.geometry
                geometry_type = int(geometry.geometry_type)
                if geometry_type != 1:
                    raise RuntimeError(
                        f"unsupported geometry type {geometry_type} for {name}"
                    )
                vertices = np.column_stack(
                    (
                        np.asarray(geometry.outline.r, dtype=float),
                        np.asarray(geometry.outline.z, dtype=float),
                    )
                )
                turns = float(element.turns_with_sign)
                turn_sum += turns
                point_psi, point_br, point_bz = polygon_greens(
                    target_r.ravel(), target_z.ravel(), vertices
                )
                psi[index] += turns * point_psi.reshape(shape)
                br[index] += turns * point_br.reshape(shape)
                bz[index] += turns * point_bz.reshape(shape)
            records.append(
                {
                    "coil": name,
                    "elements": len(coil.element),
                    "signed_turn_sum": turn_sum,
                }
            )
    meta = {
        "entry": str(entry_path),
        "dd_version": dd_version,
        "coils": records,
        "kernel": "nova.biot.polygon.polygon_greens",
    }
    return psi, br, bz, meta


def _ladder_shipped_response(
    row: dict[str, Any], target_r: np.ndarray, target_z: np.ndarray
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Return (psi, Br, Bz) per shipped conductor from the row-derived description.

    Reproduces the geometry read the ladder uses for its eighteen F-coils and
    ECOILA (``dataset_machine_description(row).physical``), the same source
    ``nova.imas.diiid_description.vacuum_response`` draws its flux-only
    response from.
    """

    description = dataset_machine_description(
        row, source_row=str(row.get("_source_path", "corpus row"))
    ).physical
    shape = target_r.shape
    names: list[str] = []
    psi = []
    br = []
    bz = []
    for conductor in description.conductors:
        if (
            conductor.vertices is None
            or not conductor.turns.affects_axisymmetric_poloidal_flux
        ):
            continue
        point_psi, point_br, point_bz = polygon_greens(
            target_r.ravel(), target_z.ravel(), conductor.vertices
        )
        names.append(conductor.name)
        psi.append(point_psi.reshape(shape))
        br.append(point_br.reshape(shape))
        bz.append(point_bz.reshape(shape))
    return tuple(names), np.stack(psi), np.stack(br), np.stack(bz)


def _ladder_omitted_response(
    geometry: dict[str, tuple[tuple[np.ndarray, float], ...]],
    target_r: np.ndarray,
    target_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (psi, Br, Bz) per omitted conductor from the raw-source geometry.

    Reproduces the geometry read ``benchmarks.diiid_diverted_root_full_currents
    .omitted_response`` performs against the raw source netCDF at DD 3.41.0.
    """

    shape = target_r.shape
    psi = np.zeros((len(MISSING_CONDUCTOR_ORDER), *shape), dtype=float)
    br = np.zeros((len(MISSING_CONDUCTOR_ORDER), *shape), dtype=float)
    bz = np.zeros((len(MISSING_CONDUCTOR_ORDER), *shape), dtype=float)
    for index, name in enumerate(MISSING_CONDUCTOR_ORDER):
        for vertices, turns in geometry[name]:
            point_psi, point_br, point_bz = polygon_greens(
                target_r.ravel(), target_z.ravel(), vertices
            )
            psi[index] += turns * point_psi.reshape(shape)
            br[index] += turns * point_br.reshape(shape)
            bz[index] += turns * point_bz.reshape(shape)
    return psi, br, bz


def _max_abs_relative(persisted: np.ndarray, ladder: np.ndarray) -> dict[str, float]:
    """Return the maximum absolute difference and its scale-normalised relative."""

    diff = np.abs(persisted - ladder)
    scale = float(np.max(np.abs(ladder)))
    max_abs = float(np.max(diff))
    return {
        "max_absolute_difference": max_abs,
        "ladder_max_absolute_value": scale,
        "max_relative_to_ladder_peak": (max_abs / scale) if scale > 0.0 else None,
    }


def _per_coil_contribution(
    name: str,
    current: float,
    persisted_field: np.ndarray,
    ladder_field: np.ndarray,
) -> float:
    """Return the maximum absolute per-coil field disagreement at one current."""

    return float(current) * float(np.max(np.abs(persisted_field - ladder_field)))


def score_frame(data: Path, shot: str, frame: int) -> dict[str, Any]:
    """Reproduce one frame's all-conductors vacuum flux and field."""

    columns = tuple(
        dict.fromkeys((*_LABEL_COLUMNS, *_CURRENT_COLUMNS, *_GEOMETRY_COLUMNS))
    )
    source = data / shot
    row = _read(source, columns)
    row["_source_path"] = str(source)

    profile, _seed, _label, _wall, _reliable, _statement = build_profile(
        row, frame, PSEUDO_WALL_EXPANSION
    )
    shipped = np.asarray(profile.operator.external_current, dtype=float)
    if shipped.size != len(POLOIDAL_CONDUCTORS):
        raise RuntimeError("forward profile does not carry the shipped 19 conductors")
    ecoila = float(shipped[ECOILA_INDEX])
    omitted_currents = {
        name: CURRENT_SCALE[name] * ecoila for name in MISSING_CONDUCTOR_ORDER
    }
    all_conductors_currents = np.r_[
        shipped,
        [omitted_currents[name] for name in MISSING_CONDUCTOR_ORDER],
    ]

    radius = np.asarray(row["efit_grid_R"], dtype=float)
    height = np.asarray(row["efit_grid_Z"], dtype=float)
    target_r, target_z = np.meshgrid(radius, height)

    shipped_names, shipped_psi, shipped_br, shipped_bz = _ladder_shipped_response(
        row, target_r, target_z
    )
    if tuple(shipped_names) != POLOIDAL_CONDUCTORS:
        raise RuntimeError("shipped conductor order differs from POLOIDAL_CONDUCTORS")
    geometry = _omitted_vertices()
    omitted_psi, omitted_br, omitted_bz = _ladder_omitted_response(
        geometry, target_r, target_z
    )

    ladder_psi = np.einsum("c,czr->zr", shipped, shipped_psi) + np.einsum(
        "c,czr->zr",
        np.asarray([omitted_currents[name] for name in MISSING_CONDUCTOR_ORDER]),
        omitted_psi,
    )
    ladder_br = np.einsum("c,czr->zr", shipped, shipped_br) + np.einsum(
        "c,czr->zr",
        np.asarray([omitted_currents[name] for name in MISSING_CONDUCTOR_ORDER]),
        omitted_br,
    )
    ladder_bz = np.einsum("c,czr->zr", shipped, shipped_bz) + np.einsum(
        "c,czr->zr",
        np.asarray([omitted_currents[name] for name in MISSING_CONDUCTOR_ORDER]),
        omitted_bz,
    )

    persisted_psi_all, persisted_br_all, persisted_bz_all, persisted_meta = (
        _persisted_response(
            PERSISTED_ENTRY,
            PERSISTED_DD_VERSION,
            ALL_CONDUCTOR_NAMES,
            target_r,
            target_z,
        )
    )
    persisted_psi = np.einsum("c,czr->zr", all_conductors_currents, persisted_psi_all)
    persisted_br = np.einsum("c,czr->zr", all_conductors_currents, persisted_br_all)
    persisted_bz = np.einsum("c,czr->zr", all_conductors_currents, persisted_bz_all)

    ladder_response = np.concatenate([shipped_psi, omitted_psi], axis=0)
    ladder_response_br = np.concatenate([shipped_br, omitted_br], axis=0)
    ladder_response_bz = np.concatenate([shipped_bz, omitted_bz], axis=0)
    per_coil = []
    for index, name in enumerate(ALL_CONDUCTOR_NAMES):
        current = float(all_conductors_currents[index])
        per_coil.append(
            {
                "coil": name,
                "current_a": current,
                "max_absolute_psi_difference_wb": _per_coil_contribution(
                    name, current, persisted_psi_all[index], ladder_response[index]
                ),
                "max_absolute_br_difference_t": _per_coil_contribution(
                    name, current, persisted_br_all[index], ladder_response_br[index]
                ),
                "max_absolute_bz_difference_t": _per_coil_contribution(
                    name, current, persisted_bz_all[index], ladder_response_bz[index]
                ),
            }
        )
    per_coil.sort(key=lambda item: item["max_absolute_psi_difference_wb"], reverse=True)

    psi_metrics = _max_abs_relative(persisted_psi, ladder_psi)
    br_metrics = _max_abs_relative(persisted_br, ladder_br)
    bz_metrics = _max_abs_relative(persisted_bz, ladder_bz)
    worst_relative = max(
        value
        for value in (
            psi_metrics["max_relative_to_ladder_peak"],
            br_metrics["max_relative_to_ladder_peak"],
            bz_metrics["max_relative_to_ladder_peak"],
        )
        if value is not None
    )
    within_roundoff = bool(worst_relative <= ROUNDOFF_RELATIVE_BOUND)
    return {
        "shot": shot,
        "frame": frame,
        "grid_shape": [int(height.size), int(radius.size)],
        "shipped_ecoila_current_a": ecoila,
        "current_vector": {
            "order": list(ALL_CONDUCTOR_NAMES),
            "currents_a": all_conductors_currents.tolist(),
            "provenance": (
                "recorded ECOILA current (shipped magnetics_ECOILA channel, "
                "kA.turn to A.turn) times each conductor's fixed drive scale "
                "from the solenoid inclusion ladder's all-conductors rung; the "
                "eighteen F-coils and ECOILA itself carry their own shipped "
                "channel currents unchanged"
            ),
        },
        "flux_wb": psi_metrics,
        "radial_field_t": br_metrics,
        "vertical_field_t": bz_metrics,
        "roundoff_bound_relative": ROUNDOFF_RELATIVE_BOUND,
        "worst_max_relative_to_ladder_peak": worst_relative,
        "within_roundoff_bound": within_roundoff,
        "per_coil_disagreement_ranked": per_coil,
        "persisted_entry_meta": persisted_meta,
    }


def render_figure(frames: list[dict[str, Any]], path: Path) -> None:
    """Plot per-frame maximum relative flux/field disagreement against the bound."""

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    labels = [f"{item['shot'][9:17]}:{item['frame']}" for item in frames]
    x = np.arange(len(frames))
    width = 0.25
    quantities = (
        ("flux_wb", "flux"),
        ("radial_field_t", "B_R"),
        ("vertical_field_t", "B_Z"),
    )
    for offset, (key, label) in enumerate(quantities):
        values = [item[key]["max_relative_to_ladder_peak"] or np.nan for item in frames]
        axis.bar(x + (offset - 1) * width, values, width, label=label)
    axis.axhline(
        ROUNDOFF_RELATIVE_BOUND,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"roundoff bound {ROUNDOFF_RELATIVE_BOUND:.0e}",
    )
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_ylabel("max |persisted - ladder| / max |ladder|")
    axis.set_title(
        "Persisted DIII-D description vs. solenoid inclusion ladder, "
        "all-conductors rung"
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(data: Path, output: Path) -> dict[str, Any]:
    """Score every declared frame and persist the reproduction receipt."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    if not PERSISTED_ENTRY.exists():
        raise RuntimeError(f"persisted machine description absent: {PERSISTED_ENTRY}")
    frames = [score_frame(data, item["shot"], item["frame"]) for item in FRAMES]
    figure_path = output / FIGURE_NAME
    render_figure(frames, figure_path)
    receipt = {
        "measurement": (
            "vacuum flux and field from the persisted DIII-D machine description "
            "against the solenoid inclusion ladder's all-conductors rung"
        ),
        "persisted_description": {
            "path": str(PERSISTED_ENTRY),
            "sha256": _sha256(PERSISTED_ENTRY),
            "dd_version": PERSISTED_DD_VERSION,
            "conductor_count": len(ALL_CONDUCTOR_NAMES),
        },
        "roundoff_bound_relative": ROUNDOFF_RELATIVE_BOUND,
        "all_frames_within_roundoff_bound": bool(
            all(item["within_roundoff_bound"] for item in frames)
        ),
        "device": "cpu",
        "nova_equilibrium_modified": False,
        "frames": frames,
        "artifacts": {
            "receipt": str(output / RECEIPT_NAME),
            "figure": str(figure_path),
        },
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.data, arguments.output)
    print(
        json.dumps(
            {
                "all_frames_within_roundoff_bound": receipt[
                    "all_frames_within_roundoff_bound"
                ],
                "frames": [
                    {
                        "shot": item["shot"],
                        "frame": item["frame"],
                        "worst_max_relative_to_ladder_peak": item[
                            "worst_max_relative_to_ladder_peak"
                        ],
                    }
                    for item in receipt["frames"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
