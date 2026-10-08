"""Audit temperature evidence for the vacuum shots in the passive fit.

The reference steel curve is a conditional conversion, not a measurement of
MAST coil-can stock. Missing shot evidence never becomes room temperature or
the vessel's published bake-out temperature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "docs/figures/mast-passive-held-out/split.json"
REFERENCE = ROOT / "docs/figures/mast-coil-case-resistivity/receipt.json"
SHOT_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")
OUTPUT = ROOT / "docs/figures/mast-coil-case-temperature/receipt.json"
PUBLIC_PROGRAMME = "https://opendata.ukaea.uk/mast-data/"

# NBS SRM 798, Table 4, as cited in REFERENCE; micro-ohm metre.
REFERENCE_POINTS = ((300.0, 0.811), (400.0, 0.890), (450.0, 0.923))
FIT_CEILING = 0.900
# Three-decimal tabulation permits this rounding interval at each endpoint.
REFERENCE_ROUNDING = 0.0005

_CASE_TEMPERATURE = re.compile(
    r"(?:coil[_ /-]?case|coil[_ /-]?can|pf[_ /-]?case|pf[_ /-]?can)"
    r".*(?:temp|thermocouple)|(?:temp|thermocouple).*"
    r"(?:coil[_ /-]?case|coil[_ /-]?can|pf[_ /-]?case|pf[_ /-]?can)",
    re.IGNORECASE,
)
_VESSEL_OR_BAKE = re.compile(r"vessel.*temp|temp.*vessel|bake|heater", re.I)
_TEMPERATURE = re.compile(
    r"(?<![a-z])(?:temp(?:erature)?|thermocouple|bake|heater)(?![a-z])", re.I
)


def reference_resistivity(temperature_k: float) -> float:
    """Interpolate the cited specimen only inside its measured range."""

    if not REFERENCE_POINTS[0][0] <= temperature_k <= REFERENCE_POINTS[-1][0]:
        raise ValueError("reference temperature is outside 300–450 K")
    for (low_t, low_r), (high_t, high_r) in zip(
        REFERENCE_POINTS, REFERENCE_POINTS[1:], strict=False
    ):
        if temperature_k <= high_t:
            return low_r + (high_r - low_r) * (temperature_k - low_t) / (high_t - low_t)
    raise AssertionError("temperature range was checked")


def crossing_temperature(resistivity: float) -> float:
    """Invert the measured piecewise-linear reference curve."""

    for (low_t, low_r), (high_t, high_r) in zip(
        REFERENCE_POINTS, REFERENCE_POINTS[1:], strict=False
    ):
        if low_r <= resistivity <= high_r:
            return low_t + (resistivity - low_r) * (high_t - low_t) / (high_r - low_r)
    raise ValueError("resistivity is outside the reference range")


def _temperature_channels(metadata: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Classify named signals without treating plasma temperatures as steel."""

    channels: dict[str, list[dict[str, str]]] = {
        "case": [],
        "vessel_or_bake": [],
        "unrelated": [],
    }
    for key, attrs in metadata.items():
        if not key.endswith("/.zattrs"):
            continue
        name = key.removesuffix("/.zattrs")
        identity = " ".join(
            str(attrs.get(field, ""))
            for field in ("name", "label", "description", "uda_name")
        )
        if not _TEMPERATURE.search(name + " " + identity):
            continue
        category = (
            "case"
            if _CASE_TEMPERATURE.search(name + " " + identity)
            else "vessel_or_bake"
            if _VESSEL_OR_BAKE.search(name + " " + identity)
            else "unrelated"
        )
        channels[category].append(
            {
                "path": name,
                "label": str(attrs.get("label", "")),
                "units": str(attrs.get("units", "")),
            }
        )
    return channels


def _refuse_unmapped_temperature(
    shot: int, channels: dict[str, list[dict[str, str]]]
) -> None:
    """Require a source-specific can mapping before accepting a candidate."""

    candidates = channels["case"] + channels["vessel_or_bake"]
    if candidates:
        raise ValueError(
            f"shot {shot}: physical temperature candidate requires a checked "
            f"sensor-to-can mapping: {candidates}"
        )


def build_receipt(split_path: Path = SPLIT, store: Path = SHOT_STORE) -> dict[str, Any]:
    """Audit the pinned cohort and retain provenance for every unknown shot."""

    split = json.loads(split_path.read_text())
    reference = json.loads(REFERENCE.read_text())
    if not any(
        claim.get("id") == "reference_resistivity"
        and claim.get("citations") == ["nist"]
        for claim in reference["claims"]
    ):
        raise ValueError("reference steel points lack their cited source")
    if len(split["training"]) != 131 or len(split["held_out"]) != 37:
        raise ValueError("passive-fit cohort split changed")
    if set(split["training"]) & set(split["held_out"]):
        raise ValueError("training and held-out cohorts overlap")

    rows: list[dict[str, Any]] = []
    unrelated: dict[str, list[dict[str, str]]] = {}
    for role in ("training", "held_out"):
        for shot in split[role]:
            path = store / f"{shot}.zarr/.zmetadata"
            raw = path.read_bytes()
            metadata = json.loads(raw)["metadata"]
            if "amc/p4l_feed_current/.zarray" not in metadata:
                raise ValueError(
                    f"shot {shot}: coil-current positive control is absent"
                )
            channels = _temperature_channels(metadata)
            _refuse_unmapped_temperature(shot, channels)
            if channels["unrelated"]:
                unrelated[str(shot)] = channels["unrelated"]
            rows.append(
                {
                    "shot": shot,
                    "split": role,
                    "temperature_status": "unknown",
                    "can_temperature_k": None,
                    "implied_resistivity_micro_ohm_m": None,
                    "reaches_fit_ceiling": None,
                    "source": str(path),
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "operations_context": (
                        "TF sliding-joint thermocouple tests, not a PF can reading"
                        if 25721 <= shot <= 25729
                        else None
                    ),
                    "reason": (
                        "no coil-can or vessel/bake temperature signal in level-1 "
                        "metadata; no shot-linked operations log or can thermal "
                        "model supplied"
                    ),
                }
            )

    lower_crossing = crossing_temperature(FIT_CEILING - REFERENCE_ROUNDING)
    upper_crossing = crossing_temperature(FIT_CEILING + REFERENCE_ROUNDING)
    return {
        "question": (
            "Does sourced can temperature put the reference steel at the "
            "fitted resistivity ceiling?"
        ),
        "split_source": split_path.resolve().relative_to(ROOT).as_posix(),
        "material_source": reference["sources"]["nist"],
        "bake_source": reference["sources"]["akers"],
        "public_programme_source": {
            "url": PUBLIC_PROGRAMME,
            "finding": (
                "The public programme index gives objectives, not calibrated "
                "PF can temperatures. Its 25721–25729 objective mentions TF "
                "sliding-joint thermocouple tests, a different component."
            ),
        },
        "reference_points": [
            {"temperature_k": t, "resistivity_micro_ohm_m": rho}
            for t, rho in REFERENCE_POINTS
        ],
        "method": (
            "piecewise-linear interpolation, no extrapolation; reference "
            "specimen is not identified as MAST can stock"
        ),
        "uncertainty_declared_before_scoring": {
            "reference_tabulation_rounding_micro_ohm_m": REFERENCE_ROUNDING,
            "case_temperature": (
                "unbounded without a can measurement or cited shot-linked thermal model"
            ),
            "sample_to_can_material": "unbounded without a can alloy identification",
            "scope": "rounding interval applies to the reference specimen only",
        },
        "fit_ceiling_micro_ohm_m": FIT_CEILING,
        "reference_crossing_temperature_k": crossing_temperature(FIT_CEILING),
        "reference_crossing_rounding_interval_k": [lower_crossing, upper_crossing],
        "published_vessel_bake_c": 140,
        "published_vessel_bake_is_shot_can_temperature": False,
        "positive_control": {
            "signal": "amc/p4l_feed_current",
            "metadata_records_present": len(rows),
        },
        "excluded_plasma_temperature_channels": unrelated,
        "counts": {
            "training": len(split["training"]),
            "held_out": len(split["held_out"]),
            "with_can_temperature_evidence": 0,
            "unknown": len(rows),
            "reference_ceiling_reached": 0,
            "reference_ceiling_not_reached": 0,
        },
        "verdict": (
            "indeterminate for all fitted shots; no can resistivity is inferred "
            "from vessel bake-out or plasma temperatures"
        ),
        "shots": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    receipt = build_receipt(store=args.store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"counts": receipt["counts"], "verdict": receipt["verdict"]}))


if __name__ == "__main__":
    main()
