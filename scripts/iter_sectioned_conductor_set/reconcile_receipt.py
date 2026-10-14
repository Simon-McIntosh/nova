"""Apply whole-coil consistency and source resistance to a saved spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.iter_sectioned_conductor_set.build import SOURCE, read_ids


def reconcile(raw: dict, reference: dict) -> dict:
    """Apply the same terminal correction as the section-level low-rank update."""
    names = raw["coil_names"]
    if names != reference["coil_names"]:
        raise ValueError("whole-coil reference ordering differs from sectioned set")
    source = read_ids(*SOURCE[0])
    source_names = [
        str(coil.name).rsplit("(", 1)[-1].removesuffix(")") for coil in source.coil
    ]
    if names != source_names:
        raise ValueError("active source resistance ordering differs from sectioned set")
    resistance = np.asarray([float(coil.resistance) for coil in source.coil])
    existing_resistance = np.asarray(
        raw.get("coil_resistance_ohm", np.zeros_like(resistance))
    )
    original = np.asarray(raw["coil_inductance_h"])
    target = np.asarray(reference["coil_inductance_h"])
    frequency = np.asarray(raw["frequency_hz"])
    shift = 2 * np.pi * frequency[:, None] * np.diag(target - original)[None, :]
    revised = dict(raw)
    revised["raw_coil_inductance_h"] = raw["coil_inductance_h"]
    revised["coil_inductance_h"] = target.tolist()
    revised["coil_resistance_ohm"] = resistance.tolist()
    revised["terminal_reference_applied"] = True
    revised["screened_imag_ohm"] = (
        np.asarray(raw["screened_imag_ohm"]) + shift
    ).tolist()
    revised["bare_imag_ohm"] = (np.asarray(raw["bare_imag_ohm"]) + shift).tolist()
    revised["screened_real_ohm"] = (
        np.asarray(raw["screened_real_ohm"])
        + (resistance - existing_resistance)[None, :]
    ).tolist()
    revised["bare_real_ohm"] = np.broadcast_to(
        resistance[None, :], np.asarray(raw["bare_imag_ohm"]).shape
    ).tolist()
    screened = np.asarray(revised["bare_real_ohm"]) + 1j * np.asarray(
        revised["bare_imag_ohm"]
    )
    bare = (
        resistance[None, :] + 2j * np.pi * frequency[:, None] * np.diag(target)[None, :]
    )
    revised["bare_error"] = float(np.max(np.abs(screened / bare - 1)))
    return revised


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(
        json.loads(args.receipt.read_text()), json.loads(args.reference.read_text())
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(
        f"SUMMARY sections={result['section_count']} passive={result['passive_count']} "
        f"bare_error={result['bare_error']:.3g}"
    )


if __name__ == "__main__":
    main()
