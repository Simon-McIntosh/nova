"""Verify the saved ring-quadrature fields and scorecard independently."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    report = json.loads((RESULTS / "ring-quadrature-results.json").read_text())
    with np.load(RESULTS / "ring-quadrature-fields.npz") as stored:
        fields = {name: stored[name] for name in stored.files}

    assert fields["targets"].shape == (731, 2)
    assert fields["centres"].shape == (566, 2)
    assert int(fields["ring_mask"].sum()) == 96
    assert int((fields["ring_mask"] & fields["available_mask"]).sum()) == 0
    assert np.all(np.isfinite(fields["own_geometry_shift"]))
    assert np.all(np.isfinite(fields["one_sided_shift"]))

    available = fields["available_mask"] & ~fields["ring_mask"]
    baseline_bits = fields["baseline_m0"][available].view(np.uint64)
    for candidate in ("own_geometry", "one_sided"):
        candidate_bits = fields[f"{candidate}_m0"][available].view(np.uint64)
        assert np.array_equal(candidate_bits, baseline_bits)
        shift = fields[f"{candidate}_shift"]
        saved = report["candidates"][candidate]
        assert np.isclose(shift[440], saved["argmax_shift_wb"], rtol=0.0, atol=1e-14)
        assert np.isclose(
            np.max(np.abs(shift)), saved["all_target_sup_wb"], rtol=0.0, atol=1e-14
        )
        assert saved["passes_all"] is False

    with (RESULTS / "ring-cell-errors.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 192
    assert {row["candidate"] for row in rows} == {"own_geometry", "one_sided"}

    expected_hashes = {}
    for line in (ROOT / "inputs/sha256sums.txt").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        expected_hashes[Path(path).name] = digest
    for name, expected in expected_hashes.items():
        assert sha256(ROOT / "inputs" / name) == expected

    figure = Path(
        "docs/figures/boundary-ring-source-completion/ring-m0-relative-error.png"
    )
    assert figure.stat().st_size > 100_000
    print(
        "PASS: 96 ring cells scored, 351 established cells bitwise unchanged, "
        "both negative candidate fields and input hashes verified"
    )


if __name__ == "__main__":
    main()
