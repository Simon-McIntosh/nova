"""Report which banked certificate receipts the carrier check would refuse.

Each receipt under the production-route-parts directory is opened, and the
carrier, fixture and declared current this tree builds for its case and
resolution are compared against what the receipt records. The check never
writes: a receipt is read, judged and left as it was.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

PATH_ROOT = Path(__file__).resolve().parents[3]
if str(PATH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATH_ROOT))

from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402

PARTS = certificate.PART_ROOT


def _resolution(path: Path) -> tuple[str, int] | None:
    """Return the case and requested cell count a part file names.

    ``_part_path`` spells the reduced resolution ``-110`` as ``reduced``, so both
    spellings are resolved here; a part under neither name is not one the driver
    can serve.
    """

    match = re.match(
        r"(?P<case>.+)-production-route-cells-(?P<cells>\d+)\.json$", path.name
    )
    if match is not None:
        return match.group("case"), -int(match.group("cells"))
    match = re.match(r"(?P<case>.+)-production-route-reduced\.json$", path.name)
    if match is not None:
        return match.group("case"), -110
    return None


def _banked_date(path: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(PATH_ROOT),
            "log",
            "-1",
            "--format=%as",
            "--",
            str(path.relative_to(PATH_ROOT)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "uncommitted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pattern", default="*.json")
    arguments = parser.parse_args()
    configure_dtypes()

    rows = []
    cache: dict[tuple[str, int], dict] = {}
    for path in sorted(PARTS.glob(arguments.pattern)):
        resolution = _resolution(path)
        if resolution is None:
            continue
        case_name, requested_cells = resolution
        persisted = json.loads(path.read_text(encoding="utf-8"))
        served = "render_data" in persisted
        key = (case_name if case_name in certificate.CASE_NAMES else case_name, requested_cells)
        if key not in cache:
            try:
                cache[key] = certificate._certificate_case_setup(
                    case_name, requested_cells, None, {}
                )
            except Exception as error:  # a case the builder no longer serves
                cache[key] = {"error": repr(error)}
        setup = cache[key]
        row = {
            "file": path.name,
            "case": persisted.get("case", case_name),
            "requested_cells": requested_cells,
            "banked_target_current_a": (persisted.get("solver") or {}).get(
                "target_current_a"
            ),
            "banked_semantic_key": (persisted.get("cache") or {}).get("semantic_key"),
            "banked_realised_cells": persisted.get("realised_cells"),
            "carries_render_data": served,
            "banked_date": _banked_date(path),
        }
        if "error" in setup:
            row["refused"] = True
            row["reason"] = f"builder refused: {setup['error']}"
            row["declared_target_current_a"] = None
            row["declared_semantic_key"] = None
            row["declared_realised_cells"] = None
        else:
            verdict = certificate._receipt_verdict(persisted, setup)
            row["refused"] = not verdict["accepted"]
            row["reason"] = verdict["reason"]
            row["declared_target_current_a"] = float(setup["target_current"])
            row["declared_semantic_key"] = (setup["machine"].cache or {}).get(
                "semantic_key"
            )
            row["declared_realised_cells"] = int(len(setup["machine"].node))
        rows.append(row)
        print(
            f"{path.name}: refused={row['refused']} served={served} "
            f"banked={row['banked_target_current_a']!r} "
            f"declared={row['declared_target_current_a']!r}",
            flush=True,
        )

    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()