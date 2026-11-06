"""Run the near-X-point measurement and write its receipt and field archive."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from benchmarks import mast_response_carrier_warm as response_carrier  # noqa: E402
import measure_promoted_nodes as measure  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit-index", type=int, default=12)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.perf_counter()
    receipt, fields = measure.measure(
        arguments.edit_index, response_carrier.DEFAULT_CARRIER
    )
    measure._print_headlines(receipt)
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_text(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            allow_nan=True,
            default=measure._json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(arguments.fields, **fields)
    print("RECEIPT %s" % arguments.receipt)
    print("FIELDS %s" % arguments.fields)
    print("SECONDS %.1f" % (time.perf_counter() - started))


if __name__ == "__main__":
    main()
