"""Exercise source and input provenance against retained measurement receipts."""

import argparse
import copy
import json
from pathlib import Path
import tempfile

from scripts.prototypes.bank_h200_slice_profile.report import (
    attributed,
    measured_source,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--mutation", choices=["revision", "source", "inputs", "compiled"]
    )
    args = parser.parse_args()
    main_path = args.run / "h200/main-slice-16-1/receipt.json"
    main_record = json.loads(main_path.read_text())
    historical = json.loads(
        (args.run / "h200/historical-read-16-1/receipt.json").read_text()
    )
    for record in (main_record, historical):
        print("POSITIVE_SOURCE", measured_source(record), flush=True)
    with tempfile.TemporaryDirectory(prefix="profile-control-") as temporary:
        record = copy.deepcopy(
            historical if args.mutation == "revision" else main_record
        )
        if args.mutation == "revision":
            record.update(
                module=main_record["module"], module_sha256=main_record["module_sha256"]
            )
        elif args.mutation == "source":
            source = Path(temporary) / "support_read.py"
            source.write_bytes(
                Path(record["module"]).read_bytes() + b"\n# source mismatch\n"
            )
            record["module"] = str(source)
        elif args.mutation == "inputs":
            inputs = Path(temporary) / "common-streams.npz"
            inputs.write_bytes(Path(record["inputs_path"]).read_bytes() + b"mismatch")
            record["inputs_path"] = str(inputs)
        elif args.mutation == "compiled":
            record["module_original"] = str(Path(temporary) / "support_read.py")
            attributed(record, main_path.parent, "solve")
            raise AssertionError("compiled module mismatch escaped refusal")
        measured_source(record)
        if args.mutation:
            raise AssertionError("provenance mutation escaped refusal")
    print("PROVENANCE_CONTROLS_PASS", flush=True)


if __name__ == "__main__":
    main()
