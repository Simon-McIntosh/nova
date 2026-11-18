"""Check evidence claims against the preserved compile and lowering receipts."""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path

from scripts.prototypes.point_program_attribution.summarise import splitter_instructions


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def check_record(fragment, root):
    parser = Text()
    parser.feed(fragment.read_text())
    text = " ".join(" ".join(parser.parts).split())
    attribution = json.loads((root / "residual-h200/attribution.json").read_text())
    fixture = json.loads((root / "bounded-fixture/lowered-size.json").read_text())
    splitter = splitter_instructions(attribution)
    assert splitter["entry"] > 0 and splitter["after"] > 0
    assert any(
        row["pass"] == splitter["following_pass"] for row in attribution["sequence"]
    )
    assert fixture["phase"] == "done" and fixture["compiled"] is False
    assert fixture["cells"] == 8 and len(fixture["points"]) == 3
    size = fixture["stablehlo_bytes"]
    assert size > 0 and fixture["ceiling"] == 10000000
    return {
        "residual_instruction_provenance": all(
            claim in text
            for claim in (
                f"{splitter['entry']:,} instructions at splitter entry",
                f"{splitter['after']:,} in the first subsequent non-diagnostic dump",
                splitter["following_pass"],
                "No immediate splitter-exit dump was recorded",
                "bounded-fixture/residual-instructions.json",
            )
        ),
        "unmeasured_control_is_explicit": all(
            claim in text
            for claim in (
                "The repaired base was not cold-profiled in this run",
                "405 s is inherited context",
                "the paired 60 s control was not exercised",
                "waived for this accepted negative result",
            )
        ),
        "bounded_fixture_size_is_reported": all(
            claim in text
            for claim in (
                "tests/equilibrium/test_bounded_point_program.py",
                f"{size:,} StableHLO bytes",
                "6,000,000-byte ceiling is not reachable by the restored form",
                "bounded-fixture/lowered-size.json",
            )
        )
        and size > 6000000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    args = parser.parse_args()
    results = check_record(args.fragment, args.receipt_root)
    for name, passed in results.items():
        print(f"{'PASSED' if passed else 'FAILED'} evidence::{name}")
    failed = sum(not passed for passed in results.values())
    print(f"{failed} failed, {len(results) - failed} passed")
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
