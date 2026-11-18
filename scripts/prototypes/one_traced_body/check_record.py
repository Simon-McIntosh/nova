"""Replay the negative-result record check from its repository receipt."""

import argparse
from pathlib import Path

from scripts.prototypes.point_program_attribution import check as attribution_check
from scripts.prototypes.point_program_attribution.summarise import splitter_instructions

REPO = Path(__file__).resolve().parents[3]
RECEIPT = Path(
    "docs/figures/kernel-compile-barrier/one-traced-body/evidence-receipt.json"
)
RECORD = Path("docs/evidence/fragments/kernel-compile-barrier/one-traced-body.html")


def required_claims(receipt):
    attribution = receipt["residual"]["attribution"]
    profile = receipt["residual"]["profile"]
    fixture = receipt["fixture"]
    splitter = splitter_instructions(attribution)
    assert splitter["entry"] > 0 and splitter["after"] > 0
    assert profile["phase"] == "done" and "H200" in profile["device"]
    assert fixture["phase"] == "done" and fixture["compiled"] is False
    assert fixture["cells"] == 8 and len(fixture["points"]) == 3
    assert fixture["stablehlo_bytes"] > 6000000 and fixture["ceiling"] == 10000000
    assert receipt["control"]["status"] == "not_measured"
    return {
        "residual_instruction_provenance": (
            f"{splitter['entry']:,} instructions at splitter entry",
            f"{splitter['after']:,} in the first subsequent non-diagnostic dump",
            f"{attribution['instructions_peak']:,} peak instructions",
            splitter["following_pass"],
            "No immediate splitter-exit dump was recorded",
            "bounded-fixture/residual-instructions.json",
        ),
        "unmeasured_control_is_explicit": (
            "The repaired base was not cold-profiled in this run",
            f"{receipt['control']['inherited_compile_seconds']} s is inherited context",
            "the paired 60 s control was not exercised",
            "waived for this accepted negative result",
        ),
        "bounded_fixture_size_is_reported": (
            fixture["fixture"],
            f"{fixture['stablehlo_bytes']:,} StableHLO bytes",
            "6,000,000-byte ceiling is not reachable by the restored form",
            "bounded-fixture/lowered-size.json",
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, default=RECORD)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args(argv)
    record_path = REPO / args.fragment
    receipt_path = REPO / args.receipt
    receipt = attribution_check.load(receipt_path)
    claims = required_claims(receipt)
    failures = attribution_check.check_record(
        {"required_claims": claims}, record_path.read_text()
    )
    for failure in failures:
        print("FAILED evidence::" + failure)
    if not failures:
        for name in claims:
            print("PASSED evidence::" + name)
    print(f"{len(failures)} failed, {len(claims) - len(failures)} passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
