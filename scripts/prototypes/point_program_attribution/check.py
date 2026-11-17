# ruff: noqa: E501

"""Check that the attribution record and its figure still say what the receipt says.

The card receipts live under the user's cache, which a later reader may not have,
so the numbers are carried in the repository beside the figure
(``docs/figures/kernel-compile-barrier/point-program-attribution/attribution-receipt.json``)
and this check compares the record and the figure against that copy.  It reads no
cache and runs no compile, so it is the durability check for the attribution: a
record edited away from its receipt, or a figure drawn from other data, fails
here.
"""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
FIGURE_DIR = Path("docs/figures/kernel-compile-barrier/point-program-attribution")
RECORD = Path(
    "docs/evidence/fragments/kernel-compile-barrier/point-program-attribution.html"
)
DEVICE = "NVIDIA H200 NVL"


def number(value):
    return "n/a" if value is None else format(value, ",")


def load(path):
    return json.loads(path.read_text())


def check_receipt(receipt):
    """The receipt must be internally consistent before it can check anything
    else: a stub's erase is the control minus its own lowering, and every arm
    that is not the control must differ from the control where it counts."""
    failures = []
    control = receipt["control_bytes"]
    blocks = {block["block"]: block for block in receipt["blocks"]}
    for block in receipt["blocks"]:
        erased = control - block["lowered_bytes"]
        if block["erased"] != (erased == 0):
            failures.append(
                "%s: erased flag %s disagrees with lowered bytes %d"
                % (block["erased"], block["block"], block["lowered_bytes"])
            )
        if abs(block["graph_share"] - erased / control) > 1e-6:
            failures.append("%s: graph share disagrees with its bytes" % block["block"])
    for name in ("residual-tangents", "identity"):
        if (
            blocks[name]["probe_bytes"] is not None
            and blocks[name]["probe_bytes"] != control
        ):
            failures.append("%s: the raising probe changed the lowering" % name)
    arms = receipt["arms"]
    if len(arms) != 3 or arms[0]["arm"] != "control":
        failures.append("the arms are not the control followed by its two stubs")
    for arm in arms:
        if arm["device"] != DEVICE:
            failures.append(
                "%s: device %r is not the H200" % (arm["arm"], arm["device"])
            )
        if arm["instructions_after_splitter"] is None:
            failures.append("%s: no instruction count after the splitter" % arm["arm"])
        if arm["cache_guard_hits"] != 0 or arm["cache_guard_misses"] < 1:
            failures.append("%s: the compile was not cold (hits/misses)" % arm["arm"])
    for arm in arms[1:]:
        if arm["instructions_after_splitter"] == arms[0]["instructions_after_splitter"]:
            failures.append("%s: the stub is invisible to the compiler" % arm["arm"])
        if arm["stablehlo_bytes"] == control:
            failures.append("%s: the stub erases nothing" % arm["arm"])
    return failures


def check_record(receipt, record):
    failures = []
    control = receipt["control_bytes"]
    for block in receipt["blocks"]:
        share = "%.2f%%" % (100.0 * block["graph_share"])
        erased = control - block["lowered_bytes"]
        if block["block"] == "identity":
            verdict = "erased (control)"
        elif block["erased"]:
            verdict = "erased, no share"
        else:
            verdict = "removed"
        row = "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            block["label"],
            number(block["lowered_bytes"]),
            number(erased),
            share,
            verdict,
        )
        if row not in record:
            failures.append("record: block row missing or stale — %s" % block["label"])
    for arm in receipt["arms"]:
        cells = "<tr><td>%s</td><td>%.3f s</td><td>%s</td><td>%s</td>" % (
            arm["label"],
            arm["compile_seconds"],
            number(arm["instructions_after_splitter"]),
            number(arm["stablehlo_bytes"]),
        )
        if cells not in record:
            failures.append("record: arm row missing or stale — %s" % arm["arm"])
        if "%.2f%%" % (100.0 * arm["coverage"]) not in record:
            failures.append("record: coverage missing — %s" % arm["arm"])
        for entry in arm["passes"]:
            row = (
                "<tr><td><code>%s</code></td><td>%.3f s</td><td>%.1f%%</td><td>%d</td><td>%s</td></tr>"
                % (
                    entry["pass"],
                    entry["seconds"],
                    entry["share_of_attributed"] * 100.0,
                    entry["dumps"],
                    number(entry["instructions_exit"]),
                )
            )
            if row not in record:
                failures.append(
                    "record: pass row missing or stale — %s (%s)"
                    % (entry["pass"], arm["arm"])
                )
    bodies = receipt["bodies"]
    for text in (
        "<code>%s</code>" % number(bodies["operations"]),
        "<code>%s</code>" % number(bodies["call_sites"]),
        "<code>%s</code>" % number(bodies["bodies_called_more_than_once"]),
        "<code>%s</code>" % number(bodies["most_call_sites_of_one_body"]),
    ):
        if text not in record:
            failures.append("record: repeated-body count missing — %s" % text)
    if DEVICE not in record:
        failures.append("record: the device every card receipt names is not quoted")
    return failures


def check_figure(receipt, svg):
    failures = []
    for arm in receipt["arms"]:
        label = "%.1f" % arm["compile_seconds"]
        if label not in svg:
            failures.append("figure: no bar label for %s (%s)" % (arm["arm"], label))
    control = receipt["arms"][0]["compile_seconds"]
    for arm, block in zip(
        receipt["arms"][1:], ("corner-height Hessian", "derivative instantiations")
    ):
        held = "%.1f s held by" % (control - arm["compile_seconds"])
        if held not in svg:
            failures.append("figure: no span label %r for the %s" % (held, block))
    if "erased" not in svg:
        failures.append("figure: the erased row carries no text mark")
    for colour in ("#cc0000", "#CC0000"):
        if colour in svg:
            failures.append(
                "figure: red %s is drawn, and nothing here is a limit" % colour
            )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt", type=Path, default=FIGURE_DIR / "attribution-receipt.json"
    )
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument(
        "--figure", type=Path, default=FIGURE_DIR / "block-attribution.svg"
    )
    args = parser.parse_args(argv)
    receipt_path = args.receipt if args.receipt.is_absolute() else REPO / args.receipt
    record_path = args.record if args.record.is_absolute() else REPO / args.record
    figure_path = args.figure if args.figure.is_absolute() else REPO / args.figure
    missing = [
        str(path)
        for path in (receipt_path, record_path, figure_path)
        if not path.exists()
    ]
    if missing:
        print("attribution check refused: missing " + ", ".join(missing))
        return 2
    receipt = load(receipt_path)
    failures = check_receipt(receipt)
    failures += check_record(receipt, record_path.read_text())
    failures += check_figure(receipt, figure_path.read_text())
    if failures:
        print("attribution check refused: %d finding(s)" % len(failures))
    for failure in failures:
        print("  " + failure)
    if failures:
        return 1
    print(
        "attribution check passed: %d blocks, %d arms and %d pass rows agree with %s"
        % (
            len(receipt["blocks"]),
            len(receipt["arms"]),
            sum(len(arm["passes"]) for arm in receipt["arms"]),
            receipt_path.name,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
