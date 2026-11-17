"""Assemble the per-block attribution of the production row program.

The CPU receipts give each block's share of the lowered graph: the bytes its
stub removes from the module, as a share of the unstubbed control.  The H200
receipts give the cold compile and the instruction count after the variadic
splitter for the arms compiled there.  A stub whose lowered size or instruction
count equals its control's is reported as erased with no share.
"""

import argparse
import json
from pathlib import Path

BLOCKS = (
    ("value", "value path"),
    ("gradient", "gradient"),
    ("hessian-height", "regular Hessian at corner heights"),
    ("residual-tangents", "near-corner weighted residual tangents"),
    ("elliptic", "elliptic terms"),
    ("identity", "identity stub (control)"),
)

ARMS = (
    ("control", "unchanged program"),
    ("hessian-height", "corner-height Hessian stubbed"),
    ("gradient", "gradient and Hessian stubbed"),
)

DIAGNOSTIC = ("priority_fusion_dump.txt", ".ll")


def load(path):
    return json.loads(Path(path).read_text())


def splitter_instructions(attribution):
    """Return the instruction count before and after the variadic splitter."""
    sequence = attribution["sequence"]
    for index, entry in enumerate(sequence):
        if entry["pass"].endswith("/variadic-op-splitter"):
            following = next(
                row
                for row in sequence[index + 1 :]
                if not row["pass"].endswith(DIAGNOSTIC)
            )
            return {
                "entry": entry["instructions"],
                "after": following["instructions"],
                "following_pass": following["pass"],
            }
    return {"entry": None, "after": None, "following_pass": None}


def block_rows(cpu, control_bytes):
    rows = []
    for name, label in BLOCKS:
        receipt = load(cpu / f"{name}.json")
        erased = control_bytes - receipt["stablehlo_bytes"]
        probe = cpu / f"{name}-probe.json"
        rows.append(
            {
                "block": name,
                "label": label,
                "lowered_bytes": receipt["stablehlo_bytes"],
                "erased_bytes": erased,
                "graph_share": erased / control_bytes,
                "erased": erased == 0,
                "replaced": receipt["replaced"],
                "probe_bytes": load(probe)["stablehlo_bytes"]
                if probe.exists()
                else None,
            }
        )
    return rows


def arm_rows(device):
    rows = []
    for name, label in ARMS:
        directory = device / name
        profile = load(directory / "profile.json")
        if "H200" not in profile["device"]:
            raise SystemExit(
                "refusing the %s receipt: device %r is not the H200, so its wall is "
                "another backend's" % (name, profile["device"])
            )
        if profile.get("phase") != "done":
            raise SystemExit(
                "refusing the %s receipt: it stopped at phase %r"
                % (name, profile.get("phase"))
            )
        attribution = load(directory / "attribution.json")
        splitter = splitter_instructions(attribution)
        rows.append(
            {
                "arm": name,
                "label": label,
                "compile_seconds": profile["compile_seconds"],
                "lower_seconds": profile["lower_seconds"],
                "stablehlo_bytes": profile["stablehlo_bytes"],
                "peak_host_rss_kib": profile["peak_host_rss_kib"],
                "device": profile["device"],
                "revision": profile["revision"],
                "cache_guard_misses": profile["cache_guard_misses"],
                "cache_guard_hits": profile["cache_guard_hits"],
                "instructions_entry": splitter["entry"],
                "instructions_after_splitter": splitter["after"],
                "attributed_seconds": attribution.get("attributed_seconds"),
                "coverage": attribution.get("coverage_of_compile"),
                "distinct_passes": attribution.get("distinct_passes"),
                "instructions_peak": attribution.get("instructions_peak"),
                "seconds_before_first_dump": attribution.get(
                    "seconds_before_first_dump"
                ),
                "seconds_after_last_dump": attribution.get("seconds_after_last_dump"),
                "table": attribution.get("table"),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = load(args.cpu / "control.json")
    receipt = {
        "program": control["program"],
        "size": control["size"],
        "revision": control["revision"],
        "control_bytes": control["stablehlo_bytes"],
        "blocks": block_rows(args.cpu, control["stablehlo_bytes"]),
        "arms": arm_rows(args.device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
