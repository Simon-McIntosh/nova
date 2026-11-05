"""Check observed update events against retained solver histories."""

import argparse
import json
from pathlib import Path
import re

import numpy as np


def read_arm(path):
    text = path.read_text()
    lines = text.splitlines()
    if not lines[-1].startswith("EXIT="):
        raise ValueError(f"incomplete gate: {path}")
    summary = re.findall(r"^=+ (.* in .*?) =+$", text, re.MULTILINE)
    if not summary:
        raise ValueError(f"missing pytest summary: {path}")
    events = [json.loads(line[7:]) for line in lines if line.startswith("UPDATE ")]
    results = [json.loads(line[7:]) for line in lines if line.startswith("RESULT ")]
    phases = {}
    for result in results:
        phase = result["phase"]
        limit = len(result["update_path"][0])
        updates = [event for event in events if event["limit"] == limit]
        assert len(updates) == max(result["steps"]), "observer missed an update"
        path_values = np.asarray(result["residual_path"])
        for index, event in enumerate(updates, 1):
            np.testing.assert_array_equal(
                event["fresh_residual"], path_values[:, index]
            )
        trials = np.sum([event["trials"] for event in updates], axis=0)
        releases = np.sum([event["releases"] for event in updates], axis=0)
        np.testing.assert_array_equal(trials + 1, result["reads"])
        np.testing.assert_array_equal(releases, result["unfreezes"])
        phases[phase] = {
            "compile_seconds": float(
                re.search(rf"^COMPILE {phase} (\S+)", text, re.M)[1]
            ),
            "execute_seconds": float(
                re.search(rf"^EXECUTE {phase} (\S+)", text, re.M)[1]
            ),
            "steps": result["steps"],
            "releases": releases.tolist(),
            "backtracks": np.sum(
                [event["backtracks"] for event in updates], axis=0
            ).tolist(),
            "max_residual": float(np.max(path_values[:, len(updates)])),
            "events": updates,
        }
    assert set(phases) == {"checkpoint", "restart"}, "missing completed phase"
    return {
        "revision": lines[0].split()[2],
        "log": str(path),
        "summary": summary[-1],
        "exit_status": int(lines[-1].split("=", 1)[1]),
        "phases": phases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    receipts = {
        arm: read_arm(args.run / f"restart-{arm}.log") for arm in ("base", "head")
    }
    (args.run / "restart-receipt.json").write_text(json.dumps(receipts, indent=2))
    for arm, receipt in receipts.items():
        print("ARM", arm, receipt["revision"], receipt["summary"])
        for phase, values in receipt["phases"].items():
            print(
                "PHASE",
                arm,
                phase,
                json.dumps({k: v for k, v in values.items() if k != "events"}),
            )
    print("HISTORY_EVENT_CHECK=passed")


if __name__ == "__main__":
    main()
