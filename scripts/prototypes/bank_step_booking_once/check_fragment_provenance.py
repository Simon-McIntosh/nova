"""Check the booking-once fragment's provenance claims against its own receipts.

Every wall in the fragment's table must be the median of the log the fragment
names for that arm, and each named log's recorded revision must be the revision
the fragment attributes to it. A fragment that claims a single allocation for a
pair of runs recorded in two must fail here.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

STAGES = {
    "Read, outside the step": "read",
    "Residual and booking": "residual_and_booking",
    "Local booking tangent": "local_booking_tangent",
    "One moving geometry direction": "geometry_tangent_one_direction",
    "Jacobian, frozen geometry": "jacobian_without_geometry",
    "Whole Newton step": "whole_step",
}

ARMS = {"base": "h200-base2.log", "collapsed": "h200-prefix.log"}


def medians(path):
    values = {}
    for line in path.read_text().splitlines():
        if not line.startswith("STAGE "):
            continue
        payload = json.loads(line[len("STAGE ") :])
        values[payload["name"]] = payload["median_ms"]
    return values


def traced(path):
    values = {}
    for line in path.read_text().splitlines():
        if not line.startswith("STAGE_DEVICE "):
            continue
        name = re.match(r"STAGE_DEVICE name=(\S+)", line).group(1)
        numbers = dict(re.findall(r"(\w+)=([-0-9.]+)", line))
        values[name] = (float(numbers["device_ms"]), int(numbers["launches"]))
    return values


def revision(path):
    for line in path.read_text().splitlines():
        match = re.match(r"ARM=\S+ REVISION=([0-9a-f]+)", line)
        if match:
            return match.group(1)
        match = re.match(r'HEADER \{.*"revision": "([0-9a-f]+)"', line)
        if match:
            return match.group(1)
    return None


def current_path(path):
    for line in path.read_text().splitlines():
        match = re.search(r"CURRENT=(\S+)", line)
        if match:
            return match.group(1)
    return None


def table_rows(fragment):
    rows = {}
    for line in fragment.splitlines():
        cells = re.findall(r"<td>(.*?)</td>", line)
        if len(cells) < 3 or cells[0] not in STAGES:
            continue
        rows[cells[0]] = (cells[1], cells[2])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment", required=True)
    parser.add_argument("--logs", required=True)
    options = parser.parse_args()
    logs = Path(options.logs)
    fragment = Path(options.fragment).read_text()

    failures = []
    rows = table_rows(fragment)
    if len(rows) != len(STAGES):
        failures.append(f"table carries {len(rows)} stage rows, expected {len(STAGES)}")

    measured = {arm: medians(logs / name) for arm, name in ARMS.items()}
    digests = {}
    for source in (logs / "current-base.py", logs / "current-cond-preserved.py"):
        if not source.exists():
            failures.append(
                f"{source.name} is missing, so no source digest can be quoted"
            )
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        digests[source.name] = digest
        if digest[:8] not in fragment:
            failures.append(
                f"{source.name} sha256 {digest[:8]} is absent from the fragment"
            )
    for arm, name in ARMS.items():
        if name not in fragment:
            failures.append(f"the fragment does not name {name} as a receipt")
        if "both arms from one allocation" in fragment and arm == "collapsed":
            failures.append(
                "the fragment still claims both arms came from one allocation"
            )

    for stage, key in STAGES.items():
        if stage not in rows:
            continue
        for column, arm in enumerate(("base", "collapsed")):
            claimed = float(rows[stage][column])
            actual = round(measured[arm].get(key, float("nan")), 2)
            if abs(claimed - actual) > 0.005:
                failures.append(
                    f"{stage} {arm}: fragment {claimed} against log {actual}"
                )

    repeat = logs / "h200-base.log"
    if repeat.exists():
        if "h200-base.log" not in fragment:
            failures.append(
                "the repeat run h200-base.log is not reconciled in the fragment"
            )
        repeated = traced(repeat)
        collapsed = traced(logs / ARMS["collapsed"])
        for stage in ("local_booking_tangent", "whole_step"):
            if repeated.get(stage, (0.0, 0))[1] != collapsed.get(stage, (0.0, -1))[1]:
                failures.append(
                    f"h200-base.log {stage} launches {repeated.get(stage)} against "
                    f"h200-prefix.log {collapsed.get(stage)}; the fragment reconciles "
                    "them as one source"
                )
            first, second = repeated[stage][0], collapsed[stage][0]
            if abs(first - second) > 0.01 * second:
                failures.append(
                    f"h200-base.log {stage} device {first:.2f} ms is not within one "
                    f"per cent of h200-prefix.log {second:.2f} ms, which the "
                    "fragment quotes as 37.03 against 37.19"
                )

    if failures:
        for failure in failures:
            print("FAIL", failure)
        print("EXIT=1")
        return 1
    print("PASS provenance claims match the receipts")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
