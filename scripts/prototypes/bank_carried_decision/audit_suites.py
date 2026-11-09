"""Audit completed per-file suite receipts against one product revision."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import nova

ROOT = Path(__file__).resolve().parents[3]
PRODUCT_PATHS = (
    "nova/equilibrium/support_read.py",
    "nova/equilibrium/topology.py",
    "tests/test_tracked_read.py",
)
FILES = (
    "test_tracked_read",
    "test_support_read",
    "test_bank_solve_driver",
    "test_bank_solve_cycle",
    "test_candidate_inspection_batched",
)


def receipt(path):
    text = path.read_text()
    lines = text.splitlines()
    assert lines and re.fullmatch(r"EXIT=\d+", lines[-1]), path
    summary = next(
        line.strip("= ")
        for line in reversed(lines)
        if re.search(r"\b\d+ (?:passed|failed|errors?)\b", line) and " in " in line
    )
    counts = {
        label: int(match.group(1))
        if (match := re.search(rf"\b(\d+) {label}\b", summary))
        else 0
        for label in ("passed", "failed", "skipped", "deselected")
    }
    failures = re.findall(r"^FAILED (\S+)", text, re.MULTILINE)
    revision = re.search(r"^# revision: (\S+)", text).group(1)
    return dict(
        log_path=str(path),
        revision=revision,
        summary=summary,
        exit_status=int(lines[-1].split("=")[1]),
        failure_ids=failures,
        completed=True,
        **counts,
    )


def product_hashes(revision):
    return {
        path: hashlib.sha256(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{revision}:{path}"]
            )
        ).hexdigest()
        for path in PRODUCT_PATHS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--product-head", required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        "HEADER",
        json.dumps(
            dict(
                revision=revision,
                tree=str(ROOT),
                cwd=str(Path.cwd().resolve()),
                nova=nova.__file__,
            )
        ),
        flush=True,
    )
    # The parser must find the known failing mutation before zero failures
    # in the real suite receipts can be accepted as an absence claim.
    control = receipt(args.run / "incidence/negative-control.log")
    assert control["exit_status"] != 0
    assert control["failed"] == len(control["failure_ids"]) == 3
    expected = product_hashes(args.product_head)
    checked = {args.product_head: expected}
    rows = []
    totals = {
        arm: dict(passed=0, failed=0, skipped=0, deselected=0)
        for arm in ("base", "HEAD")
    }
    for name in FILES:
        pair = {}
        for arm in ("base", "HEAD"):
            item = receipt(args.run / "guarded" / f"{arm}-{name}.log")
            assert item["exit_status"] == 0 and item["passed"] > 0, item
            assert item["failed"] == 0 and not item["failure_ids"], item
            if arm == "base":
                assert item["revision"] == args.base, item
            else:
                if item["revision"] not in checked:
                    checked[item["revision"]] = product_hashes(item["revision"])
                assert checked[item["revision"]] == expected, item
            for key in totals[arm]:
                totals[arm][key] += item[key]
            pair[arm] = item
        added = sorted(
            set(pair["HEAD"]["failure_ids"]) - set(pair["base"]["failure_ids"])
        )
        assert not added
        rows.append(dict(file=name + ".py", added_failure_ids=added, **pair))
    assert totals["base"]["passed"] == 35
    assert totals["HEAD"]["passed"] == 44
    result = dict(
        base=args.base,
        product_head=args.product_head,
        rows=rows,
        totals=totals,
        added_failure_count=0,
        parser_control=control,
        product_hashes=checked,
    )
    output = args.run / "paired-suite-audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "AUDIT",
        json.dumps(
            dict(
                path=str(output),
                totals=totals,
                added_failure_count=0,
                parser_control_failures=3,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
