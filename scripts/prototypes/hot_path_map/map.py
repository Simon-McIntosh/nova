"""Compose the per-module line map from per-route coverage JSON reports.

Every module under ``nova/`` receives one row: the executable lines each route
executed against that route's total, and the union of the two. A module the
coverage report does not carry is recorded ``not measured`` rather than zero.
"""

import argparse
import ast
import json
from pathlib import Path

# Modules the bank-solve device trace attributes kernels to. Its compiled loop
# is ``support_read.solve_batch`` plus the modules that entry imports.
TRACE_MODULES = (
    "nova/equilibrium/support_read.py",
    "nova/equilibrium/topology.py",
    "nova/equilibrium/current.py",
    "nova/equilibrium/clip_quadrature.py",
    "nova/equilibrium/fixed_points.py",
    "nova/equilibrium/solve.py",
    "nova/equilibrium/constraint.py",
)

# Bundled entry points the device trace attributes its compiled kernels to. The
# line map is checked at body granularity here, because a module can be reached
# by a different function while the traced loop itself never runs.
TRACE_ENTRIES = (
    ("nova/equilibrium/support_read.py", "read"),
    ("nova/equilibrium/support_read.py", "read_batch"),
    ("nova/equilibrium/support_read.py", "read_and_book"),
    ("nova/equilibrium/support_read.py", "bank_program"),
)


def body_lines(source, function):
    """Return the line span of ``function``'s body, excluding its definition."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name == function:
                start = min(item.lineno for item in node.body)
                return start, node.end_lineno
    return None


def relative(path, root):
    resolved = Path(path).resolve()
    try:
        return Path(resolved).relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def load(report, root):
    """Return module -> (executed line set, total executable lines)."""
    data = json.loads(Path(report).read_text())
    rows = {}
    for path, entry in data["files"].items():
        summary = entry["summary"]
        rows[relative(path, root)] = (
            set(entry.get("executed_lines", [])),
            summary["covered_lines"] + summary["missing_lines"],
        )
    return rows


def cell(rows, name):
    if name not in rows:
        return None
    executed, total = rows[name]
    return dict(executed=len(executed), total=total, lines=sorted(executed))


def percent(executed, total):
    return round(100.0 * executed / total, 2) if total else 0.0


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--inverse", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--negative-control", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    forwards = load(args.forward, root)
    inverses = load(args.inverse, root)

    modules = []
    for path in sorted((root / "nova").rglob("*.py")):
        name = path.relative_to(root).as_posix()
        forward = cell(forwards, name)
        inverse = cell(inverses, name)
        union = set()
        if forward:
            union.update(forward["lines"])
        if inverse:
            union.update(inverse["lines"])
        total = (forward or inverse or {}).get("total")
        row = dict(
            module=name,
            total=total,
            measured=forward is not None or inverse is not None,
            forward=None,
            inverse=None,
            union=dict(executed=len(union), lines=sorted(union)),
        )
        for key, value in (("forward", forward), ("inverse", inverse)):
            if value:
                row[key] = dict(
                    executed=value["executed"],
                    percent=percent(value["executed"], value["total"]),
                    lines=value["lines"],
                )
        row["union"]["percent"] = percent(len(union), total) if total else 0.0
        modules.append(row)

    control = next(
        (row for row in modules if row["module"] == args.negative_control), None
    )
    contradictions = []
    for name in TRACE_MODULES:
        row = next((item for item in modules if item["module"] == name), None)
        if row is None:
            contradictions.append(dict(module=name, reason="absent from nova tree"))
        elif row["forward"] is None:
            contradictions.append(
                dict(module=name, reason="not measured by the forward route")
            )
        elif row["forward"]["executed"] == 0:
            contradictions.append(
                dict(
                    module=name,
                    reason="device trace attributes kernels here; forward route read 0",
                )
            )

    entries = []
    for name, function in TRACE_ENTRIES:
        source = (root / name).read_text()
        span = body_lines(source, function)
        row = next((item for item in modules if item["module"] == name), None)
        union = set(row["union"]["lines"]) if row else set()
        if span is None:
            entries.append(
                dict(
                    module=name,
                    function=function,
                    executed=0,
                    body_lines=0,
                    reason=f"{function} not found in {name}",
                )
            )
            continue
        start, stop = span
        if row is None:
            entries.append(
                dict(
                    module=name,
                    function=function,
                    executed=0,
                    body_lines=0,
                    reason=f"{name} absent from nova tree",
                )
            )
            continue
        fired = sorted(number for number in union if start <= number <= stop)
        entries.append(
            dict(
                module=name,
                function=function,
                executed=len(fired),
                body_lines=stop - start + 1,
                reason=(
                    f"device trace attributes kernels to {function}, and no route "
                    f"executed a line of its body"
                    if not fired
                    else None
                ),
            )
        )
    contradictions.extend(item for item in entries if item["reason"])

    payload = dict(
        revision=args.revision,
        tree=args.tree,
        routes={
            "forward": (
                "nova.equilibrium.forward.ForwardProfile.solve on a stored MAST case"
            ),
            "inverse": (
                "tests/test_equilibrium_magnetics_fit.py"
                "::test_known_state_recovery[mast]"
            ),
        },
        negative_control={
            "module": args.negative_control,
            "measured": bool(control and control["measured"]),
            "forward": None if control is None else control["forward"],
            "inverse": None if control is None else control["inverse"],
        },
        device_trace_entries=entries,
        device_trace_contradictions=contradictions,
        modules=modules,
    )
    args.output_json.write_text(json.dumps(payload, indent=1))
    print(
        "MAP",
        json.dumps(
            dict(
                modules=len(modules),
                measured=sum(1 for row in modules if row["measured"]),
                not_measured=[row["module"] for row in modules if not row["measured"]],
                contradictions=[item["module"] for item in contradictions],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
