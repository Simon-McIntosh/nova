"""Exercise the terminal pinned update and the paired spectrum receipt guard."""

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def sixth_update_is_checked(source: Path) -> bool:
    """Run the actual continuation loop with a root reached on update six."""
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "pinned_measure"
    )
    outer = next(
        node
        for node in function.body
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
    )
    loop = next(
        node
        for node in outer.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "iteration"
    )
    verdict = next(
        node
        for node in outer.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "accepted"
            for target in node.targets
        )
    )
    program = ast.fix_missing_locations(
        ast.Module(body=[loop, verdict], type_ignores=[])
    )

    def evaluate(state, field, drive):
        del field, drive
        return np.array([max(0.0, 6.0 - state[0][0, 0])]), np.zeros(1), np.ones(1, bool)

    def step(state, field, drive):
        del state, field, drive
        return (np.ones((1, 1)),), np.zeros(1), np.ones(1, bool)

    scope = dict(
        np=np,
        jnp=np,
        evaluate=evaluate,
        step=step,
        trial_state=(np.zeros((1, 1)),),
        trial_field=np.zeros(1),
        requested=1.001,
        tolerance=1e-10,
        reads=0,
        updates=0,
    )
    exec(compile(program, str(source), "exec"), scope)
    return bool(scope["accepted"] and scope["norms"][0] == 0 and scope["updates"] == 6)


def centroid_row_is_live(source: Path) -> bool:
    """Exercise the actual bordered update on a one-cell synthetic state."""
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "pinned_step"
    )
    program = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))

    def shifted(_machine, field):
        return SimpleNamespace(field=field), None

    def residual(operator, *_args):
        return SimpleNamespace(
            residual=(np.zeros(1), np.zeros(1), np.array([operator.field])),
            booked=None,
        )

    reader = SimpleNamespace(
        read=lambda *_args: SimpleNamespace(valid=True),
        step_residual=residual,
        step_jacobian=lambda *_args, **_kwargs: np.eye(3),
        _equilibrate=lambda *_args: (np.eye(3), np.ones(3), np.ones(3)),
    )
    scope = dict(
        jnp=np,
        shifted=shifted,
        streams=SimpleNamespace(scaled_profile=lambda *_args: None),
        support_read=reader,
        profile_fixture=SimpleNamespace(gradient=None),
    )
    exec(compile(program, str(source), "exec"), scope)
    state = (np.array([1.0]), np.array([0.0]), np.array([0.1]))
    machine = SimpleNamespace(
        profile=None, convention=None, policy=None, quadrature=None
    )
    updates, field_update, valid = scope["pinned_step"](
        machine, state, None, 1.0, 0.0, 0.0
    )
    after = tuple(part + delta for part, delta in zip(state, updates, strict=True))
    return bool(
        valid
        and np.isclose(after[2][0] / after[0][0], 0.0, atol=1e-12)
        and np.isclose(field_update, 0.1, atol=1e-12)
    )


def paired_spectra_are_required(compare: Path, scratch: Path) -> bool:
    """A missing conditioning arm must make an otherwise matching pair red."""
    base = dict(
        null=dict(
            drive=1.0,
            singular=1e-6,
            cosine_vertical=0.1,
            cosine_radial=0.9,
            cosine_state_vertical=0.1,
            cosine_state_radial=0.9,
            centroid=0.0,
        ),
        unpinned=dict(drive=1.0, certified=False, reads=1, updates=1),
        pinned=dict(drive=1.0, certified=False, reads=1, updates=1),
    )
    left, right = scratch / "left.json", scratch / "right.json"
    left.write_text(json.dumps(base))
    right.write_text(json.dumps(base))
    red = subprocess.run(
        [sys.executable, str(compare), str(left), str(right)],
        capture_output=True,
        text=True,
        check=False,
    )
    row = dict(drive=1.0, smallest=1e-6, original=1e-6, bordered=1e-8, valid=True)
    base["condition"] = dict(unpinned=[row, row], pinned=[row, row])
    left.write_text(json.dumps(base))
    right.write_text(json.dumps(base))
    green = subprocess.run(
        [sys.executable, str(compare), str(left), str(right)],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        red.returncode != 0 and green.returncode == 0 and "condition=1" in green.stdout
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tree", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--receipts", type=Path)
    args = parser.parse_args()
    folder = args.tree / "scripts/prototypes/bank_fold_vertical"
    args.scratch.mkdir(parents=True, exist_ok=True)
    results = dict(
        terminal_update=sixth_update_is_checked(folder / "measure.py"),
        centroid_row=centroid_row_is_live(folder / "measure.py"),
        paired_spectra=paired_spectra_are_required(folder / "compare.py", args.scratch),
        shared_condition=(
            "condition_rows(" in (folder / "measure.py").read_text()
            and not (folder / "run_condition.sh").exists()
        ),
    )
    if args.receipts is not None:
        receipts = [
            json.loads((args.receipts / name).read_text())
            for name in ("repair-first.json", "repair-second.json")
        ]
        results["durable_receipts"] = all(
            len(item["condition"][arm]) >= 2
            for item in receipts
            for arm in ("unpinned", "pinned")
        )
    print("CONTRACT " + json.dumps(results, sort_keys=True), flush=True)
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
