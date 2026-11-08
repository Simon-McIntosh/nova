"""Exercise the terminal pinned update and the paired spectrum receipt guard."""

import argparse
import ast
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def physical_centroid_translates(source: Path) -> bool:
    """Rigidly translated uniform-current cells move their centroid by dz."""
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "centroid"
    )
    program = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = dict(jnp=np)
    exec(compile(program, str(source), "exec"), scope)
    state = (np.ones(3), np.zeros(3), np.zeros(3))
    centres = np.array([-0.2, 0.1, 0.4])
    dz = 0.037

    def height(centres):
        args = (state, centres) if len(function.args.args) == 2 else (state,)
        return float(scope["centroid"](*args))

    before, after = height(centres), height(centres + dz)
    passed = np.isclose(before, centres.mean(), atol=1e-14) and np.isclose(
        after - before, dz, atol=1e-14
    )
    print(
        f"CENTROID_TRANSLATION before={before:.12g} after={after:.12g} "
        f"observed_dz={after - before:.12g} expected_dz={dz} passed={bool(passed)}",
        flush=True,
    )
    return bool(passed)


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
        return (np.ones((1, 1)),), np.zeros(1), np.ones(1, bool), np.ones(1, bool)

    scope = dict(
        np=np,
        jnp=np,
        accepted_read=lambda valid, residual, tolerance, *, height=None: bool(
            np.all(np.asarray(valid) & (np.asarray(residual) <= tolerance))
            and (height is None or np.all(np.abs(height) <= 1e-6))
        ),
        evaluate=evaluate,
        step=step,
        trial_state=(np.zeros((1, 1)),),
        trial_field=np.zeros(1),
        requested=1.001,
        tolerance=1e-10,
        reads=0,
        updates=0,
        pin_reported=True,
    )
    exec(compile(program, str(source), "exec"), scope)
    return bool(scope["accepted"] and scope["norms"][0] == 0 and scope["updates"] == 6)


def centroid_row_is_live(source: Path) -> bool:
    """Exercise the actual bordered update on a one-cell synthetic state."""
    tree = ast.parse(source.read_text())
    program = ast.fix_missing_locations(
        ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name in ("centroid", "centroid_gradient", "pinned_step")
            ],
            type_ignores=[],
        )
    )

    def shifted(_machine, field):
        return SimpleNamespace(field=field, geometry=_machine.operands.geometry), None

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
        profile=None,
        convention=None,
        policy=None,
        quadrature=None,
        operands=SimpleNamespace(
            geometry=SimpleNamespace(centre=np.array([[1.0, 0.25]]))
        ),
    )
    updates, field_update, valid, row_present = scope["pinned_step"](
        machine, state, None, 1.0, 0.0, 0.25, True
    )
    after = tuple(part + delta for part, delta in zip(state, updates, strict=True))
    return bool(
        valid
        and row_present
        and np.isclose(after[2][0] / after[0][0], 0.0, atol=1e-12)
        and np.isclose(field_update, 0.1, atol=1e-12)
    )


def centroid_row_matches_difference(source: Path) -> bool:
    """Differentiate height while unequal currents occupy distinct heights."""
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("centroid", "centroid_gradient")
    ]
    scope = dict(jnp=np)
    program = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(program, str(source), "exec"), scope)
    state = (np.array([1.0, 2.0]), np.zeros(2), np.array([0.03, -0.02]))
    centres = np.array([-0.2, 0.4])
    flat = np.concatenate(state)
    numerical = []
    for direction in np.eye(flat.size):
        plus = tuple(np.split(flat + 1e-6 * direction, 3))
        minus = tuple(np.split(flat - 1e-6 * direction, 3))
        numerical.append(
            (scope["centroid"](plus, centres) - scope["centroid"](minus, centres))
            / 2e-6
        )
    return bool(
        np.allclose(
            scope["centroid_gradient"](state, centres), numerical, rtol=1e-8, atol=1e-10
        )
    )


def product_row_probe_is_sensitive(source: Path) -> bool:
    """A known extra assembled row must change the printed product status."""
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "bank_pin_status"
    )
    program = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    machine = SimpleNamespace(
        operands=None,
        cache=None,
        convention=None,
        policy=None,
        profile=None,
        quadrature=None,
    )
    moments = tuple(np.ones((1, 1)) for _ in range(3))
    rows = []

    def residual(*_args):
        return SimpleNamespace(residual=tuple(np.zeros(1) for _ in rows))

    scope = dict(
        jnp=np,
        support_read=SimpleNamespace(read=lambda *_args: None, step_residual=residual),
        streams=SimpleNamespace(scaled_profile=lambda *_args: None),
        emit=lambda *_args, **_kwargs: None,
    )
    exec(compile(program, str(source), "exec"), scope)
    outputs = []
    for count in (3, 4):
        rows[:] = range(count)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            scope["bank_pin_status"](machine, moments, np.zeros((1, 1)))
        outputs.append(stream.getvalue().strip())
    return outputs == ["PIN_ACTIVE=absent", "PIN_ACTIVE=true"]


def both_arms_share_acceptance(source: Path) -> bool:
    tree = ast.parse(source.read_text())
    calls = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        calls[node.name] = sum(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "accepted_read"
            for child in ast.walk(node)
        )
    return calls.get("run_unpinned", 0) >= 1 and calls.get("pinned_measure", 0) >= 2


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
        physical_centroid=physical_centroid_translates(folder / "measure.py"),
        terminal_update=sixth_update_is_checked(folder / "measure.py"),
        centroid_row=centroid_row_is_live(folder / "measure.py"),
        centroid_derivative=centroid_row_matches_difference(folder / "measure.py"),
        product_row_probe=product_row_probe_is_sensitive(folder / "measure.py"),
        shared_acceptance=both_arms_share_acceptance(folder / "measure.py"),
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
