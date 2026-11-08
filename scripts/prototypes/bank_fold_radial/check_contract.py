"""Exercise radius actuation and the shared terminal acceptance loop."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.prototypes.bank_fold_radial import measure


def main():
    source = Path(measure.__file__)

    def evaluate(state, field, drive):
        return np.maximum(0.0, 6.0 - state[0][:, 0]), np.zeros(1), np.ones(1, bool)

    def step(state, field, drive):
        return (np.ones((1, 1)),), np.zeros(1), True, np.ones(1, bool)

    terminal = measure.correct_pinned(
        step,
        evaluate,
        (np.zeros((1, 1)),),
        np.zeros(1),
        1.001,
        1e-10,
        6,
        lambda _: None,
    )
    assert terminal.accepted and terminal.steps == 6 and terminal.reads == 55
    assert terminal.residual[0] == 0.0
    points = np.array([[2.0, 3.0], [4.0, -1.0]])
    jets = np.asarray(measure.uniform_vertical_field_jet(points))
    assert np.allclose(jets[:, 1] / (2 * np.pi * points[:, 0]), 1.0)
    assert np.all(jets[:, [2, 4, 5]] == 0)
    tree = ast.parse(source.read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("centroid_row", "linearized_system", "pinned_step")
    ]
    current_residual = 0.0
    scope = dict(
        centroid=lambda state, centres: (
            np.sum(state[0] * centres + state[1]) / np.sum(state[0])
        ),
        jnp=np,
        ROW_SCALE=measure.ROW_SCALE,
        SOURCE_SCALE=measure.SOURCE_SCALE,
        shifted=lambda _m, f: (SimpleNamespace(field=f), None),
        streams=SimpleNamespace(scaled_profile=lambda *_: None),
        profile_fixture=SimpleNamespace(gradient=None),
        support_read=SimpleNamespace(
            read=lambda *_: SimpleNamespace(valid=True),
            step_residual=lambda op, *_: SimpleNamespace(
                residual=(
                    np.array([current_residual]),
                    np.array([op.field]),
                    np.zeros(1),
                ),
                booked=None,
            ),
            step_jacobian=lambda *_, **__: np.eye(3),
            _equilibrate=lambda *_: (np.eye(3), np.ones(3), np.ones(3)),
        ),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            str(source),
            "exec",
        ),
        scope,
    )
    state = (np.array([1.0]), np.array([0.1]), np.array([0.3]))
    machine = SimpleNamespace(
        profile=None,
        convention=None,
        policy=None,
        quadrature=None,
        operands=SimpleNamespace(
            geometry=SimpleNamespace(centre=np.array([[2.0, 0.0]]))
        ),
    )
    updates, field, valid, present = scope["pinned_step"](
        machine, state, None, 1.0, 0.0, 2.0, True
    )
    after = tuple(a + b for a, b in zip(state, updates, strict=True))
    assert valid and present
    assert np.isclose((2.0 * after[0][0] + after[1][0]) / after[0][0], 2.0)
    assert np.isclose(measure.centroid(state, np.array([2.0])), 2.1)
    assert np.isclose(field, 0.1)
    assert np.isclose(after[2][0], 0.3)
    current_residual = -0.05
    coupled, strength, _, _ = scope["pinned_step"](
        machine, state, None, 1.0, 0.0, 2.1, True
    )
    after = tuple(a + b for a, b in zip(state, coupled, strict=True))
    assert np.isclose(after[0][0], 1.05)
    assert np.isclose((2.0 * after[0][0] + after[1][0]) / after[0][0], 2.1)
    assert np.isclose(strength, -0.005)
    # A disconnected actuator must not be accepted as an active pin.
    with np.errstate(divide="ignore", invalid="ignore"):
        _, _, _, absent = scope["pinned_step"](
            machine, state, None, 1.0, 0.0, 2.0, False
        )
    assert not absent
    print(
        "CONTRACT terminal_update=1 shared_acceptance=1 radius_row=1 "
        "vertical_field=1 physical_centroid=1 current_block_derivative=1 "
        "disconnected_pin_refused=1",
        flush=True,
    )


if __name__ == "__main__":
    main()
