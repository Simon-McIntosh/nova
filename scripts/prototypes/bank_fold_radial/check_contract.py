"""Exercise radius actuation and the shared terminal acceptance loop."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.prototypes.bank_fold_radial import measure
from scripts.prototypes.bank_fold_vertical.check_contract import (
    both_arms_share_acceptance,
    sixth_update_is_checked,
)


def main():
    source = Path(measure.__file__)
    assert sixth_update_is_checked(source)
    assert both_arms_share_acceptance(source)
    points = np.array([[2.0, 3.0], [4.0, -1.0]])
    jets = np.asarray(measure.uniform_vertical_field_jet(points))
    assert np.allclose(jets[:, 1] / (2 * np.pi * points[:, 0]), 1.0)
    assert np.all(jets[:, [2, 4, 5]] == 0)
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "pinned_step"
    )
    scope = dict(
        jnp=np,
        ROW_SCALE=measure.ROW_SCALE,
        SOURCE_SCALE=measure.SOURCE_SCALE,
        shifted=lambda _m, f: (SimpleNamespace(field=f), None),
        streams=SimpleNamespace(scaled_profile=lambda *_: None),
        profile_fixture=SimpleNamespace(gradient=None),
        support_read=SimpleNamespace(
            read=lambda *_: SimpleNamespace(valid=True),
            step_residual=lambda op, *_: SimpleNamespace(
                residual=(np.zeros(1), np.array([op.field]), np.zeros(1)), booked=None
            ),
            step_jacobian=lambda *_, **__: np.eye(3),
            _equilibrate=lambda *_: (np.eye(3), np.ones(3), np.ones(3)),
        ),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(source),
            "exec",
        ),
        scope,
    )
    state = (np.array([1.0]), np.array([2.1]), np.array([0.3]))
    machine = SimpleNamespace(
        profile=None, convention=None, policy=None, quadrature=None
    )
    updates, field, valid, present = scope["pinned_step"](
        machine, state, None, 1.0, 0.0, 2.0, True
    )
    after = tuple(a + b for a, b in zip(state, updates, strict=True))
    assert valid and present
    assert np.isclose(after[1][0] / after[0][0], 2.0)
    assert np.isclose(field, 0.1)
    assert np.isclose(after[2][0], 0.3)
    # A disconnected actuator must not be accepted as an active pin.
    with np.errstate(divide="ignore", invalid="ignore"):
        _, _, _, absent = scope["pinned_step"](
            machine, state, None, 1.0, 0.0, 2.0, False
        )
    assert not absent
    print(
        "CONTRACT terminal_update=1 shared_acceptance=1 radius_row=1 "
        "vertical_field=1 disconnected_pin_refused=1",
        flush=True,
    )


if __name__ == "__main__":
    main()
