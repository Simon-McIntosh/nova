"""Regression checks for measured-route controls and receipt-backed evidence."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.prototypes.bank_hessian_boundary import document, inspect, report


@pytest.fixture
def measured(monkeypatch, tmp_path):
    root = tmp_path / ".cache/nova-hessian-boundary"
    (root / "nodes").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(report, "ROOT", root)
    truth = np.array([[2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.longdouble)
    reference = dict(
        indices=np.arange(4),
        plasma=np.stack([truth * s for s in (1, 2, 3)]),
        coils=np.zeros((3, 4, 2)),
        plasma_bound=np.full((3, 4, 2), 1e-12),
        coil_bound=np.zeros((3, 4, 2)),
    )
    archives, states = [], []
    for state, label in enumerate(inspect.STATES):
        jets = {}
        for key, scale in (("actual", 1.25), ("expected", 0.8)):
            jets[key] = np.zeros((4, 6), dtype=np.longdouble)
            jets[key][:, 4:6] = reference["plasma"][state] * scale
        archives.append(dict(**jets, moments=np.ones((1, 3))))
        states.append(
            dict(
                state=label,
                interior=dict(
                    bank=[0.25, 0.25], direct=[0.2, 0.2], reference_bound=[1e-12, 1e-12]
                ),
                negative_control=dict(threshold=1e-8),
            )
        )
    (root / "split.json").write_text(json.dumps(dict(states=states)))
    (root / "retained-inputs.json").write_text("[]")
    for node in (2, 3, -1):
        values = np.ones((5, 2, 3, 2))
        values[:, 1, :, 1] = 2
        forcing = np.ones((5, 2, 3))
        forcing[:, 1] = 2
        record = dict(
            node=node,
            corner=node == 2,
            full_field=node == 3,
            source_cells=[0],
            values=values.tolist(),
            error=np.full_like(values, 1e-12).tolist(),
            trace=forcing.tolist(),
            forcing=forcing.tolist(),
            stencils=[dict(contained=True, step_m=1e-4, clearance_m=1)],
        )
        (root / "nodes" / f"node-{node}.json").write_text(json.dumps(record))
    monkeypatch.setattr(
        report,
        "load",
        lambda: (
            (),
            archives,
            reference,
            None,
            None,
            np.array([True, True, False, False]),
        ),
    )
    monkeypatch.setattr(
        report,
        "coefficients",
        lambda *args: dict(relative=[[0, 0]], nodes=[], nonzero_per_state=[1, 1, 1]),
    )
    monkeypatch.setattr(
        report.subprocess, "check_output", lambda *args, **kwargs: "fixture-revision\n"
    )
    monkeypatch.setattr(sys, "argv", ["report"])
    return root, archives


def test_mutation_changes_measured_route_errors(measured, monkeypatch):
    root, _ = measured
    report.main()
    baseline = json.loads((root / "receipt.json").read_text())
    monkeypatch.setattr(sys, "argv", ["report", "--mutate"])
    with pytest.raises(AssertionError, match="psi_ZZ"):
        report.main()
    mutated = json.loads((root / "mutation.json").read_text())
    for before, after in zip(baseline["states"], mutated["states"], strict=True):
        for route in ("bank", "direct"):
            np.testing.assert_allclose(
                after["interior"][route][0], before["interior"][route][0], atol=1e-12
            )
            assert (
                abs(after["interior"][route][1] - before["interior"][route][1]) > 1e-7
            )
        assert not after["control"]["accepted"]


def test_zeroed_measured_curvature_is_refused(measured):
    _, archives = measured
    for archive in archives:
        archive["actual"][:, 5] = 0
        archive["expected"][:, 5] = 0
    with pytest.raises(AssertionError, match="psi_ZZ"):
        report.main()


def test_document_check_refuses_changed_receipt(measured, monkeypatch, tmp_path):
    root, _ = measured
    report.main()
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["nodes"] = [dict(node=i) for i in range(92)]
    (root / "receipt.json").write_text(json.dumps(receipt))
    fragment = tmp_path / "fragment.html"
    fragment.write_text(
        '<section><div id="bank-hessian-boundary-limits">stale</div></section>'
    )
    monkeypatch.setattr(sys, "argv", ["document", str(fragment)])
    document.main()
    monkeypatch.setattr(sys, "argv", ["document", str(fragment), "--check"])
    document.main()
    receipt["states"][0]["interior"]["bank"][1] = 0.123456789
    (root / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(SystemExit) as refusal:
        document.main()
    assert refusal.value.code == 1
    monkeypatch.setattr(sys, "argv", ["document", str(fragment)])
    document.main()
    assert "0.123456789" in fragment.read_text()


def test_clearance_has_one_owner_and_split_calls_it(monkeypatch):
    from scripts.prototypes.bank_hessian_boundary import limits

    assert limits.clearance is inspect.clearance
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.longdouble)
    points = np.array([[1, 1], [3, 1], [0, 0], [3, 3]], dtype=np.longdouble)
    np.testing.assert_allclose(
        inspect.clearance(points, (square,)), [1, 1, 0, np.sqrt(2)]
    )
    assert inspect.clearance(points[0], (square,)) == 1
    calls = []
    shared = inspect.clearance
    monkeypatch.setattr(
        inspect, "clearance", lambda p, q: (calls.append(p.shape), shared(p, q))[1]
    )

    def load(path):
        if path.name == "geometry.npz":
            return dict(vertices=square[None], vertex_count=[4])
        if path.name == "qualified-reference.npz":
            return dict(indices=np.arange(4), steps=np.array([0.5]))
        return dict(points=points)

    monkeypatch.setattr(inspect.np, "load", load)
    *_, distance, interior = inspect.load()
    np.testing.assert_array_equal(interior, [True, True, False, True])
    assert len(calls) == 1


def test_gate_regenerates_and_checks_document(tmp_path):
    root = tmp_path / "repository"
    package = root / "scripts/prototypes/bank_hessian_boundary"
    package.mkdir(parents=True)
    for folder in (root / "scripts", root / "scripts/prototypes", package):
        (folder / "__init__.py").write_text("")
    for name in ("inspect", "limits"):
        (package / f"{name}.py").write_text("print('fixture measurement')\n")
    (package / "report.py").write_text(
        "import sys\nraise SystemExit(1 if '--mutate' in sys.argv else 0)\n"
    )
    (package / "document.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "p = Path(__file__).with_name('document-calls.txt')\n"
        "with p.open('a') as f:\n"
        "    f.write('check\\n' if '--check' in sys.argv else 'render\\n')\n"
    )
    (package / "test_measurement.py").write_text(
        "def test_fixture():\n    assert True\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text("#!/bin/sh\nprintf 'fixture-revision\\n'\n")
    git.chmod(0o755)
    import os

    environment = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    output = tmp_path / "logs"
    output.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(Path(__file__).with_name("gate.sh")),
            str(root),
            str(output),
            "fixture-revision",
        ],
        env=environment,
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (package / "document-calls.txt").read_text().splitlines() == [
        "render",
        "check",
    ]


def test_height_derivative_measures_known_curvature(monkeypatch, tmp_path):
    from scripts.prototypes.bank_hessian_boundary import limits

    monkeypatch.setattr(limits, "ROOT", tmp_path)
    (tmp_path / "nodes").mkdir()
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    monkeypatch.setattr(
        limits,
        "coupling",
        lambda polygons: dict(
            inverse_second_moment=np.eye(2)[None], centre=np.array([[1, 1]])
        ),
    )

    def gradient(points, packed, moments):
        r, z = points.T
        field = np.stack((2 * r + z, r + 6 * z), axis=-1)
        return field[None] * np.arange(1, 4)[:, None, None]

    monkeypatch.setattr(limits, "gradient", gradient)
    result = limits.evaluate(
        (
            0,
            9,
            [1, 1],
            (square,),
            np.zeros((3, 1, 3)),
            np.zeros((3, 2)),
            True,
            (square,),
            np.array([0]),
        )
    )
    expected = np.array([[1, 6], [2, 12], [3, 18]])
    np.testing.assert_allclose(
        result["values"], np.broadcast_to(expected, (5, 2, 3, 2)), rtol=1e-8
    )
    assert all(s["contained"] for s in result["stencils"])
