"""The 2D coilset path imports under nova's base dependencies alone.

A downstream project that installs only nova's base dependencies must import
``nova.frame.coilset``, ``nova.imas.machine``, ``nova.circuit.conductor``,
``nova.circuit.linkage`` and ``nova.circuit.linkage`` and build a two-coil
coilset with its linkage matrix, without the 3D visualisation stack (``vedo``,
``pyvista``, ``meshio``, ``trimesh``, ``vtk``) being importable.  The check runs
in a subprocess whose import machinery refuses those names, so a stray eager
import anywhere on the path surfaces as a failure rather than a warning.

``test_vtkgeo_features_match_trishell`` is the reverse guard: with the stack
present, the frame-declared vtk column names must equal the ones ``TriShell``
produces values for, so the stack-free schema cannot drift from the 3D one.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BLOCKED_MODULES = ("meshio", "vedo", "pyvista", "vtk", "trimesh")

_SUBPROCESS = textwrap.dedent(
    """
    import importlib
    import os
    import sys

    BLOCKED = {blocked!r}
    requested = []

    class _Block3D:
        def find_spec(self, name, path=None, target=None):
            root = name.split(".")[0]
            if root in BLOCKED:
                requested.append(root)
                raise ModuleNotFoundError(
                    f"No module named {{root!r}}", name=root
                )
            return None

    sys.meta_path.insert(0, _Block3D())

    for name in (
        "nova.frame.coilset",
        "nova.imas.machine",
        "nova.circuit.conductor",
        "nova.circuit.linkage",
        "nova.biot.greens",
    ):
        importlib.import_module(name)

    import numpy as np  # noqa: E402
    from nova.frame.coilset import CoilSet
    from nova.circuit.conductor import ConductorSet
    from nova.circuit.linkage import circuit_linkage_matrix

    coilset = CoilSet(dcoil=-1)
    coilset.coil.insert([4.0, 6.0], [1.0, 1.0], 0.1, 0.1, label="PF")
    frame = coilset.frame
    radius = frame["x"].to_numpy()
    height = frame["z"].to_numpy()
    conductors = ConductorSet(
        circuit=np.arange(len(radius), dtype=np.int64),
        r=radius,
        z=height,
        dr=frame["dx"].to_numpy(),
        dz=frame["dz"].to_numpy(),
        current_share=np.ones(len(radius)),
    )
    lmat = np.asarray(circuit_linkage_matrix(conductors, list(range(len(radius)))))

    loaded = sorted(name for name in BLOCKED if name in sys.modules)
    assert lmat.shape == (2, 2), lmat.shape
    assert not loaded, f"3D modules loaded on the base path: {{loaded}}"
    print("base-install coilset OK", lmat.shape, "requested", sorted(set(requested)))
    """.format(blocked=BLOCKED_MODULES)
)


def test_2d_coilset_builds_without_the_3d_stack():
    """The base-dependency path imports and links without the 3D stack."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPOSITORY_ROOT)
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("vedo") is None
    or importlib.util.find_spec("pyvista") is None,
    reason="3D stack unavailable; the drift guard needs both feature lists.",
)
def test_vtkgeo_features_match_trishell():
    """The stack-free frame schema equals the 3D geometry feature order."""
    from nova.frame.vtkgeo import TRISHELL_FEATURES
    from nova.geometry.volume import TriShell

    assert list(TRISHELL_FEATURES) == list(TriShell.features)
