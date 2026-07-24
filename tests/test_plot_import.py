"""Pin the matplotlib-free core import graph.

Matplotlib, seaborn and vedo are optional extras. Importing nova and its
frame / biot / graphics modules must never pull matplotlib into the
interpreter; the vtk plotting modules must import without vedo. A subprocess
with those top-level packages blocked proves it independently of whatever the
test environment happens to have installed.
"""

import subprocess
import sys
import textwrap


CORE_MODULES = [
    "nova",
    "nova.frame.coilset",
    "nova.frame.framespace",
    "nova.frame.vtkgeo",
    "nova.frame.vtkplot",
    "nova.frame.polyplot",
    "nova.graphics.plot",
    "nova.biot.biotframe",
]


def _run_with_blocked(blocked, modules):
    """Import modules in a subprocess with top-level packages blocked."""
    script = textwrap.dedent(
        """
        import sys
        blocked = set({blocked!r})
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise ModuleNotFoundError(f"blocked: {{name}}")
                return None
        sys.meta_path.insert(0, Blocker())
        for name in {modules!r}:
            __import__(name)
        leaked = sorted(
            {{k.split(".")[0] for k in sys.modules if k.split(".")[0] in blocked}}
        )
        assert not leaked, f"blocked packages leaked into sys.modules: {{leaked}}"
        """
    ).format(blocked=list(blocked), modules=modules)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )


def test_core_import_graph_is_matplotlib_free():
    """No core module imports matplotlib, seaborn, vedo or pyvista."""
    result = _run_with_blocked(
        ["matplotlib", "seaborn", "vedo", "pyvista", "descartes"], CORE_MODULES
    )
    assert result.returncode == 0, result.stderr
