"""Count the traced and lowered program size of a compile-ratio row.

Compile seconds separate two revisions only on a quiet host. The jaxpr and
StableHLO counts of the same program do not depend on the host at all, so they
settle a row whose timing arm is dominated by machine noise. Every count is
taken in its own process, from the same function the compile-ratio row compiles.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = "cb42f656f76a7d7e56da8a53d4754c22d131bdc0"
BASE_FILES = ("gradedresidual.py", "polygonanalytic.py")
PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
ROOT = Path(__file__).resolve().parents[3]
TESTS = ROOT / "tests" / "biot"
ROWS = (
    ("test_vertex_edge_tangent_recurrences", "regularised"),
    ("test_edge_term_tangents", "second_residual"),
)

CHILD = r"""
import json, sys
import jax
jax.config.update("jax_enable_compilation_cache", False)
from nova.jax.config import configure_dtypes
configure_dtypes()
assert jax.config.jax_enable_x64 is True
sys.path.insert(0, sys.argv[1])
module = __import__(sys.argv[2])
name, arm = sys.argv[3], sys.argv[4]
case = module.CASES[name]
arguments = (case[-2], case[-1])
function = module.compile_arm(name, arm)

def count(jaxpr):
    total = 0
    for equation in jaxpr.eqns:
        total += 1
        for value in equation.params.values():
            for sub in value if isinstance(value, (list, tuple)) else [value]:
                inner = getattr(sub, "jaxpr", None)
                if inner is not None:
                    total += count(inner)
                elif hasattr(sub, "eqns"):
                    total += count(sub)
    return total

equations = count(jax.make_jaxpr(function)(*arguments).jaxpr)
text = jax.jit(function).lower(*arguments).as_text()
ops = sum(1 for line in text.splitlines() if "stablehlo." in line and "=" in line)
print(json.dumps({"equations": equations, "stablehlo_ops": ops,
                  "stablehlo_lines": len(text.splitlines()),
                  "module_file": module.__file__}))
"""


def overlay(directory: Path) -> Path:
    package = directory / "nova" / "biot"
    package.mkdir(parents=True)
    for name in BASE_FILES:
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "%s:nova/biot/%s" % (BASE, name)]
        )
        (package / name).write_bytes(source)
    (directory / "sitecustomize.py").write_text(
        "import nova\n"
        "nova.__path__.insert(0, %r)\n" % str(directory / "nova")
        + "import nova.biot\n"
        + "nova.biot.__path__.insert(0, %r)\n" % str(package),
        encoding="utf-8",
    )
    return directory


def count(path, module, name, arm):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(path or ""), str(ROOT)) if p)
    result = subprocess.run(
        [PYTHON, "-c", CHILD, str(TESTS), module, name, arm],
        cwd=str(ROOT), env=env, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return {"exit": result.returncode,
                "stderr": result.stderr.strip().splitlines()[-4:]}
    row = json.loads(result.stdout.strip().splitlines()[-1])
    row["exit"] = 0
    return row


def main():
    out = {}
    with tempfile.TemporaryDirectory(prefix="graph-count-XXXXXX") as directory:
        base_overlay = overlay(Path(directory))
        for module, name in ROWS:
            for arm in ("primal", "tangent"):
                key = "%s/%s" % (name, arm)
                out.setdefault(key, {})
                out[key]["base"] = count(base_overlay, module, name, arm)
                out[key]["head"] = count(None, module, name, arm)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
