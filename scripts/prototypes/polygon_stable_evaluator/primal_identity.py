"""Run the retained primal identity instrument against the current evaluator."""

import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


def main():
    source_revision = "ff92ac8f63e3088b4950104a0d146f37dcdc1842"
    with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as scratch:
        paths = {}
        for name in ("pole_algebra", "primal_identity"):
            path = Path(scratch) / (name + ".py")
            path.write_bytes(
                subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{source_revision}:scripts/prototypes/polygon_tangent_repair/{name}.py",
                    ]
                )
            )
            paths[name] = path
        name = "scripts.prototypes.polygon_tangent_repair.pole_algebra"
        spec = importlib.util.spec_from_file_location(name, paths["pole_algebra"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print("IDENTITY_INSTRUMENT_REVISION", source_revision, flush=True)
        runpy.run_path(str(paths["primal_identity"]), run_name="__main__")


if __name__ == "__main__":
    main()
