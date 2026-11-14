"""Prove constant bit identity and exercise the full-shell affine operator guard."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot import tiledassembly as tiled  # noqa: E402
from scripts.iron_linear_magnetisation.controls import require_clean_script, shell  # noqa: E402
from scripts.iron_linear_magnetisation.kernel_cost import load_owner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--mode", choices=["check", "negative"], default="check")
    args = parser.parse_args()
    blob = require_clean_script(__file__, args.blob, args.revision)
    assert jax.config.jax_enable_x64
    jax.config.update("jax_enable_compilation_cache", False)
    if args.mode == "negative":
        import pytest

        source = Path(tiled.__file__).read_text()
        marker = "    x, y = jnp.asarray(x.reshape(-1)), jnp.asarray(y.reshape(-1))"
        if source.count(marker) != 1:
            raise SystemExit("quadrature mutation did not identify one rule")
        source = source.replace(
            marker,
            marker + "\n    if linear:\n"
            "        weight = weight.at[weight.size // 2].multiply(1.1)",
        )
        path = Path(os.environ["TMPDIR"]) / "perturbed_quadrature_owner.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(
            "perturbed_quadrature_owner", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        iron.linear_galerkin_cell_interaction = module.linear_galerkin_cell_interaction
        os.environ["NOVA_LINEAR_OPERATOR_REFERENCE"] = str(args.reference)
        raise SystemExit(
            pytest.main(
                [
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "tests/test_magnetisation.py::test_linear_shell_operator_pin",
                    "-s",
                ]
            )
        )
    cells = iron._mesh(shell(), 0.04)
    baseline = load_owner(args.base_revision, "constant_reference_owner")
    before = baseline.galerkin_cell_interaction(cells.polygons, plan=iron._PLAN)
    after = tiled.galerkin_cell_interaction(cells.polygons, plan=iron._PLAN)
    row = dict(
        revision=args.revision,
        script_blob=blob,
        base_revision=args.base_revision,
        device=jax.devices()[0].device_kind,
        x64=True,
        cells=len(cells.polygons),
        completed=False,
        constant_bit_identical=bool(np.array_equal(before, after)),
        constant_before_sha256=hashlib.sha256(before.tobytes()).hexdigest(),
        constant_after_sha256=hashlib.sha256(after.tobytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    if not row["constant_bit_identical"]:
        raise SystemExit("constant basis changed bits")
    log = args.output.with_suffix(".log")
    command = [sys.executable, __file__, *sys.argv[1:], "--mode", "negative"]
    with log.open("w") as stream:
        stream.write(f"# revision: {args.revision}\n# cwd: {Path.cwd().resolve()}\n")
        stream.write(f"# module: {iron.__file__}\n# command: {shlex.join(command)}\n")
        stream.flush()
        done = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"EXIT={done.returncode}\n")
    text = log.read_text()
    marker = re.search(r"40 mm operator metrics (\{[^\n]+\})", text)
    failures = re.findall(r"^FAILED (\S+)", text, re.MULTILINE)
    row.update(
        negative_exit_status=done.returncode,
        negative_failure_ids=failures,
        negative_control_log=str(log),
        negative_metrics=json.loads(marker[1]) if marker else None,
    )
    row["completed"] = done.returncode != 0 and bool(failures) and marker is not None
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    if not row["completed"]:
        raise SystemExit("full-shell quadrature mutation did not fail the operator pin")


if __name__ == "__main__":
    main()
