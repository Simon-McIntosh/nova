"""Measure affine magnetisation on the analytic high-permeability shell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from scripts.iron_linear_magnetisation.controls import (  # noqa: E402
    SHELL_MU,
    SHELL_PROBES,
    UNIFORM_SIZES,
    exact_shielding,
    loop,
    shell,
    require_clean_script,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--sizes", type=float, nargs="+", default=UNIFORM_SIZES)
    parser.add_argument("--basis", choices=("constant", "linear"), default="linear")
    parser.add_argument("--interaction-dir", type=Path)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    actual = require_clean_script(__file__, args.blob, args.revision)
    args.output.mkdir(parents=True, exist_ok=True)
    previous = None
    for size in args.sizes:
        row = dict(
            revision=args.revision,
            script_blob=actual,
            basis=args.basis,
            cell_size_m=size,
            lane=jax.default_backend(),
            devices=[device.device_kind for device in jax.devices()],
            module=str(Path(iron.__file__).resolve()),
            cwd=str(Path.cwd().resolve()),
            x64=bool(jax.config.jax_enable_x64),
            completed=False,
        )
        path = args.output / f"{args.basis}-{size:g}.json"

        def save():
            temporary = path.with_suffix(".json.partial")
            temporary.write_text(json.dumps(row, indent=2) + "\n")
            temporary.replace(path)

        save()
        started = time.monotonic()
        production = iron._galerkin_interaction

        def timed(*arguments, **keywords):
            cells = arguments[0]
            unknowns = len(cells.polygons) * (6 if args.basis == "linear" else 2)
            row.update(
                cells=len(cells.polygons),
                unknowns=unknowns,
                dense_factor_bytes=8 * unknowns**2,
            )
            save()
            if (
                jax.default_backend() == "gpu"
                and row["dense_factor_bytes"] > 13_653_508_096
            ):
                row["requires_cpu"] = True
                save()
                raise SystemExit(
                    "dense factor exceeds the device argument limit; rerun on CPU"
                )
            assembly_start = time.monotonic()
            if args.interaction_dir:
                receipt = json.loads(
                    (args.interaction_dir / f"assembly-{size:g}.json").read_text()
                )
                if not receipt["completed"] or receipt["cells"] != len(cells.polygons):
                    raise SystemExit(
                        "cached interaction does not name this completed mesh"
                    )
                result = np.load(receipt["operator_path"])
                import hashlib

                digest = hashlib.file_digest(
                    Path(receipt["operator_path"]).open("rb"), "sha256"
                ).hexdigest()
                if digest != receipt["operator_sha256"]:
                    raise SystemExit("cached interaction digest differs")
                row["interaction_load_seconds"] = time.monotonic() - assembly_start
                row["interaction_receipt"] = str(
                    args.interaction_dir / f"assembly-{size:g}.json"
                )
                row["assembly_lane"] = receipt["devices"]
                row["assembly_revision"] = receipt["revision"]
                row["assembly_seconds"] = receipt["assembly_seconds"]
                save()
                return result
            result = production(*arguments, **keywords)
            row["assembly_seconds"] = time.monotonic() - assembly_start
            save()
            return result

        iron._galerkin_interaction = timed
        try:
            solution = iron.solve_magnetisation(
                shell(),
                loop,
                [0, 10],
                [SHELL_MU, SHELL_MU],
                cell_size=size,
                basis=args.basis,
                tolerance=1e-10,
            )
        finally:
            iron._galerkin_interaction = production
        row["solve_seconds"] = time.monotonic() - started
        air = loop(SHELL_PROBES)
        total = air + solution.field(SHELL_PROBES)
        shielding = np.linalg.norm(air, axis=1) / np.linalg.norm(total, axis=1)
        errors = np.abs(shielding / exact_shielding() - 1)
        from scripts.iron_linear_magnetisation.reference import exterior_reference

        reference = np.linalg.norm(air, axis=1) / np.linalg.norm(
            exterior_reference(10)[1], axis=1
        )
        finite_errors = np.abs(shielding / reference - 1)
        row.update(
            finite_loop_shielding=reference.tolist(),
            finite_loop_relative_errors=finite_errors.tolist(),
            finite_loop_worst_error=float(finite_errors.max()),
            cells=solution.cell_count,
            unknowns=solution.cell_count * (6 if args.basis == "linear" else 2),
            probe_coordinates_m=SHELL_PROBES.tolist(),
            shielding=shielding.tolist(),
            shielding_exact=exact_shielding(),
            relative_errors=errors.tolist(),
            worst_error=float(errors.max()),
            converged=solution.converged,
            residual_history=solution.residual_history,
            iterations=solution.iterations,
            completed=True,
            total_seconds=time.monotonic() - started,
        )
        if previous is not None:
            row["per_probe_order"] = (
                np.log(np.asarray(previous["relative_errors"]) / errors)
                / np.log(previous["cell_size_m"] / size)
            ).tolist()
        if previous is not None:
            row["finite_loop_per_probe_order"] = (
                np.log(
                    np.asarray(previous["finite_loop_relative_errors"]) / finite_errors
                )
                / np.log(previous["cell_size_m"] / size)
            ).tolist()
        save()
        print(json.dumps(row), flush=True)
        if not solution.converged:
            raise SystemExit("material solve did not converge")
        previous = row


if __name__ == "__main__":
    main()
