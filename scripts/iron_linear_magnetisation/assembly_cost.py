"""Separate evaluator compilation, launch and host assembly on a spherical shell."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot import tiledassembly as tiled  # noqa: E402
from scripts.iron_linear_magnetisation.controls import require_clean_script, shell  # noqa: E402


class Evaluators:
    """Time each fixed-shape executable once and count every padded pair."""

    def __init__(self):
        self.programs = {}
        self.executables = {}

    def wrap(self, name, function):
        def evaluate(*arguments):
            tile = isinstance(function, tiled.TileEvaluator)
            prepared = function.prepare(*arguments) if tile else None
            arrays = prepared[2] if tile else arguments
            key = (name, tuple((np.shape(x), str(x.dtype)) for x in arrays))
            label = repr(key)
            if key not in self.executables:
                started = time.monotonic()
                self.executables[key] = (
                    function.compile(prepared)
                    if tile
                    else function.lower(*arguments).compile()
                )
                self.programs[label] = dict(
                    evaluator=name,
                    shapes=[list(np.shape(x)) for x in arguments],
                    compile_seconds=time.monotonic() - started,
                    execute_seconds=0.0,
                    calls=0,
                    padded_pairs=0,
                )
            row = self.programs[label]
            started = time.monotonic()
            result = self.executables[key](*arrays)
            jax.block_until_ready(result)
            row["execute_seconds"] += time.monotonic() - started
            row["calls"] += 1
            row["padded_pairs"] += function.plan.block if tile else len(arguments[0])
            return (
                function.materialize(result, prepared[0], prepared[1])
                if tile
                else result
            )

        return evaluate

    def factory(self, name, original):
        cached = {}

        def factory(*arguments, **keywords):
            key = (arguments, tuple(sorted(keywords.items())))
            if key not in cached:
                cached[key] = self.wrap(
                    f"{name}{key}", original(*arguments, **keywords)
                )
            return cached[key]

        return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--sizes", type=float, nargs="+", default=[0.04, 0.02])
    parser.add_argument("--comparison", type=Path)
    args = parser.parse_args()
    stamp = dict(
        revision=args.revision,
        script_blob=require_clean_script(__file__, args.blob, args.revision),
    )
    assert jax.config.jax_enable_x64
    jax.config.update("jax_enable_compilation_cache", False)
    args.output.mkdir(parents=True, exist_ok=True)
    timing = Evaluators()
    for name in ("_mutual_edge_evaluator", "_polygon_current_evaluator"):
        if hasattr(tiled, name):
            setattr(tiled, name, timing.factory(name, getattr(tiled, name)))
    for size in args.sizes:
        row = dict(
            **stamp,
            cell_size_m=size,
            devices=[device.device_kind for device in jax.devices()],
            module=str(Path(tiled.__file__).resolve()),
            cwd=str(Path.cwd().resolve()),
            x64=True,
            completed=False,
        )
        path = args.output / f"assembly-{size:g}.json"

        def save():
            path.write_text(json.dumps(row, indent=2) + "\n")

        cells = iron._mesh(shell(), size)
        geometry = tiled.linear_magnetisation_geometry(cells.polygons)
        packed = tiled._polygon_current_geometry(cells.polygons)
        row.update(
            cells=len(cells.polygons),
            atoms=len(geometry["atoms"]),
            triangles=len(packed[2]),
            volume_points=len(geometry["points"]),
        )
        save()
        before = {key: value.copy() for key, value in timing.programs.items()}
        started = time.monotonic()
        operator, _ = tiled.linear_galerkin_cell_interaction(
            cells.polygons, plan=iron._PLAN
        )
        row["assembly_seconds"] = time.monotonic() - started
        programs = []
        for key, program in timing.programs.items():
            old = before.get(key, {})
            programs.append(
                dict(
                    program,
                    **{
                        field: program[field] - old.get(field, 0)
                        for field in (
                            "compile_seconds",
                            "execute_seconds",
                            "calls",
                            "padded_pairs",
                        )
                    },
                )
            )
        row.update(
            programs=programs,
            compiled_programs=sum(key not in before for key in timing.programs),
            compile_seconds=sum(p["compile_seconds"] for p in programs),
            execute_seconds=sum(p["execute_seconds"] for p in programs),
            kernel_evaluations_per_cell_pair=sum(p["padded_pairs"] for p in programs)
            / len(cells.polygons) ** 2,
            seconds_per_cell_pair=row["assembly_seconds"] / len(cells.polygons) ** 2,
        )
        row["host_seconds"] = (
            row["assembly_seconds"] - row["compile_seconds"] - row["execute_seconds"]
        )
        row["projected_seconds"] = {
            str(count): row["seconds_per_cell_pair"] * count**2
            for count in (5864, 11708)
        }
        if args.comparison and size == 0.04:
            delivered = np.load(args.comparison)
            row["operator_relative_difference"] = float(
                np.linalg.norm(operator - delivered) / np.linalg.norm(delivered)
            )
            if row["operator_relative_difference"] > 1e-12:
                save()
                raise SystemExit("linear operator differs from delivered interaction")
        target = args.output / f"operator-{size:g}.npy"
        np.save(target, operator)
        row["operator_path"] = str(target)
        row["operator_sha256"] = hashlib.file_digest(
            target.open("rb"), "sha256"
        ).hexdigest()
        row["completed"] = True
        save()
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
