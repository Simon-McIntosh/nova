"""Read every runtime operand and summarize the observed stopping sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.fixed_point import _smooth_relative_sup_merit
from nova.jax.config import configure_dtypes


def analyze(source):
    record = json.loads((source / "receipt.json").read_text())
    stages = json.loads((source / "stages.json").read_text())
    inner = json.loads((source / "inner.json").read_text())
    ladder_path = source / "ladder.json"
    ladders = json.loads(ladder_path.read_text()) if ladder_path.exists() else []
    assert len(ladders) == record["counts"]["ladder"]
    assert sum(row["attempted"] for row in inner) == record["attempted"]
    assert sum(row["accepted"] for row in inner) == record["accepted"]
    invalid_images = []
    arrays_read = 0
    for row in stages:
        with np.load(source / f"{row['stage']}-{row['index']}.npz") as data:
            state, mapped = data["state"], data["mapped"]
            assert bool(np.isfinite(state).all()) == row["finite_state"]
            assert bool(np.isfinite(mapped).all()) == row["finite_image"]
            with np.errstate(invalid="ignore", divide="ignore"):
                residual = np.max(np.abs(mapped - state)) / max(
                    np.max(np.abs(mapped)), 1e-30
                )
            np.testing.assert_allclose(residual, row["residual"], rtol=2e-13)
            if row["finite_state"] and not row["finite_image"]:
                invalid_images.append(
                    dict(
                        stage=row["stage"],
                        index=row["index"],
                        nonfinite_flux_entries=int((~np.isfinite(mapped[:-1])).sum()),
                        finite_constraint_image=bool(np.isfinite(mapped[-1])),
                    )
                )
            arrays_read += 1
    for row in inner:
        with np.load(source / f"inner-{row['index']}.npz") as data:
            for key in data.files:
                assert np.isfinite(data[key]).all()
            arrays_read += 1
    with np.load(source / "terminal.npz") as terminal:
        physical = terminal["state"]
        last = np.load(source / f"inner-{len(inner) - 1}.npz")
        normalized = last["state"][:-1]
        pivot = int(np.argmax(np.abs(normalized)))
        flux_scale = float(physical[pivot] / normalized[pivot])
        np.testing.assert_allclose(
            normalized * flux_scale, physical, rtol=2e-13, atol=2e-14
        )
        augmented_residual = last["mapped"] - last["state"]
        np.testing.assert_allclose(
            augmented_residual[-1:], record["constraint_residual"], atol=2e-14
        )
        constraint_share = float(
            augmented_residual[-1] ** 2 / np.sum(augmented_residual**2)
        )
        terminal_merit = float(
            _smooth_relative_sup_merit(
                jnp.asarray(last["mapped"]), jnp.asarray(last["state"])
            )
        )
        residual = terminal["residual"]
        maximum = int(np.argmax(np.abs(residual)))
        position = None
        if maximum < terminal["flux"].size:
            ri, zi = np.unravel_index(maximum, terminal["flux"].shape)
            position = [float(terminal["radius"][ri]), float(terminal["height"][zi])]
        summary = dict(
            case=source.name,
            base_revision=record["base_revision"],
            runtime_arrays_read=arrays_read,
            ladder_rows_read=len(ladders),
            invalid_images=invalid_images,
            terminal_merit=terminal_merit,
            reconstructed_flux_scale_wb=flux_scale,
            constraint_squared_augmented_norm_fraction=constraint_share,
            constraint_relative_sup=float(
                abs(augmented_residual[-1]) / np.max(np.abs(last["mapped"]))
            ),
            physical_sup_index=maximum,
            physical_sup_position_rz=position,
            physical_sup_signed_wb=float(residual[maximum]),
            terminal_inner_reason=inner[-1]["reason"],
            terminal_inner_attempted=inner[-1]["attempted"],
            terminal_inner_accepted=inner[-1]["accepted"],
            all_warmup_images_finite=all(
                row["finite_image"] for row in stages if row["stage"] == "warmup"
            ),
            last_ladder=ladders[-1] if ladders else None,
        )
        last.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    assert jax.config.jax_enable_x64
    summaries = [
        analyze(path.parent) for path in sorted(args.input.glob("*/receipt.json"))
    ]
    args.output.write_text(json.dumps(summaries, indent=2) + "\n")
    for row in summaries:
        print(json.dumps(row))
    print("OPERAND CHECK PASSED: every stage, inner return and ladder receipt read")


if __name__ == "__main__":
    main()
