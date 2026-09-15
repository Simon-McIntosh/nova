"""Lower the pinned-branch gate solve to StableHLO for two DIII-D gate rows.

Records whether the gate rows compile to one program per lattice mesh or bake
the prescribed target current into the program: a two-row bank, each row also
carrying the other row's target current, is lowered under ``jax.jit`` and the
four raw digests compared.  When all four raw digests are equal the solve is a
single program per mesh (the current arrived as traced data); otherwise each
target current is baked in as a constant.

The four variants' MLIR and a JSON receipt are written to the artifacts
directory; a copy of the receipt carrying the worktree revision and a verdict
field is written to the figure path under ``docs/figures/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import efit_forward_parity_slice as mast
from benchmarks import efit_reproduction_gate as gate
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECEIPT = (
    REPO_ROOT
    / "docs/figures/diiid-vertical-force-balance/two-field-gate"
    / "target-current-digest-probe.json"
)
DEFAULT_ARTIFACTS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-handoff/digest-probe"
)

VARIANT_NAMES = (
    "row_one",
    "row_two",
    "row_one_with_row_two_target_current",
    "row_two_with_row_one_target_current",
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalise_constants(text: str) -> str:
    text = re.sub(r"dense<[^>]*>", "dense<CONST>", text)
    return re.sub(r"stablehlo.constant[^\n]*", "stablehlo.constant CONST", text)


def lower(profile, state, target_current: float) -> tuple[str, dict[str, object]]:
    initial = jnp.asarray(state)

    def solve(dynamic_state):
        return profile.solve_branch(
            dynamic_state,
            TopologyClass.DIVERTED,
            route="newton_krylov",
            target_current=target_current,
            tolerance=mast.FIXED_POINT_CRITERION,
            newton_steps=mast.NEWTON_STEPS,
            gmres_iterations=mast.GMRES_ITERATIONS,
            warmup=mast.WARMUP_SWEEPS,
            relaxation=mast.RELAXATION,
            step_cap=mast.STEP_CAP,
        )

    stablehlo = str(jax.jit(solve).lower(initial).compiler_ir(dialect="stablehlo"))
    return stablehlo, {
        "stablehlo_sha256": digest(stablehlo),
        "constant_normalised_stablehlo_sha256": digest(normalise_constants(stablehlo)),
        "stablehlo_bytes": len(stablehlo.encode()),
        "input_shape": list(initial.shape),
        "input_dtype": str(initial.dtype),
    }


def worktree_revision() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--no-mlir", action="store_true")
    args = parser.parse_args()

    configure_dtypes()
    if not bool(jax.config.x64_enabled):
        raise RuntimeError("stablehlo probe requires x64")
    selected = mast.select_slices_by_shot(mast.DECOMPOSITION_BANK)[:2]
    response_cache, carrier = gate._persisted_mast_response_cache(
        gate.DEFAULT_MAST_RESPONSE_CARRIER
    )
    built = []
    for selected_row, qualification in selected:
        case, context = mast._mast_case_from_selection(
            mast.SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = mast._passive_inclusive_case(
            case, context, response_cache
        )
        if not policy["response_matrix_reused"]:
            raise RuntimeError("response carrier was not reused")
        built.append(
            {
                "identity": {
                    "shot": int(selected_row["shot"]),
                    "slice_index": int(selected_row["slice_index"]),
                },
                "state": passive_case["state"],
                "profile": profile,
                "target_current_a": abs(float(case["reference"]["plasma_current_a"])),
                "lattice_shape": list(profile.lattice.shape),
                "lattice_node_count": int(profile.lattice.node_count),
                "wall_node_count": int(len(profile.operator.wall.coordinate)),
                "profile_leaf_shapes": [
                    list(np.shape(leaf)) for leaf in jax.tree.leaves(profile)
                ],
            }
        )

    variants = {
        "row_one": (
            built[0]["profile"],
            built[0]["state"],
            built[0]["target_current_a"],
        ),
        "row_two": (
            built[1]["profile"],
            built[1]["state"],
            built[1]["target_current_a"],
        ),
        "row_one_with_row_two_target_current": (
            built[0]["profile"],
            built[0]["state"],
            built[1]["target_current_a"],
        ),
        "row_two_with_row_one_target_current": (
            built[1]["profile"],
            built[1]["state"],
            built[0]["target_current_a"],
        ),
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    lowered = {}
    for name, (profile, state, target_current) in variants.items():
        stablehlo, record = lower(profile, state, target_current)
        path = args.artifacts / f"{name}.mlir"
        if not args.no_mlir:
            path.write_text(stablehlo)
        lowered[name] = {
            **record,
            "target_current_a": target_current,
            "stablehlo_path": str(path),
        }
        print(name, json.dumps(lowered[name], sort_keys=True), flush=True)

    raw_digests = tuple(lowered[name]["stablehlo_sha256"] for name in VARIANT_NAMES)
    one_program = len(set(raw_digests)) == 1
    verdict = "one-program-per-mesh" if one_program else "target-current-baked"

    receipt = {
        "revision": worktree_revision(),
        "verdict": verdict,
        "jax_x64_enabled": bool(jax.config.x64_enabled),
        "jax_backend": jax.default_backend(),
        "carrier": carrier,
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"state", "profile"}
            }
            for row in built
        ],
        "lowered_variants": lowered,
        "comparisons": {
            "row_programs_identical": (
                lowered["row_one"]["stablehlo_sha256"]
                == lowered["row_two"]["stablehlo_sha256"]
            ),
            "row_programs_identical_after_constant_normalisation": (
                lowered["row_one"]["constant_normalised_stablehlo_sha256"]
                == lowered["row_two"]["constant_normalised_stablehlo_sha256"]
            ),
            "target_current_alone_changes_row_one_program": (
                lowered["row_one"]["stablehlo_sha256"]
                != lowered["row_one_with_row_two_target_current"]["stablehlo_sha256"]
            ),
            "target_current_alone_changes_row_two_program": (
                lowered["row_two"]["stablehlo_sha256"]
                != lowered["row_two_with_row_one_target_current"]["stablehlo_sha256"]
            ),
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (args.artifacts / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    headline = {"revision": receipt["revision"], "verdict": verdict}
    print(json.dumps(headline, sort_keys=True))
    print(json.dumps(receipt["comparisons"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
