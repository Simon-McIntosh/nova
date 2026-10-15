"""Measure a bounded point mask against the independent single-null certificate."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
import nova
from scipy.ndimage import label


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("support_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(reference_module: Path, reference_results: Path, output: Path):
    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    reference = load_reference(reference_module)
    previous = json.loads(reference_results.read_text())["diverted"]
    case = reference.Case("diverted")
    case.jphi = lambda radius: np.asarray(
        -case.cert.grad_shafranov_source(
            np.stack((radius, np.zeros_like(radius)), axis=-1)
        )
        / (reference.MU0 * radius)
    )
    r0, r1, z0, z1 = case.domain

    def grid(n):
        radius = np.linspace(r0, r1, n)
        height = np.linspace(z0, z1, n)
        spacing = (r1 - r0) / (n - 1)
        return radius, height, spacing

    ref_r, ref_z, ref_h = grid(385)
    ref_sources, ref_current = reference.cell_sources(case, ref_r, ref_z, ref_h)
    ref_flux = reference.greens_flux(case.observations, ref_sources, ref_current)
    ref_total = float(ref_current.sum())
    rows = {}
    for n in (65, 129, 257):
        radius, height, spacing = grid(n)
        rr, zz = np.meshgrid(radius, height, indexing="ij")
        psi_n = case.psi_n(rr, zz)
        unbounded = psi_n > 0
        bounded = unbounded & (psi_n <= 1)
        seed = (
            min(int(np.searchsorted(radius, case.axis[0])), n - 1),
            min(int(np.searchsorted(height, case.axis[1])), n - 1),
        )
        masks = {"unbounded": unbounded, "windowed": bounded}
        for name, mask in tuple(masks.items()):
            labels, _ = label(mask, structure=np.ones((3, 3), dtype=bool))
            masks["connected_" + name] = mask & (labels == labels[seed])
        errors = {}
        for name, mask in masks.items():
            sources, currents = reference.node_sources(
                radius, height, mask, spacing, case.jphi
            )
            mapped = reference.greens_flux(
                case.observations,
                sources,
                currents * (ref_total / currents.sum()),
            )
            errors[name] = float(
                np.max(
                    np.abs(
                        mapped[: reference.OBS_INTERIOR]
                        - ref_flux[: reference.OBS_INTERIOR]
                    )
                )
                / case.span
            )
        prior = previous[str(n)]["B"]["error_sup"]
        assert abs(errors["connected_unbounded"] - prior) < 1e-10
        private = unbounded & ~masks["connected_unbounded"]
        private_current = case.jphi(rr[private]) * spacing**2
        total_current = case.jphi(rr[unbounded]) * spacing**2
        rows[n] = {
            "errors": errors,
            "selected": {key: int(value.sum()) for key, value in masks.items()},
            "private_count": int(private.sum()),
            "private_psi_n_range": (
                [float(psi_n[private].min()), float(psi_n[private].max())]
                if private.any()
                else []
            ),
            "private_current_fraction": float(
                private_current.sum() / total_current.sum()
            ),
            "reference_positive_control": prior,
        }
        print("ROW " + json.dumps({"nodes": n, **rows[n]}), flush=True)
    result = {
        "reference_grid": 385,
        "reference_sources": len(ref_current),
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-module", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_module, args.reference_results, args.output)


if __name__ == "__main__":
    main()
