"""Resolve control discretisation and the nonlinear correction budget separately."""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

from scripts.iron_limb_production import measure as production

import jax
import numpy as np
from scipy.optimize import brentq, minimize_scalar

iron = production.iron
ORIGINAL = production.EVIDENCE / "production/material-20261114/summary.json"


@contextmanager
def reuse_mesh():
    """Reuse an identical authored mesh across constitutive comparison arms."""
    original = iron._mesh
    cache = {}

    def mesh(region, size, *args, **kwargs):
        key = (region.wkb, size)
        if key not in cache:
            cache.clear()
            cache[key] = original(region, size, *args, **kwargs)
        return cache[key]

    iron._mesh = mesh
    try:
        with production.reuse_operators():
            yield
    finally:
        iron._mesh = original


def save(record, output):
    production.write(output / "summary.json", record)


def memory_plan(record, output):
    """Reserve four dense float64 matrices plus device workspace before assembly."""
    flat, _, _ = production.linear_regions()
    stats = jax.devices()[0].memory_stats()
    limit = stats["bytes_limit"]
    record["memory_limit_bytes"] = limit
    record["memory_policy"] = (
        "four dense (2*cells)^2 float64 matrices, below 80% of device limit"
    )
    record["memory_candidates"] = []
    accepted = []
    for size in (0.035, 0.025, 0.03):
        cells = iron._mesh(flat, size)
        count = len(cells.centroids)
        peak = 4 * (2 * count) ** 2 * 8
        row = dict(
            size_m=size, cells=count, dense_peak_bytes=peak, admitted=peak < 0.8 * limit
        )
        record["memory_candidates"].append(row)
        print("MEMORY " + json.dumps(row), flush=True)
        save(record, output)
        if row["admitted"]:
            accepted.append(size)
        del cells
        gc.collect()
        if 0.035 in accepted and 0.025 in accepted:
            break
    sizes = sorted(accepted, reverse=True)
    assert len(sizes) >= 2, "device does not admit two finer control rungs"
    record["new_sizes_m"] = [0.035, min(sizes)]
    assert record["new_sizes_m"][0] > record["new_sizes_m"][1]
    save(record, output)
    return record["new_sizes_m"]


def observe(record, output, case, references, name, size, mu, factor):
    started = perf_counter()
    solution = production.control_result(case, name, size, mu, factor)
    points = np.asarray(case["probes"])
    field = solution.field(points)
    reference = np.asarray(references[mu]["iron_average_T"])
    error = float(
        np.linalg.norm(field[:16] - reference[:16]) / np.linalg.norm(reference[:16])
    )
    row = dict(
        model=name,
        size_m=size,
        mu=mu,
        cells=solution.cell_count,
        cross_field_factor=factor
        if name in ("geometry", "fitted", "refitted")
        else 0.0,
        converged=solution.converged,
        iterations=solution.iterations,
        residual_history=solution.residual_history,
        vessel_relative_rms=error,
        iron_T=field.tolist(),
        device=record["device"],
        seconds=perf_counter() - started,
    )
    record["control"].append(row)
    save(record, output)
    print(
        "CONTROL " + json.dumps({k: v for k, v in row.items() if k != "iron_T"}),
        flush=True,
    )
    assert solution.converged
    return row


def convergence_rows(record):
    """Extrapolate only strictly shrinking three-rung error sequences."""
    record["mesh_steps"] = []
    record["richardson"] = []
    for model in ("smear", "plain", "geometry", "fitted"):
        for mu in (50.0, 200.0):
            rows = sorted(
                [r for r in record["control"] if r["model"] == model and r["mu"] == mu],
                key=lambda r: r["size_m"],
                reverse=True,
            )
            for coarse, fine in zip(rows[:-1], rows[1:], strict=True):
                record["mesh_steps"].append(
                    dict(
                        model=model,
                        mu=mu,
                        coarse_m=coarse["size_m"],
                        fine_m=fine["size_m"],
                        error_step=fine["vessel_relative_rms"]
                        - coarse["vessel_relative_rms"],
                    )
                )
            for a, b, c in zip(rows[:-2], rows[1:-1], rows[2:], strict=True):
                hs = np.array([a["size_m"], b["size_m"], c["size_m"]])
                es = np.array(
                    [
                        a["vessel_relative_rms"],
                        b["vessel_relative_rms"],
                        c["vessel_relative_rms"],
                    ]
                )
                row = dict(model=model, mu=mu, sizes_m=hs.tolist(), errors=es.tolist())
                if np.all(np.diff(es) < 0):
                    ratio = (es[0] - es[1]) / (es[1] - es[2])

                    def equation(p):
                        return ((hs[0] / hs[1]) ** p - 1) / (
                            1 - (hs[2] / hs[1]) ** p
                        ) - ratio

                    if equation(0.01) * equation(12) < 0:
                        order = brentq(equation, 0.01, 12)
                        extrapolate = es[2] - (es[1] - es[2]) / (
                            (hs[1] / hs[2]) ** order - 1
                        )
                        row.update(
                            order=float(order), extrapolated_error=float(extrapolate)
                        )
                    else:
                        row["reason"] = (
                            "monotone errors but no positive order in [0.01,12]"
                        )
                else:
                    row["reason"] = "errors do not shrink monotonically"
                record["richardson"].append(row)
    finest = min(record["new_sizes_m"])
    record["finest_acceptance"] = []
    for mu in (50.0, 200.0):
        geometry = next(
            r
            for r in record["control"]
            if r["model"] == "geometry" and r["size_m"] == finest and r["mu"] == mu
        )
        smear = next(
            r
            for r in record["control"]
            if r["model"] == "smear" and r["size_m"] == finest and r["mu"] == mu
        )
        ratio = geometry["vessel_relative_rms"] / smear["vessel_relative_rms"]
        record["finest_acceptance"].append(
            dict(
                mu=mu,
                size_m=finest,
                error=geometry["vessel_relative_rms"],
                smear_error_ratio=ratio,
                passed=geometry["vessel_relative_rms"] < 0.1 and ratio <= 1 / 3,
            )
        )
    record["control_passed"] = all(r["passed"] for r in record["finest_acceptance"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    device = jax.devices()[0]
    assert device.platform == "gpu"
    raw = ORIGINAL.read_bytes()
    previous = json.loads(raw)
    case, references = production.reference_inputs()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    record = dict(
        revision=revision,
        device=device.device_kind,
        module=iron.__file__,
        cwd=str(Path.cwd().resolve()),
        source_receipt=str(ORIGINAL),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        original_fit=previous["fit"],
        geometry_factor=production.GEOMETRY_FACTOR,
        control=[r for r in previous["control"] if r["model"] != "solid"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save(record, args.output)
    print(
        "PROVENANCE " + json.dumps({k: v for k, v in record.items() if k != "control"}),
        flush=True,
    )
    sizes = memory_plan(record, args.output)
    old_factor = previous["fit"]["factor"]
    for size in sizes:
        with reuse_mesh():
            for model in ("plain", "geometry", "fitted"):
                for mu in (50.0, 200.0):
                    observe(
                        record,
                        args.output,
                        case,
                        references,
                        model,
                        size,
                        mu,
                        old_factor if model == "fitted" else production.GEOMETRY_FACTOR,
                    )
            if size == min(sizes):

                def objective(factor):
                    errors = []
                    for mu in (50.0, 200.0):
                        solution = production.control_result(
                            case, "fitted", size, mu, factor
                        )
                        ref = np.asarray(references[mu]["iron_average_T"])[:16]
                        errors.append(
                            np.linalg.norm(
                                solution.field(np.asarray(case["probes"])[:16]) - ref
                            )
                            / np.linalg.norm(ref)
                        )
                    value = float(np.linalg.norm(errors))
                    record.setdefault("fit_samples", []).append(
                        dict(factor=float(factor), errors=errors, objective=value)
                    )
                    save(record, args.output)
                    print("FIT " + json.dumps(record["fit_samples"][-1]), flush=True)
                    return value

                fit = minimize_scalar(
                    objective,
                    bounds=(0.0, 1.0),
                    method="bounded",
                    options={"xatol": 0.005},
                )
                assert fit.success
                record["finest_fit"] = dict(
                    size_m=size,
                    factor=float(fit.x),
                    error=float(fit.fun),
                    step_from_original=float(fit.x - old_factor),
                )
                save(record, args.output)
                for mu in (50.0, 200.0):
                    row = observe(
                        record,
                        args.output,
                        case,
                        references,
                        "fitted",
                        size,
                        mu,
                        float(fit.x),
                    )
                    row["model"] = "refitted"
                    save(record, args.output)
            for mu in (50.0, 200.0):
                observe(record, args.output, case, references, "smear", size, mu, 0.0)
        gc.collect()
        jax.clear_caches()
    convergence_rows(record)
    save(record, args.output)
    production.measure_west(record, args.output, sizes=(0.04,), max_iterations=30)
    record["west_correction_budget"] = 30
    record["passed"] = record["control_passed"] and record["west_passed"]
    save(record, args.output)
    print(
        "SUMMARY passed="
        + str(record["passed"])
        + " control_passed="
        + str(record["control_passed"])
        + " west_passed="
        + str(record["west_passed"]),
        flush=True,
    )
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
