"""Independent annulus extrapolation and resolved symmetry controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from scripts.iron_limb_reference.drive import FilamentDrive  # noqa: E402
from scripts.iron_limb_reference.measure import (  # noqa: E402
    axisymmetric_column_flux,
    solve_resolution,
    toroidal_field,
    write_json,
)
from scripts.iron_limb_reference.surface import (  # noqa: E402
    Panels,
    SurfaceModel,
    circuit_panels,
    image_frames,
)


def richardson(sizes, values):
    """Fit a positive observed order; refuse oscillatory or unresolved triples.

    Vector observables share an order estimated from successive difference
    norms. Their increments must point in the same direction. Uncertainty is
    the norm of the extrapolate minus the finest value, relative to the
    extrapolate. No assumed order substitutes for an inadmissible fit.
    """
    h = np.asarray(sizes, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(h) != 3 or len(y) != 3 or not np.all(np.diff(h) < 0):
        raise ValueError("Richardson needs three decreasing mesh sizes")
    if not np.isfinite(y).all():
        raise ValueError("Richardson values must be finite")
    first, second = y[0] - y[1], y[1] - y[2]
    a, b = np.linalg.norm(first), np.linalg.norm(second)
    record = dict(sizes_m=h.tolist(), finest=y[-1].tolist(), valid=False)
    if not a or not b:
        return dict(record, reason="zero mesh increment; observed order unresolved")
    alignment = float(np.sum(first * second) / (a * b))
    record["increment_alignment"] = alignment
    if alignment <= 0:
        return dict(record, reason="oscillatory mesh increments")

    def ratio(order):
        u, v = (h[:2] / h[2]) ** order
        return (u - v) / (v - 1)

    observed = a / b
    if not ratio(1e-5) < observed < ratio(20.0):
        return dict(
            record, reason="no positive observed order", difference_ratio=observed
        )
    order = brentq(lambda p: ratio(p) - observed, 1e-5, 20.0)
    extrapolate = y[2] + (y[2] - y[1]) / ((h[1] / h[2]) ** order - 1)
    correction = np.linalg.norm(extrapolate - y[2])
    scale = np.linalg.norm(extrapolate)
    return dict(
        record,
        valid=True,
        observed_order=float(order),
        extrapolated=extrapolate.tolist(),
        uncertainty_absolute=float(correction),
        uncertainty_relative=float(correction / scale) if scale else None,
    )


def extrapolate(rows, vessel_count):
    levels = rows[-3:]
    sizes = [r["size_m"] for r in levels]
    result = []
    for index, mu in enumerate((50.0, 200.0)):
        data = [r["rows"][index] for r in levels]
        field = np.asarray([r["iron_average_T"] for r in data])
        air = np.asarray(data[-1]["air_T"])
        result.append(
            dict(
                mu=mu,
                flux=richardson(sizes, [r["column_flux_Wb"] for r in data]),
                vessel=richardson(sizes, field[:, :vessel_count]),
                outside=[
                    richardson(sizes, field[:, k] + air[k])
                    for k in range(vessel_count, len(air))
                ],
            )
        )
    for row in result:
        vessel = row["vessel"]
        vessel["finest_rms_T"] = float(
            np.linalg.norm(vessel["finest"]) / np.sqrt(vessel_count)
        )
        if vessel["valid"]:
            vessel["extrapolated_rms_T"] = float(
                np.linalg.norm(vessel["extrapolated"]) / np.sqrt(vessel_count)
            )
            vessel["uncertainty_rms_T"] = vessel["uncertainty_absolute"] / np.sqrt(
                vessel_count
            )
    return result


def compare_extrapolated(axisymmetric, surface):
    comparisons = []
    for left, right in zip(axisymmetric, surface):
        items = [
            ("flux", left["flux"], right["flux"]),
            ("vessel", left["vessel"], right["vessel"]),
        ]
        items += [
            (f"outside_{k}", a, b)
            for k, (a, b) in enumerate(zip(left["outside"], right["outside"]))
        ]
        row = dict(mu=left["mu"], measures={}, passed=True)
        for name, a, b in items:
            if not (a["valid"] and b["valid"]):
                row["measures"][name] = dict(
                    passed=False, reason="unresolved extrapolation"
                )
                row["passed"] = False
                continue
            av, bv = np.asarray(a["extrapolated"]), np.asarray(b["extrapolated"])
            relative = float(np.linalg.norm(av - bv) / np.linalg.norm(bv))
            row["measures"][name] = dict(
                relative_difference=relative,
                passed=relative < 0.02,
                axisymmetric_uncertainty_relative=a["uncertainty_relative"],
                surface_uncertainty_relative=b["uncertainty_relative"],
            )
            row["passed"] &= relative < 0.02
        comparisons.append(row)
    return comparisons


def verified_prior(case_path, prior_path):
    """Reuse measured rows only when their owning source and case still match."""
    prior = json.loads(prior_path.read_text())
    if hashlib.sha256(case_path.read_bytes()).hexdigest() != prior["case_sha256"]:
        raise ValueError("prior control case digest differs")
    paths = [
        "scripts/iron_limb_reference/surface.py",
        "scripts/iron_limb_reference/drive.py",
        "scripts/iron_limb_reference/measure.py",
        "scripts/prototypes/iron_limb_fill/axisymmetric.py",
        "nova/biot/magnetisation.py",
    ]
    for path in paths:
        old = subprocess.check_output(["git", "show", f"{prior['revision']}:{path}"])
        if old != Path(path).read_bytes():
            raise ValueError(f"prior source differs: {path}")
    print("PRIOR_VERIFIED " + prior["revision"], flush=True)
    return prior


def annulus(case, args, record):
    prior = verified_prior(args.case, args.prior)
    record["prior_receipt"] = str(args.prior)
    record["surface"] = [r for r in prior["annulus"] if r["size_m"] in (0.1, 0.075)]
    record["axisymmetric"] = [
        r for r in prior["axisymmetric"] if r["size_m"] in (0.1, 0.075)
    ]
    if len(record["surface"]) != 2 or len(record["axisymmetric"]) != 2:
        raise ValueError("the known-present prior mesh pair is incomplete")
    write_json(args.output / "summary.json", record)
    record["surface"].append(
        solve_resolution(case, 0.05, args.output, annulus=True, bore=0.05)
    )
    write_json(args.output / "summary.json", record)
    gc.collect()
    jax.clear_caches()

    from scripts.prototypes.iron_limb_fill.axisymmetric import (
        Model,
        axisymmetric_drive,
        linear_regions,
    )

    external = axisymmetric_drive(case["sources"])
    probes = np.asarray(case["probes"])
    for size in (0.05, 0.0375):
        # The optional rung is selected by remaining runtime, never by agreement.
        if size == 0.0375 and time.monotonic() - args.started > 3600:
            record["optional_mesh"] = (
                "0.0375 m skipped: elapsed time exceeded one hour before this rung"
            )
            break
        model = Model(linear_regions()[0], size, external, lambda c: np.ones(len(c)))
        row = dict(size_m=size, cells=len(model.cells.centroids), rows=[])
        record["axisymmetric"].append(row)
        for mu in (50.0, 200.0):
            state = model.linear(mu)
            value = dict(
                mu=mu,
                column_flux_Wb=axisymmetric_column_flux(model, state, mu),
                iron_average_T=model.probe(state, probes).tolist(),
                air_T=external(probes).tolist(),
            )
            row["rows"].append(value)
            np.save(args.output / f"axisymmetric-{size:g}-mu{mu:g}.npy", state)
            write_json(args.output / "summary.json", record)
            print(
                "AXISYMMETRIC_ROW " + json.dumps(dict(size_m=size, **value)), flush=True
            )
        del model, state
        gc.collect()
        jax.clear_caches()
    record["axisymmetric_extrapolation"] = extrapolate(
        record["axisymmetric"], case["vessel_count"]
    )
    record["surface_extrapolation"] = extrapolate(
        record["surface"], case["vessel_count"]
    )
    record["comparison"] = compare_extrapolated(
        record["axisymmetric_extrapolation"], record["surface_extrapolation"]
    )
    record["passed"] = all(r["passed"] for r in record["comparison"])


def symmetry(case, args, record):
    probes = np.asarray(case["probes"])
    record["configurations"] = []
    for missing in (None, 0):
        panels = circuit_panels(0.5, missing_limb=missing)
        if missing is None:
            panels = Panels(
                np.concatenate(
                    [panels.vertices @ f.T for f in image_frames(6, False)[0]]
                )
            )
        print(
            f"SYMMETRY_BUILD missing={missing} panels={len(panels.vertices)}",
            flush=True,
        )
        model = SurfaceModel(panels, FilamentDrive(case["sources"]), rotations=1)
        configuration = dict(missing_limb=missing, panels=len(panels.vertices), rows=[])
        record["configurations"].append(configuration)
        for mu in (50.0, 200.0):
            charge, residual = model.solve(mu)
            angle, field = toroidal_field(model, charge, probes, 144, full=True)
            outside = field[:, case["vessel_count"] :]
            scale = np.linalg.norm(outside)
            if not np.isfinite(outside).all() or scale <= 0:
                raise ValueError("symmetry instrument needs a finite nonzero field")
            defect = np.linalg.norm(outside - np.roll(outside, 24, axis=0)) / scale
            row = dict(
                mu=mu,
                residual=residual,
                sixfold_defect_relative=float(defect),
                outside_field_norm_T=float(scale),
                angle_rad=angle.tolist(),
                iron_resolved_cylindrical_T=field.tolist(),
                neutrality=model.last_solve,
            )
            configuration["rows"].append(row)
            tag = "symmetric" if missing is None else "missing-limb"
            np.save(args.output / f"{tag}-mu{mu:g}-charge.npy", charge)
            write_json(args.output / "summary.json", record)
            print(
                "SYMMETRY_ROW "
                + json.dumps(
                    dict(
                        configuration=tag,
                        mu=mu,
                        residual=residual,
                        sixfold_defect_relative=float(defect),
                    )
                ),
                flush=True,
            )
        del model, panels
        gc.collect()
    record["comparison"] = []
    for base, removed in zip(*(c["rows"] for c in record["configurations"])):
        noise, signal = (
            base["sixfold_defect_relative"],
            removed["sixfold_defect_relative"],
        )
        record["comparison"].append(
            dict(
                mu=base["mu"],
                noise=noise,
                signal=signal,
                signal_to_noise=signal / max(noise, 1e-300),
                passed=signal > max(0.001, 10 * noise),
            )
        )
    record["passed"] = all(r["passed"] for r in record["comparison"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["annulus", "symmetry"])
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    args.started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    assert jax.config.jax_enable_x64
    case = json.loads(args.case.read_text())
    record = dict(
        mode=args.mode,
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        host=socket.gethostname(),
        case_sha256=hashlib.sha256(args.case.read_bytes()).hexdigest(),
        completed=False,
    )
    write_json(args.output / "summary.json", record)
    (annulus if args.mode == "annulus" else symmetry)(case, args, record)
    record["completed"] = True
    record["wall_seconds"] = time.monotonic() - args.started
    write_json(args.output / "summary.json", record)
    print("CONTROL_SUMMARY " + json.dumps(record.get("comparison")), flush=True)
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
