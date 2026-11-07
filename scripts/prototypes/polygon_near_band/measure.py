"""Measure polygon flux moments for the packaged MAST and DIII-D coil sets."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from unittest.mock import patch

import jax
import numpy as np
import shapely

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64

from nova.biot.polygon import pad_batch  # noqa: E402
import nova.biot.polygonanalytic as polygonanalytic  # noqa: E402
from nova.catalog.mast_geometry import MachineGeometryRegistry  # noqa: E402
from nova.imas.diiid_current import (  # noqa: E402
    DEFAULT_ACTIVE_COIL_DD_VERSION,
)

MAST_SHOT = 21978
DIIID_ENTRY = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")
DIIID_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/netcdf-description/"
    "netcdf_description_receipt.json"
)
DIIID_DD_VERSION = DEFAULT_ACTIVE_COIL_DD_VERSION
REPETITIONS = 7


@dataclass(frozen=True)
class CoilSet:
    name: str
    polygons: tuple[np.ndarray, ...]


def _polygon_rows(geometry) -> list[np.ndarray]:
    if geometry.geom_type == "Polygon":
        members = (geometry,)
    elif geometry.geom_type == "MultiPolygon":
        members = tuple(geometry.geoms)
    else:
        return []
    polygons = []
    for member in members:
        vertices = np.asarray(member.exterior.coords, dtype=np.float64)[:-1, :2]
        if len(vertices) >= 3 and np.isfinite(vertices).all():
            polygons.append(vertices)
    return polygons


def _mast_coils() -> CoilSet:
    selected = MachineGeometryRegistry.default().select(MAST_SHOT)
    geometry = selected.configuration.geometry
    polygons = [
        polygon
        for _, encoded in sorted(geometry["active_components"].items())
        for polygon in _polygon_rows(shapely.from_wkb(bytes.fromhex(encoded)))
    ]
    if not polygons:
        raise RuntimeError("the selected MAST registry row has no active polygons")
    return CoilSet("MAST", tuple(polygons))


def _diiid_coils() -> CoilSet:
    if not DIIID_ENTRY.is_file():
        raise FileNotFoundError(DIIID_ENTRY)
    receipt = json.loads(DIIID_RECEIPT.read_text(encoding="utf-8"))
    source = receipt["source"]
    if Path(source["path"]) != DIIID_ENTRY:
        raise RuntimeError(f"DIII-D receipt source does not match {DIIID_ENTRY}")
    if source["backend"] != "imas-python netCDF DBEntry":
        raise RuntimeError("DIII-D geometry receipt is not from the IMAS reader")
    if receipt["dd_versions"]["pf_active"] != DIIID_DD_VERSION:
        raise RuntimeError("DIII-D geometry receipt has a different DD version")
    if 0 not in source["occurrences"]["pf_active"]:
        raise RuntimeError("DIII-D pf_active occurrence 0 is absent from the receipt")
    active = receipt["quantities"]["pf_active"]
    polygons = [
        np.asarray(element["outline_vertices_m"], dtype=np.float64)
        for coil in active["coils"]
        for element in coil["elements"]
    ]
    if len(polygons) != active["element_count"]:
        raise RuntimeError("DIII-D receipt element count does not match its polygons")
    if not polygons:
        raise RuntimeError("the DIII-D pf_active IDS contains no polygon elements")
    return CoilSet("DIII-D", tuple(polygons))


def _target_cloud(polygons: tuple[np.ndarray, ...]) -> np.ndarray:
    bounds = np.concatenate(polygons)
    lower = bounds.min(axis=0)
    upper = bounds.max(axis=0)
    padding = np.array([0.15, 0.15])
    side = int(np.ceil(np.sqrt(len(polygons))))
    radial = np.linspace(lower[0] - padding[0], upper[0] + padding[0], side)
    vertical = np.linspace(lower[1] - padding[1], upper[1] + padding[1], side)
    grid_r, grid_z = np.meshgrid(radial, vertical, indexing="xy")
    grid = np.column_stack((grid_r.ravel(), grid_z.ravel()))
    selected = grid[np.linspace(0, len(grid) - 1, len(polygons), dtype=int)]
    selected[0] = polygons[0][0]
    return selected


def _near_band_fraction(
    polygons: tuple[np.ndarray, ...], targets: np.ndarray
) -> tuple[float, float, int, int]:
    selected_pairs = 0
    selected_corners = 0
    total_corners = 0
    if len(polygons) != len(targets):
        raise ValueError("each source section must have one paired target")
    for polygon, target in zip(polygons, targets, strict=True):
        radius = abs(target[0])
        offset = polygon[:, 0] - radius
        level = polygon[:, 1] - target[1]
        mask = (np.abs(offset) <= 1e-3) & (np.hypot(level, offset) < 1e-4 * radius)
        selected_pairs += int(np.any(mask))
        selected_corners += int(np.count_nonzero(mask))
        total_corners += int(mask.size)
    pair_count = len(polygons)
    return (
        selected_pairs / pair_count,
        selected_corners / total_corners,
        selected_pairs,
        pair_count,
    )


def _pack(polygons: tuple[np.ndarray, ...], targets: np.ndarray):
    if len(polygons) != len(targets):
        raise ValueError("each source section must have one paired target")
    edge, weight, norm = pad_batch(list(polygons))
    pair_count = len(polygons)
    target_r = targets[:, 0]
    target_z = targets[:, 1]
    centre = np.stack([np.mean(polygon, axis=0) for polygon in polygons]).T
    reflection_axis = np.full(pair_count, np.nan)
    reflection_partner = np.broadcast_to(
        np.arange(len(edge))[:, None], (len(edge), pair_count)
    ).copy()
    return (
        target_r,
        target_z,
        edge,
        weight,
        norm,
        centre,
        centre,
        reflection_axis,
        reflection_partner,
    )


def _run_case(
    coilset: CoilSet,
    device: str,
    arm: str,
    kernel,
    *,
    force_all: bool = False,
) -> dict:
    polygons = coilset.polygons[:1] if arm == "single" else coilset.polygons
    targets = polygons[0][:1] if arm == "single" else _target_cloud(coilset.polygons)
    pair_fraction, corner_fraction, selected, total = _near_band_fraction(
        polygons, targets
    )
    if force_all:
        pair_fraction = corner_fraction = 1.0
        selected = total
    if arm == "single" and selected != 1:
        raise AssertionError(
            "the exact-corner positive control did not select the band"
        )
    packed = _pack(polygons, targets)
    device_inputs = tuple(jax.device_put(value) for value in packed)
    compiled_kernel = jax.jit(
        lambda *values: kernel.packed_analytic_moments(jax.numpy, *values)[:3]
    )
    print(f"MEASUREMENT_MODULE={Path(kernel.__file__).resolve()}", flush=True)
    print(f"MEASUREMENT_CWD={Path.cwd().resolve()}", flush=True)
    compile_started = time.perf_counter()
    band_patch = (
        patch.object(
            kernel,
            "_corner_remainder_band",
            side_effect=lambda radius, level, offset, xp: xp.ones_like(
                radius, dtype=bool
            ),
        )
        if force_all
        else nullcontext()
    )
    with band_patch:
        executable = compiled_kernel.lower(*device_inputs).compile()
    compile_seconds = time.perf_counter() - compile_started
    executable(*device_inputs)[0].block_until_ready()
    walls = []
    for _ in range(REPETITIONS):
        started = time.perf_counter()
        outputs = executable(*device_inputs)
        jax.block_until_ready(outputs)
        walls.append(time.perf_counter() - started)
    median_seconds = statistics.median(walls)
    actual_device = str(jax.devices()[0])
    return {
        "machine": coilset.name,
        "arm": arm,
        "requested_device": device,
        "device": actual_device,
        "backend": jax.default_backend(),
        "sections": len(polygons),
        "vertices": sum(len(polygon) for polygon in polygons),
        "target_count": len(targets),
        "pair_count": total,
        "module_file": str(Path(kernel.__file__).resolve()),
        "measurement_cwd": str(Path.cwd().resolve()),
        "near_band_target_fraction": pair_fraction,
        "near_band_corner_fraction": corner_fraction,
        "selected_pairs": selected,
        "compile_ms": compile_seconds * 1e3,
        "wall_ms": median_seconds * 1e3,
        "wall_samples_ms": [value * 1e3 for value in walls],
    }


@contextmanager
def _baseline_kernel(revision: str):
    """Load the pinned kernel family into a temporary tree for comparison."""
    root = Path(__file__).resolve().parents[3]
    module_names = (
        "rangefunction",
        "completeelliptic",
        "elliptic",
        "gradedresidual",
        "momentchannel",
        "polygonanalytic",
        "greens",
    )
    replacements = {}
    with tempfile.TemporaryDirectory(
        prefix="polygon-near-band-", dir=os.environ.get("TMPDIR")
    ) as directory:
        for name in module_names:
            source_path = Path(directory) / f"{name}.py"
            source_path.write_bytes(
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(root),
                        "show",
                        f"{revision}:nova/biot/{name}.py",
                    ]
                )
            )
            spec = importlib.util.spec_from_file_location(
                f"reference_{name}", source_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for key, value in tuple(vars(module).items()):
                origin = getattr(value, "__module__", "")
                if origin in replacements:
                    setattr(module, key, getattr(replacements[origin], value.__name__))
            replacements[f"nova.biot.{name}"] = module
        yield replacements["nova.biot.polygonanalytic"]


def _measure_kernel(coilsets, device: str, kernel, *, force_all: bool) -> dict:
    return {
        coilset.name: {
            "section_count": len(coilset.polygons),
            "polygon_vertex_counts": [len(row) for row in coilset.polygons],
            "measurements": [
                _run_case(coilset, device, arm, kernel, force_all=force_all)
                for arm in ("single", "batched")
            ],
        }
        for coilset in coilsets
    }


@contextmanager
def _without_stable_quadrature(kernel):
    """Remove only the expensive weighted quadratures from the base kernel."""
    original_residual = kernel._Vertex._first_residual

    def residual_without_weight(self, nodes, *, weight=None):
        if weight is not None:
            return self.xp.zeros_like(self.radius)
        return original_residual(self, nodes, weight=weight)

    with (
        patch.object(
            kernel,
            "_weighted_root",
            side_effect=lambda vertex, term, *, differential=None: vertex.xp.zeros_like(
                vertex.radius
            ),
        ),
        patch.object(kernel._Vertex, "_first_residual", residual_without_weight),
    ):
        yield


def measure_branch_share(device: str, baseline_revision: str) -> dict:
    """Measure baseline walls with and without the selected weighted quadratures."""
    if jax.default_backend() != device:
        raise RuntimeError(f"requested {device}, JAX selected {jax.default_backend()}")
    coilsets = (_mast_coils(), _diiid_coils())
    result = {
        "baseline_revision": baseline_revision,
        "device": str(jax.devices()[0]),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "stable_quadrature_removed": "weighted_root and weighted residual only",
        "coilsets": {},
    }
    with _baseline_kernel(baseline_revision) as kernel:
        for coilset in coilsets:
            rows = []
            for arm in ("single", "batched"):
                full = _run_case(coilset, device, arm, kernel)
                with _without_stable_quadrature(kernel):
                    reduced = _run_case(coilset, device, arm, kernel)
                share = (full["wall_ms"] - reduced["wall_ms"]) / full["wall_ms"]
                row = {
                    "arm": arm,
                    "near_band_target_fraction": full["near_band_target_fraction"],
                    "near_band_corner_fraction": full["near_band_corner_fraction"],
                    "stable_quadrature_wall_ms": full["wall_ms"],
                    "stable_quadrature_removed_wall_ms": reduced["wall_ms"],
                    "stable_branch_wall_share": share,
                    "full": full,
                    "without_stable_quadrature": reduced,
                }
                rows.append(row)
                result["coilsets"][coilset.name] = {
                    "section_count": len(coilset.polygons),
                    "polygon_vertex_counts": [len(row) for row in coilset.polygons],
                    "measurements": rows,
                }
                print(
                    "BRANCH_SHARE "
                    f"machine={coilset.name} arm={arm} "
                    f"near_targets={full['near_band_target_fraction']:.6g} "
                    f"near_corners={full['near_band_corner_fraction']:.6g} "
                    f"baseline_ms={full['wall_ms']:.6g} "
                    f"without_stable_ms={reduced['wall_ms']:.6g} "
                    f"stable_share={share:.6g}",
                    flush=True,
                )
    return result


def assert_throughput(record: dict, *, maximum_ratio: float = 0.95):
    """Require the batched flux kernel to beat its recorded before wall."""
    for machine, entry in record["after"]["coilsets"].items():
        before = next(
            row["wall_ms"]
            for row in record["before"]["coilsets"][machine]["measurements"]
            if row["arm"] == "batched"
        )
        after = next(
            row["wall_ms"] for row in entry["measurements"] if row["arm"] == "batched"
        )
        bound = before * maximum_ratio
        print(
            f"THROUGHPUT machine={machine} before_wall_ms={before:.6g} "
            f"after_wall_ms={after:.6g} bound_ms={bound:.6g}",
            flush=True,
        )
        assert after < bound, (
            f"{machine} batched wall {after:.6g} ms did not beat "
            f"the {maximum_ratio:.3f} baseline bound ({before:.6g} ms)"
        )


def compare(device: str, baseline_revision: str, *, force_all: bool = False) -> dict:
    if jax.default_backend() != device:
        raise RuntimeError(f"requested {device}, JAX selected {jax.default_backend()}")
    coilsets = (_mast_coils(), _diiid_coils())
    with _baseline_kernel(baseline_revision) as baseline_kernel:
        before = _measure_kernel(coilsets, device, baseline_kernel, force_all=False)
    after = _measure_kernel(coilsets, device, polygonanalytic, force_all=force_all)
    record = {
        "baseline_revision": baseline_revision,
        "head_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": str(jax.devices()[0]),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "forced_full_band": force_all,
        "before": {"module_file": "temporary git source", "coilsets": before},
        "after": {"module_file": polygonanalytic.__file__, "coilsets": after},
    }
    assert_throughput(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    record = (
        compare(args.device, args.baseline_revision, force_all=args.force_all)
        if args.compare
        else measure_branch_share(args.device, args.baseline_revision)
    )
    encoded = json.dumps(record, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
