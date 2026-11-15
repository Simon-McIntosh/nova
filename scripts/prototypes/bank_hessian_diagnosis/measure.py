"""Compare retained point jets with independently differenced source fields."""

# ruff: noqa: E402
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import argparse
import concurrent.futures
import multiprocessing

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import numpy as np

assert jax.config.jax_enable_x64
ROOT = Path.home() / ".cache/nova-hessian-diagnosis"
SOURCE = Path.home() / ".cache/nova-hessian-boundary"
STATES = (220, 333, 336)


def retain():
    """Own and hash every upstream input used by the measurement."""
    paths = [f"diagnosis/accuracy-{state}.npz" for state in STATES]
    paths += ["diagnosis/components.npz", "h200/geometry.npz", "split.json"]
    paths += ["measurement/qualified-reference.npz"]
    paths += ["source/scripts/prototypes/bank_hessian_reference/probe.py"]
    records = []
    for relative in paths:
        original, target = SOURCE / relative, ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(original, target)
        digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        assert digest == hashlib.file_digest(original.open("rb"), "sha256").hexdigest()
        records.append(dict(source=str(original), copy=str(target), sha256=digest))
    (ROOT / "inputs.json").write_text(json.dumps(records, indent=2))


def reference():
    path = ROOT / "source/scripts/prototypes/bank_hessian_reference/probe.py"
    spec = importlib.util.spec_from_file_location("retained_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stencil(task):
    """Differentiate one smooth node in both directions and retain all estimates."""
    node, point, packed, moments = task
    ref = reference()
    steps = np.array([0.001, 0.0005, 0.00025], dtype=np.longdouble)
    shifted = [np.asarray(point, dtype=np.longdouble)]
    for axis in range(2):
        for step in steps:
            for sign in (1, -1):
                p = np.asarray(point, dtype=np.longdouble).copy()
                p[axis] += sign * step
                shifted.append(p)
    values = np.concatenate(
        [
            ref.gradient(np.array(shifted[first : first + 2]), packed, moments)
            for first in range(0, len(shifted), 2)
        ],
        axis=1,
    )
    raw = (values[:, 1::2] - values[:, 2::2]).reshape(3, 2, 3, 2)
    raw /= 2 * steps[None, None, :, None]
    fourth = (4 * raw[:, :, 1:] - raw[:, :, :-1]) / 3
    sixth = (16 * fourth[:, :, 1] - fourth[:, :, 0]) / 15
    bound = 4 * np.abs(sixth - fourth[:, :, 1])
    result = dict(
        node=int(node),
        gradient=values[:, 0],
        hessian=sixth.transpose(0, 2, 1),
        bound=bound.transpose(0, 2, 1),
        raw=raw,
    )
    np.savez(ROOT / f"node-{node}.npz", **result)
    print("NODE_COMPLETE", node, flush=True)
    return int(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    retain()
    ref = reference()
    with np.load(ROOT / "h200/geometry.npz") as geometry:
        polygons = tuple(
            p[:n]
            for p, n in zip(geometry["vertices"], geometry["vertex_count"], strict=True)
        )
    packed = ref.coupling(polygons)
    split = json.loads((ROOT / "split.json").read_text())
    indices = np.array([row["node"] for row in split["nodes"] if row["interior"]])
    assert len(indices) == 64
    archives = [
        dict(np.load(ROOT / f"diagnosis/accuracy-{state}.npz")) for state in STATES
    ]
    moments = np.stack([data["moments"] for data in archives])
    selected = indices[:1] if args.probe else indices
    tasks = [
        (i, archives[0]["points"][i], packed, moments)
        for i in selected
        if not (ROOT / f"node-{i}.npz").exists()
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for node in pool.map(stencil, tasks):
            print("RETAINED", node, flush=True)
    print("STENCILS_COMPLETE", len(selected), flush=True)


if __name__ == "__main__":
    main()
