"""Bind count receipts to the selected seeds, bank arrays and source."""

import hashlib
import json
import shutil

import jax
import numpy as np


def array_hash(values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        assert not array.dtype.hasobject
        descriptor = json.dumps([array.dtype.str, list(array.shape)]).encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def provenance(root, initial, stationary, bank_inputs, width):
    source_paths = sorted((root / "nova/equilibrium").glob("*.py"))
    source_paths += [
        root / name
        for name in (
            "nova/jax/config.py",
            "scripts/prototypes/bank_warm_predictor/measure.py",
            "scripts/prototypes/bank_warm_predictor/program.py",
            "scripts/prototypes/bank_warm_predictor/receipts.py",
            "scripts/prototypes/bank_warm_start/streams.py",
            "scripts/prototypes/bank_solve_profile/fixture.py",
            "scripts/prototypes/bank_newton_step/measure.py",
            "scripts/analytic_oracle_fixtures/measure.py",
            "benchmarks/solovev_certificate.py",
            "tests/equilibrium/test_topology_read.py",
            "tests/rotating_equilibrium_references.py",
        )
    ]
    inputs = dict(
        width=width,
        steps=[1e-5, 1e-3],
        slices=list(range(1, 8)),
        seed_sha256=array_hash((*initial, stationary)),
        bank_sha256=array_hash(jax.tree_util.tree_leaves(bank_inputs)),
        bank_structure=str(jax.tree_util.tree_structure(bank_inputs)),
        sources={
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        jax=jax.__version__,
        numpy=np.__version__,
        x64=bool(jax.config.jax_enable_x64),
        backend=jax.default_backend(),
        device_kind=jax.devices()[0].device_kind,
    )
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return dict(
        inputs_hash=hashlib.sha256(encoded).hexdigest(),
        inputs=inputs,
        completed=False,
        rows=0,
    )


def write_metadata(stem, metadata):
    destination = stem.with_suffix(".inputs.json")
    pending = destination.with_name(destination.name + ".pending")
    pending.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    pending.replace(destination)


def persist(stem, rows, arrays, metadata, *, complete):
    for suffix in (".json", ".npz"):
        destination = stem.with_suffix(suffix)
        pending = destination.with_name(destination.name + ".pending")
        if suffix == ".json":
            pending.write_text(json.dumps(rows, indent=2))
        else:
            with pending.open("wb") as output:
                np.savez(output, **arrays)
        pending.replace(destination)
    metadata.update(
        completed=complete,
        rows=len(rows),
        receipt_sha256={
            suffix: hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest()
            for suffix in (".json", ".npz")
        },
    )
    write_metadata(stem, metadata)


def reuse(source, destination, expected):
    """Reuse only a complete receipt with matching inputs and intact files."""
    sidecar = source.with_suffix(".inputs.json")
    if not sidecar.exists():
        print("REUSE refused: no recorded inputs hash at", sidecar, flush=True)
        return False
    recorded = json.loads(sidecar.read_text())
    if not recorded.get("completed") or recorded.get("rows") != 14:
        print("REUSE refused: receipt is incomplete", flush=True)
        return False
    if recorded.get("inputs_hash") != expected["inputs_hash"]:
        print("REUSE refused: input hashes differ", flush=True)
        return False
    for suffix in (".json", ".npz"):
        path = source.with_suffix(suffix)
        if not path.exists() or hashlib.sha256(
            path.read_bytes()
        ).hexdigest() != recorded.get("receipt_sha256", {}).get(suffix):
            print("REUSE refused: receipt digest mismatch", suffix, flush=True)
            return False
    for suffix in (".json", ".npz"):
        shutil.copyfile(source.with_suffix(suffix), destination.with_suffix(suffix))
    recorded["reused_from"] = str(source)
    write_metadata(destination, recorded)
    print("REUSE accepted:", recorded["inputs_hash"], flush=True)
    return True
