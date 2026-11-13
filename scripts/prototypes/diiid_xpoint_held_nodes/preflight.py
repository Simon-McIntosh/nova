"""Import cohort readers and validate row inputs without building or solving."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from provenance import ROOT, TRACE_HELPER, load_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import numpy as np
    from nova.equilibrium import fixed_point
    from nova.equilibrium.forward import ForwardProfile

    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    assert Path(fixed_point.__file__).resolve().is_relative_to(args.tree.resolve())

    def refuse(*_args, **_kwargs):
        raise AssertionError("profile construction and forward solves are forbidden")

    ForwardProfile.solve = refuse
    trace = load_trace()
    assert Path(trace.__file__).resolve() == TRACE_HELPER.resolve()
    try:
        load_trace(ROOT.parent / "outside-trace.py")
    except ValueError:
        print("EXTERNAL_TRACE_REFUSED", flush=True)
    else:
        raise AssertionError("external executable trace was admitted")
    spec = importlib.util.spec_from_file_location(
        "shadow_measure", Path(__file__).with_name("measure.py")
    )
    measurement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measurement)
    assert measurement.core.load_trace is load_trace
    admission = measurement.core.branch_measure.admission
    readers = {}
    for name, relative in (
        ("setup_capture", "scripts/prototypes/diiid_residual_plateau/measure.py"),
        ("setup_material", "scripts/prototypes/diiid_material_mask/measure.py"),
    ):
        module = admission.load(name, ROOT / relative)
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
        readers[name] = str(module.__file__)
    record = dict(
        frame=args.frame,
        branch=args.branch,
        module=fixed_point.__file__,
        cwd=str(Path.cwd()),
        trace=str(TRACE_HELPER),
        readers=readers,
        trace_sha256=hashlib.sha256(TRACE_HELPER.read_bytes()).hexdigest(),
    )
    if args.branch == "mast":
        from benchmarks import efit_forward_parity_slice as bench
        import zarr

        bench.build_profile = refuse
        path = bench.SHOT_STORE / "21978.zarr"
        group = zarr.open_group(str(path), mode="r")["efm"]
        radius, height, reference = bench._stored_map(group, args.frame)
        radius, height, reference, _selection = bench._benchmark_spatial_grid(
            radius, height, reference, 33
        )
        limiter = np.c_[np.asarray(group["limiterr"]), np.asarray(group["limiterz"])]
        rr, zz = np.meshgrid(radius, height, indexing="ij")
        targets = np.vstack((np.c_[rr.ravel(), zz.ravel()], limiter))
        response = bench._digest_prescribed_response_inputs(group, targets)
        geometry = (
            bench.MachineGeometryRegistry.default().select(21978).configuration.geometry
        )
        _families, drives, _mapping = bench._circuit_drives(
            group, args.frame, geometry, "fcoil_c"
        )
        for key in ("pprime", "ffprime", "ppsi_c", "fpsi_c"):
            assert np.isfinite(np.asarray(group[key][args.frame])).all()
        assert np.isfinite(reference).all() and np.isfinite(drives).all()
        assert len(bench._stored_lcfs(group, args.frame)) > 3
        record.update(
            input=str(path),
            reference_shape=list(reference.shape),
            response_input_digest=response["combined_sha256"],
        )
    else:
        from benchmarks import diiid_forward_gs_match as bench

        bench.build_profile = refuse
        selected = bench.select_frames(
            sorted(bench.DEFAULT_DATA.glob("*.parquet")),
            bench.EXECUTION_FRAME_COUNT,
            bench.polarity_population(),
        )
        match = next(item for item in selected if item.frame == args.frame)
        row = bench._read(
            match.path,
            bench._LABEL_COLUMNS
            + bench._GEOMETRY_COLUMNS
            + bench._CURRENT_COLUMNS
            + bench._PLASMA_CURRENT_COLUMNS,
        )
        row["_source_path"] = str(match.path)
        label, coordinate, pressure, diamagnetic = bench._label_state(row, args.frame)
        assert np.isfinite(label).all() and len(coordinate) >= 2
        assert np.isfinite(pressure).all() and np.isfinite(diamagnetic).all()
        wall, _receipt = bench._physical_wall_ring(
            bench.DEFAULT_MACHINE_ARTIFACT_CACHE, bench.DEFAULT_MACHINE_ARTIFACT_DIGEST
        )
        machine = bench.dataset_machine_description(row, source_row=str(match.path))
        time = float(row["efit_times"][args.frame])
        currents = bench.shipped_current_at(
            row, machine.physical, bench.POLOIDAL_CONDUCTORS, time
        )
        target = float(bench._target_current(row, time))
        assert np.isfinite(currents).all() and np.isfinite(target) and target != 0
        assert np.isfinite(wall).all() and len(wall) > 3
        record.update(
            input=str(match.path),
            reference_shape=list(label.shape),
            source_nodes=len(coordinate),
            target_current_a=target,
            wall_nodes=len(wall),
        )
    print("ROW_SETUP_READY " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
