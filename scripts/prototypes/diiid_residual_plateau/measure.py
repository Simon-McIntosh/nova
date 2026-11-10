"""Capture executed fixed-point operands without replacing solver expressions."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from nova.equilibrium import fixed_point, forward, reduced_newton
from nova.equilibrium.topology import Topology
from nova.jax.config import configure_dtypes

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_forward_convergence"))
import measure_branches as measure  # noqa: E402


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, default=lambda x: np.asarray(x).tolist()) + "\n"
    )


class Capture:
    """Host callbacks persist operands actually executed by the compiled solve."""

    def __init__(self, output):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.counts = {
            "inner": 0,
            "ladder": 0,
            "reduced_factory": 0,
            "warmup": 0,
            "linearization": 0,
        }
        self.steps = []
        self.inner = []
        self.stages = []
        self.system = None
        self.profile = None

    def install(self):
        original_inner = fixed_point._newton_krylov_inner
        original_ladder = fixed_point._backtracking_scores
        original_factory = reduced_newton._reduced_kernels
        original_residual = fixed_point._relative_residual
        original_assembly = forward.assemble_augmented_system

        def assembly(profile, *args, **kwargs):
            self.profile = profile
            self.system = original_assembly(profile, *args, **kwargs)
            return self.system

        def residual(mapped, state):
            result = original_residual(mapped, state)
            caller = inspect.currentframe().f_back
            if caller.f_code.co_filename == fixed_point.__file__:
                name = caller.f_code.co_name
                if name in ("advance", "newton_body"):
                    stage = "warmup" if name == "advance" else "linearization"
                    jax.debug.callback(
                        lambda *values: self.save_stage(stage, *values),
                        state,
                        mapped,
                        result,
                        ordered=True,
                    )
            return result

        def factory(*args, **kwargs):
            self.counts["reduced_factory"] += 1
            return original_factory(*args, **kwargs)

        def inner(map_fn, initial, **kwargs):
            result = original_inner(map_fn, initial, **kwargs)
            terminal = (
                result[0]
                if isinstance(result, tuple) and not hasattr(result, "state")
                else result
            )
            jax.debug.callback(
                self.save_inner,
                initial,
                map_fn(initial),
                terminal.state,
                map_fn(terminal.state),
                terminal.residual,
                terminal.attempted_newton_promotions,
                terminal.accepted_newton_promotions,
                terminal.termination_reason,
                ordered=True,
            )
            return result

        def ladder(*args, **kwargs):
            result = original_ladder(*args, **kwargs)
            jax.debug.callback(
                self.save_ladder,
                args[3],
                result.incumbent_merit,
                result.merits,
                result.predicted_merits,
                result.factors,
                result.ladder_selected,
                result.ladder_accepted,
                result.ladder_distrusted,
                ordered=True,
            )
            return result

        reduced_newton._reduced_kernels = factory
        fixed_point._newton_krylov_inner = inner
        fixed_point._backtracking_scores = ladder
        fixed_point._relative_residual = residual
        forward.assemble_augmented_system = assembly

    def save_stage(self, name, state, mapped, residual):
        index = self.counts[name]
        self.counts[name] += 1
        np.savez_compressed(
            self.output / f"{name}-{index}.npz", state=state, mapped=mapped
        )
        row = dict(
            stage=name,
            index=index,
            residual=float(residual),
            finite_state=bool(np.isfinite(state).all()),
            finite_image=bool(np.isfinite(mapped).all()),
            image_max=float(np.max(np.abs(mapped))),
        )
        self.stages.append(row)
        write_json(self.output / "stages.json", self.stages)
        print("STAGE " + json.dumps(row), flush=True)

    def save_inner(
        self, initial, first, state, mapped, residual, attempted, accepted, reason
    ):
        index = self.counts["inner"]
        self.counts["inner"] += 1
        np.savez_compressed(
            self.output / f"inner-{index}.npz",
            initial=initial,
            first=first,
            state=state,
            mapped=mapped,
        )
        row = dict(
            index=index,
            residual=float(residual),
            attempted=int(attempted),
            accepted=int(accepted),
            reason=int(reason),
            initial_sup=float(np.max(np.abs(first - initial))),
            terminal_sup=float(np.max(np.abs(mapped - state))),
        )
        self.inner.append(row)
        write_json(self.output / "inner.json", self.inner)
        print("INNER " + json.dumps(row), flush=True)

    def save_ladder(
        self, step, before, merits, predicted, factors, selected, accepted, distrusted
    ):
        self.counts["ladder"] += 1
        row = dict(
            index=len(self.steps),
            direction_l2=float(np.linalg.norm(step)),
            direction_sup=float(np.max(np.abs(step))),
            before=float(before),
            merits=np.asarray(merits).tolist(),
            predicted=np.asarray(predicted).tolist(),
            factors=np.asarray(factors).tolist(),
            selected=int(selected),
            accepted=bool(accepted),
            distrusted=bool(distrusted),
        )
        self.steps.append(row)
        write_json(self.output / "ladder.json", self.steps)
        print("LADDER " + json.dumps(row), flush=True)


def host_read(operator, state):
    """Unwrap only the outer topology jit and collect its actual return locals."""
    original = Topology.read_qualification
    body = original.__wrapped__
    captures = []

    def wrapped(*args, **kwargs):
        local = {}

        def trace(frame, event, value):
            if frame.f_code is body.__code__ and event == "return":
                local.update(frame.f_locals)
            return trace

        previous = sys.gettrace()
        sys.settrace(trace)
        try:
            result = body(*args, **kwargs)
            jax.block_until_ready(result)
        finally:
            sys.settrace(previous)
        captures.append(local)
        return result

    Topology.read_qualification = wrapped
    try:
        result = operator._fixed_design_read(jnp.asarray(state))
    finally:
        Topology.read_qualification = original
    return captures[-1], result


def split_residual(residual, mapped, masks):
    scale = max(float(np.max(np.abs(mapped))), 1e-30)
    total = float(np.sum(residual**2))
    rows = []
    for name, selected in masks.items():
        values = residual[selected]
        rows.append(
            dict(
                block=name,
                count=int(selected.sum()),
                sup_wb=float(np.max(np.abs(values), initial=0)),
                relative_sup=float(np.max(np.abs(values), initial=0) / scale),
                l2_wb=float(np.linalg.norm(values)),
                squared_norm_fraction=float(np.sum(values**2) / max(total, 1e-300)),
            )
        )
    return rows


def prepare_mast():
    """Reuse the fitted-circuit MAST reference builder for shot 21978/35."""
    from benchmarks import efit_forward_parity_slice as mast

    case, context = mast._mast_case_from_selection(
        mast.SHOT_STORE, {"shot": 21978, "slice_index": 35}, None, grid_points=33
    )
    case, profile, _policy = mast._passive_inclusive_case(case, context)
    group = context["group"]
    limiter = np.c_[np.asarray(group["limiterr"]), np.asarray(group["limiterz"])]
    contour = np.asarray(case["boundary"])
    return dict(
        profile=profile,
        seed=case["state"],
        current=np.asarray(profile.operator.external_current),
        target_current_a=abs(case["reference"]["plasma_current_a"]),
        label=context["reference_flux"],
        axis=case["axis"],
        saddle=contour[np.argmin(contour[:, 1])],
        contour=contour,
        limiter=limiter,
    )


def run(frame, branch, output, machine="diiid"):
    configure_dtypes()
    assert jax.config.jax_enable_x64
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        json.dumps(
            dict(
                revision=revision,
                cwd=str(Path.cwd().resolve()),
                module=fixed_point.__file__,
                backend=jax.default_backend(),
            )
        ),
        flush=True,
    )
    prepared = (
        prepare_mast()
        if machine == "mast"
        else measure._prepare_frame(
            frame, float(measure.bench.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION)
        )
    )
    profile = prepared["profile"]
    kind = measure.TopologyClass[branch.upper()]
    seed = (
        prepared["seed"]
        if machine == "mast"
        else prepared["cold"].branches.flux[int(kind)]
    )
    request = measure.bench._registered_solve_request(
        profile,
        seed,
        jnp.asarray(prepared["current"]),
        prepared["target_current_a"],
        carrier_identity=f"plateau:{frame}:{branch}",
    )
    capture = Capture(output)
    capture.install()
    print(
        "REQUEST " + json.dumps(dict(route=request.route, policy=repr(request.policy))),
        flush=True,
    )
    receipt = profile.solve(request)
    jax.block_until_ready(receipt.equilibrium.flux)
    jax.effects_barrier()
    state = np.asarray(receipt.equilibrium.flux)
    profile = capture.profile or profile
    local, read = host_read(profile.operator, state)
    system = capture.system
    unknowns = jnp.concatenate(
        tuple(
            jnp.ravel(record.normalized_unknown)
            for record in receipt.equilibrium.constraints
        )
    )
    augmented = jnp.concatenate((jnp.asarray(state) / system.flux_scale, unknowns))
    augmented_image = jax.jit(system.map_fn)(augmented)
    mapped = np.asarray(augmented_image[: state.size] * system.flux_scale)
    constraint_residual = np.asarray(augmented_image[state.size :] - unknowns)
    residual = mapped - state
    lattice = profile.lattice
    count = lattice.node_count
    physical = profile.operator.physical_node_number
    connected = np.asarray(local["connected"], bool).reshape(lattice.shape)
    band = binary_dilation(connected) & ~binary_erosion(connected)
    masks = {}
    for name, region in (
        ("support_interior", connected & ~band),
        ("boundary_band", band),
        ("outside_plasma", ~connected & ~band),
    ):
        mask = np.zeros(state.size, bool)
        mask[:count] = region.ravel()
        masks[name] = mask
    for name, start, end in (
        ("wall_flux", count, physical),
        ("direct_sample_flux", physical, state.size),
    ):
        mask = np.zeros(state.size, bool)
        mask[start:end] = True
        masks[name] = mask
    shadow = np.asarray(profile.operator.residual_shadow_mask(jnp.asarray(state)), bool)
    fixed = receipt.equilibrium.fixed_point
    limiter = (
        prepared["limiter"]
        if machine == "mast"
        else measure.bench._physical_wall_ring(
            measure.bench.DEFAULT_MACHINE_ARTIFACT_CACHE,
            measure.bench.DEFAULT_MACHINE_ARTIFACT_DIGEST,
        )[0]
    )
    arrays = dict(
        state=state,
        mapped=mapped,
        residual=residual,
        shadow=shadow,
        connected=connected,
        boundary_band=band,
        flux=state[:count].reshape(lattice.shape),
        radius=lattice.radius,
        height=lattice.height,
        label=prepared["label"],
        label_boundary=prepared["contour"],
        reference_axis=prepared["axis"],
        reference_saddle=prepared["saddle"],
        limiter=limiter,
        axis=local["data_o"],
        saddle=local["data_x"],
        boundary=local["data_b"],
    )
    np.savez_compressed(output / "terminal.npz", **arrays)
    row = dict(
        machine=machine,
        frame=frame,
        branch=branch,
        base_revision=revision,
        route=request.route,
        module=fixed_point.__file__,
        storage_order="radius_height",
        residual=float(fixed.residual),
        converged=bool(fixed.converged),
        recomputed_residual=float(
            fixed_point._relative_residual(jnp.asarray(mapped), jnp.asarray(state))
        ),
        support=int(connected.sum()),
        shadow_count=int(shadow.sum()),
        attempted=int(receipt.attempted_newton_promotions),
        accepted=int(receipt.accepted_newton_promotions),
        residual_history=np.asarray(receipt.residual_history).tolist(),
        changed_mask_entries=np.asarray(receipt.changed_mask_entries).tolist(),
        block_region=split_residual(residual, mapped, masks),
        coil_rows=0,
        constraint_rows=int(unknowns.size),
        constraint_residual=constraint_residual.tolist(),
        augmented_residual=float(
            fixed_point._relative_residual(augmented_image, augmented)
        ),
        counts=capture.counts,
        inner=capture.inner,
        last_ladder=capture.steps[-1] if capture.steps else None,
        terminal_reason=int(receipt.termination_reason),
    )
    write_json(output / "receipt.json", row)
    print("SUMMARY " + json.dumps(row), flush=True)
    assert capture.counts["inner"] > 0, "actual fixed-point seam did not execute"
    assert connected.sum() > 0, "support-collapse premise reproduced"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--branch", choices=("limited", "diverted"))
    parser.add_argument("--machine", choices=("diiid", "mast"), default="diiid")
    args = parser.parse_args()
    if args.frame is not None:
        run(args.frame, args.branch, args.output, args.machine)
    else:
        for frame, branch in ((179, "limited"), (137, "diverted")):
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--frame",
                    str(frame),
                    "--branch",
                    branch,
                    "--output",
                    str(args.output / f"frame-{frame}-{branch}"),
                ],
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--machine",
                "mast",
                "--frame",
                "35",
                "--branch",
                "diverted",
                "--output",
                str(args.output / "mast-21978-35"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
