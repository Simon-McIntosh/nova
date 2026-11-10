"""Measure branch retention on governed-wall seeds and continued lower roots."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_warmup_admission"))
import measure as admission  # noqa: E402

original_load = admission.load
original_row = admission.row
original_batch = admission.batch


def row(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from nova.equilibrium import fixed_point
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64
    instances, branches, seeds = [], [], []

    def load(name, path):
        module = original_load(name, path)
        if name == "operand_capture":
            source = inspect.getsource(module.run)
            if "receipt.mask_history" in source:
                assert source.count("receipt.mask_history") == 1
                source = source.replace(
                    "receipt.mask_history", "receipt.changed_mask_entries"
                )
                padding = "\n" * (module.run.__code__.co_firstlineno - 1)
                exec(compile(padding + source, str(path), "exec"), module.__dict__)
            print("COLLECTOR_RECEIPT_FIELD=changed_mask_entries", flush=True)
        if name == "material_measure" and args.branch == "diverted":
            prepare = module.probe._prepare_frame

            def admitted_seed(frame, expansion):
                prepared = prepare(frame, expansion)
                operator = prepared["profile"].operator
                cold = np.asarray(prepared["cold"].branches.flux[1])
                label = np.asarray(prepared["label_seed"])

                @jax.jit
                def inspect(state):
                    _mask, topology, connected, admitted = operator._fixed_design_read(
                        state
                    )
                    return topology, connected, admitted

                candidates = [("cold", cold), ("label", label)] + [
                    (f"label_fraction_{fraction}", cold + fraction * (label - cold))
                    for fraction in (0.75, 0.5, 0.25)
                ]
                selected, choices = None, []
                for name, state in candidates:
                    topology, connected, admitted = jax.device_get(
                        inspect(jnp.asarray(state))
                    )
                    valid = bool(
                        admitted
                        and topology.boundary_is_xpoint
                        and np.isfinite(topology.x_point).all()
                        and topology.x_point[1] < topology.axis[1]
                    )
                    choices.append(
                        dict(
                            candidate=name,
                            admitted_lower=valid,
                            boundary_class="diverted"
                            if topology.boundary_is_xpoint
                            else "limited",
                            support=int(connected.sum()),
                            saddle_rz_m=np.asarray(topology.x_point).tolist(),
                        )
                    )
                    # Prefer the admitted label, then a cold or intermediate field.
                    if valid and (selected is None or name == "label"):
                        selected = (name, state)
                    if selected is not None and name != "cold":
                        break
                name, state = (
                    selected if selected is not None else ("cold_unadmitted", cold)
                )
                prepared["cold"] = prepared["cold"]._replace(
                    branches=prepared["cold"].branches._replace(
                        flux=prepared["cold"]
                        .branches.flux.at[1]
                        .set(jnp.asarray(state))
                    )
                )
                seeds.append(
                    dict(
                        selected=name,
                        admitted_lower=selected is not None,
                        sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                        candidates=choices,
                    )
                )
                print("SEED_ADMISSION " + json.dumps(seeds[-1]), flush=True)
                return prepared

            module.probe._prepare_frame = admitted_seed
        if name == "operand_capture":
            original_capture = module.Capture

            class BranchCapture(original_capture):
                def __init__(self, output):
                    super().__init__(output)
                    instances.append(self)

                def install(self):
                    super().install()
                    from nova.equilibrium.forward import ForwardProfile

                    solve = ForwardProfile.solve

                    def persisted(profile, *a, **kw):
                        request = a[0] if a else kw["request"]
                        state = np.asarray(request.seed_policy.state)
                        seed_hash = hashlib.sha256(state.tobytes()).hexdigest()
                        if seeds:
                            assert len(seeds) == 1 and seeds[0]["sha256"] == seed_hash
                        else:
                            seeds.append(dict(selected="request", sha256=seed_hash))
                        print("REQUEST_SEED_SHA256=" + seed_hash, flush=True)
                        receipt = solve(profile, *a, **kw)
                        fixed = receipt.equilibrium.fixed_point
                        saved = dict(
                            revision=args.revision,
                            module=fixed_point.__file__,
                            cwd=str(Path.cwd().resolve()),
                            residual=float(fixed.residual),
                            converged=bool(fixed.converged),
                            terminal_reason=int(receipt.termination_reason),
                            attempted=int(receipt.attempted_newton_promotions),
                            accepted=int(receipt.accepted_newton_promotions),
                            residual_history=np.asarray(
                                receipt.residual_history
                            ).tolist(),
                            changed_mask_entries=np.asarray(
                                receipt.changed_mask_entries
                            ).tolist(),
                            seed_admission=seeds,
                            branch_refusals=branches,
                            warmup_refused_proposals=np.asarray(
                                fixed.warmup_refused_proposals
                            ).tolist(),
                            warmup_first_nonfinite_index=np.asarray(
                                fixed.warmup_first_nonfinite_index
                            ).tolist(),
                        )
                        pending = self.output / "solve-receipt.pending.json"
                        pending.write_text(json.dumps(saved, indent=2) + "\n")
                        pending.replace(self.output / "solve-receipt.json")
                        print("DURABLE_SOLVE_RECEIPT " + json.dumps(saved), flush=True)
                        return receipt

                    ForwardProfile.solve = persisted
                    inner = fixed_point._newton_krylov_inner

                    def measured(*a, **kw):
                        result = inner(*a, **kw)
                        terminal = result if hasattr(result, "state") else result[0]
                        names = (
                            "warmup_branch_refused_proposals",
                            "warmup_first_branch_refusal_index",
                            "newton_branch_refused_proposals",
                            "newton_first_branch_refusal_index",
                        )
                        values = [
                            getattr(
                                terminal,
                                name,
                                jnp.array([0 if "proposals" in name else -1]),
                            )
                            for name in names
                        ]
                        jax.debug.callback(
                            lambda *arrays: branches.append(
                                {
                                    name: np.asarray(array).tolist()
                                    for name, array in zip(names, arrays, strict=True)
                                }
                            ),
                            *values,
                            ordered=True,
                        )
                        return result

                    fixed_point._newton_krylov_inner = measured

            module.Capture = BranchCapture
        return module

    admission.load = load
    original_row(args)
    capture = instances[0]
    profile = capture.profile
    reader_connected = hasattr(capture.system.shadowed_map_fn, "_read_branch_topology")
    if hasattr(fixed_point.FixedPointResult, "newton_branch_refused_proposals"):
        assert reader_connected, "augmented route lost its topology reader"
    record_path = args.output / "receipt.json"
    record = json.loads(record_path.read_text())
    assert len(branches) == len(record["inner"]) > 0, (
        "missing inner branch-counter callbacks"
    )
    record["augmented_topology_reader_connected"] = reader_connected
    with np.load(args.output / "terminal.npz") as archive:
        data = {name: archive[name] for name in archive.files}
    enrich(
        args.output,
        profile,
        float(capture.system.flux_scale),
        record,
        data,
        branches,
        seeds,
    )


def enrich(output, profile, scale, record, data, branches, seeds):
    """Read saved terminal and trip states without advancing a forward solve."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
    from nova.linalg.tensor_spline import fit_tensor_spline
    from nova.media.poloidal import inside_wall_units

    operator = profile.operator
    reference = np.asarray(record["efit_x_point_rz_m"])
    count = profile.lattice.node_count
    flux_size = data["state"].size

    @jax.jit
    def inspect(state):
        _mask, topology, connected, admitted = operator._fixed_design_read(state)
        return topology, connected, admitted

    @jax.jit
    def lower_root(state, starts):
        field = fit_tensor_spline(
            jnp.asarray(data["radius"]),
            jnp.asarray(data["height"]),
            state[:count].reshape(data["flux"].shape).T,
        )
        return polish_stationary_points(field, starts, jnp.ones(starts.shape[0], bool))

    def topology_record(state):
        topology, connected, admitted = jax.device_get(inspect(jnp.asarray(state)))
        starts = np.vstack((reference, np.asarray(topology.x_point)))
        roots = jax.device_get(lower_root(jnp.asarray(state), jnp.asarray(starts)))
        positions = np.asarray(roots["position_rz"])
        valid = (
            np.asarray(roots["converged"])
            & (np.asarray(roots["hessian_type"]) == -1)
            & (positions[:, 1] < topology.axis[1])
        )
        distances = np.linalg.norm(positions - reference, axis=1)
        index = int(np.argmin(np.where(valid, distances, np.inf)))
        position = positions[index]
        qualified = bool(valid.any())
        return dict(
            boundary_class="diverted" if topology.boundary_is_xpoint else "limited",
            support=int(connected.sum()),
            axis_admitted=bool(admitted),
            selected_saddle_rz_m=np.asarray(topology.x_point).tolist(),
            lower_saddle_qualified=qualified,
            lower_saddle_probe_converged=np.asarray(roots["converged"]).tolist(),
            lower_saddle_probe_qualified=valid.tolist(),
            lower_saddle_probe_gradient_norm=np.asarray(
                roots["gradient_norm"]
            ).tolist(),
            lower_saddle_probe_positions_rz_m=positions.tolist(),
            lower_saddle_rz_m=position.tolist() if qualified else None,
            lower_saddle_flux_wb=float(roots["value"][index]) if qualified else None,
            lower_saddle_inside_wall=bool(
                inside_wall_units(position[None], data["limiter"])[0]
            )
            if qualified
            else None,
            lower_saddle_distance_m=float(np.linalg.norm(position - reference))
            if qualified
            else None,
        )

    trips = []
    for number, inner in enumerate(record["inner"]):
        with np.load(output / f"inner-{number}.npz") as values:
            initial = np.asarray(values["initial"][:flux_size] * scale)
            returned = np.asarray(values["state"][:flux_size] * scale)
        trips.append(
            dict(
                trip=number,
                initial=topology_record(initial),
                returned=topology_record(returned),
                **inner,
            )
        )
        (output / "trips.json").write_text(json.dumps(trips, indent=2) + "\n")
    terminal = topology_record(data["state"])
    assert terminal["support"] == record["support"], (
        "terminal support control disagrees"
    )
    if terminal["lower_saddle_qualified"]:
        data["continued_saddle"] = np.r_[
            terminal["lower_saddle_rz_m"], terminal["lower_saddle_flux_wb"]
        ]
        np.savez_compressed(output / "terminal.npz", **data)
    record.update(
        terminal_topology=terminal,
        trip_topology=trips,
        branch_refusals=branches,
        seed_admission=seeds,
        warmup_branch_refused_proposals=sum(
            sum(item["warmup_branch_refused_proposals"]) for item in branches
        ),
        newton_branch_refused_proposals=sum(
            sum(item["newton_branch_refused_proposals"]) for item in branches
        ),
    )
    (output / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    print("BRANCH_SUMMARY " + json.dumps(record), flush=True)


def batch(args):
    import numpy as np

    original_batch(args)
    records = json.loads((args.output / "comparison.json").read_text())
    paired = {(r["arm"], r["case"]): r for r in records}
    verdict = {}
    seed_pairs = {}
    for case in {case for _arm, case in paired}:
        base, head = (paired[arm, case] for arm in ("base", "head"))
        assert len(base["seed_admission"]) == len(head["seed_admission"]) == 1
        seed_hash = base["seed_admission"][0]["sha256"]
        assert seed_hash == head["seed_admission"][0]["sha256"]
        assert head["augmented_topology_reader_connected"] is True
        seed_pairs[case] = seed_hash
    assert len(seed_pairs) == 5
    verdict["paired_seed_sha256"] = seed_pairs
    verdict["augmented_topology_reader_connected"] = True
    for frame, limit in ((137, 0.018), (146, 0.005)):
        name = f"frame-{frame}-diverted"
        base, head = (paired[arm, name] for arm in ("base", "head"))
        assert (
            base["seed_admission"][0]["sha256"] == head["seed_admission"][0]["sha256"]
        )
        topology = head["terminal_topology"]
        distance = topology["lower_saddle_distance_m"]
        verdict[name] = bool(
            topology["boundary_class"] == "diverted"
            and distance is not None
            and distance <= limit
        )
    base, head = (paired[arm, "mast-21978-35"] for arm in ("base", "head"))
    with (
        np.load(Path(base["source_path"]) / "terminal.npz") as b,
        np.load(Path(head["source_path"]) / "terminal.npz") as h,
    ):
        difference = float(np.max(np.abs(b["state"] - h["state"])))
        scale = max(1.0, float(np.max(np.abs(b["state"]))))
        verdict["mast_max_abs_state_difference_wb"] = difference
        verdict["mast_unchanged_to_roundoff"] = bool(
            difference <= 8 * np.finfo(float).eps * scale
        )
    verdict["passed"] = bool(
        verdict["frame-137-diverted"]
        and verdict["frame-146-diverted"]
        and verdict["mast_unchanged_to_roundoff"]
    )
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print("COHORT_VERDICT " + json.dumps(verdict), flush=True)


if __name__ == "__main__":
    admission.row = row
    admission.batch = batch
    admission.__file__ = __file__
    admission.CASES = (
        (137, "diverted"),
        (146, "diverted"),
        (146, "limited"),
        (179, "limited"),
        (35, "mast"),
    )
    admission.main()
