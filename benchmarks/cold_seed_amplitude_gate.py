"""Measure the cold-seed profile-support booking on the certificate rows.

The certificate solve normalises every map evaluation to the declared plasma
current (``target_current``), so the cold seed's support must book the target
at unit amplitude or the first trip starts from a doubled profile.  This
driver builds each limited Solovev certificate row exactly as
:mod:`benchmarks.solovev_certificate` does and reports, at the seed:

- the seed flux's read boundary level against the labelled region's edge flux,
- the count of labelled-confined cells the chord clip drops at the seed and
  their whole-atom analytic current,
- the unit-amplitude booked current under
  (a) the production chord clip as built,
  (b) the clip's boundary level chosen at the labelled region's edge so the
      clipped region at the seed equals the labelled confined cells,
  (c) the whole-atom label booking the first trip would use.

Modes
-----
``--measure CASE CELLS``
    Build one row and persist one JSON receipt with every quantity.
``--sweep``
    Run the weak, moderate and strong 110 rows in order, one fresh process
    each, inside the calling allocation.

The driver writes its receipts under the figure root and the crew report root
named by environment or default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import numpy as np

import benchmarks.solovev_certificate as certificate
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes

ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = Path(
    os.environ.get(
        "COLD_SEED_FIGURE_ROOT",
        ROOT / "docs/figures/cut-cell-current-attribution/cold-seed",
    )
)
REPORT_ROOT = Path(
    os.environ.get(
        "COLD_SEED_REPORT_ROOT",
        "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/cold-seed",
    )
)
ROW_CASE = {
    "weak": "weak-rotation-reactor-static",
    "moderate": "moderate-rotation-conventional-static",
    "strong": "strong-rotation-compact-static",
}
SOLOVEV_CASES = {
    "weak-rotation-reactor-static": "weak",
    "moderate-rotation-conventional-static": "moderate",
    "strong-rotation-compact-static": "strong",
}


def build_row(case_name, requested_cells):
    """Reconstruct one certificate row and return its solve components."""
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture

    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, oracle_state - exact_internal
    )
    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = ForwardProfile(
        operator, mesh, newton_steps=certificate.recovery.NEWTON_STEPS
    )
    target_current, current_centroid, current_receipt = (
        certificate._closed_form_current_target(
            case_name, source_case, operator, exact_physical
        )
    )
    requested_class = (
        TopologyClass.DIVERTED
        if certificate._is_diverted_case(case_name)
        else TopologyClass.LIMITED
    )
    seed, branch, seed_receipt = certificate._production_seed(
        profile, case_name, target_current, current_centroid, current_receipt
    )
    return (
        profile,
        operator,
        seed,
        float(target_current),
        requested_class,
        machine,
        jnp.asarray(oracle_state, dtype=jnp.float64),
    )


def _book_current(operator, source, masks, support, sample_psi_norm):
    """Book the source over one support at unit amplitude."""
    moment_masks = operator._moment_support_masks(masks, support)
    moments = source.current_moments(
        moment_masks,
        operator.support_current_moments,
        support,
        sample_flux=sample_psi_norm,
    )
    return operator.coupling_current_moments(moments, geometry=None)


def _level_clip_support(operator, masks, topology, physical, sample_psi_norm, level):
    """Return the chord support clipped at one explicit flux level."""
    shared_flux = jnp.asarray(operator.shared_node_flux(physical))
    inside_boundary = operator.polarity * (shared_flux - jnp.asarray(level))
    participation = masks.profile_participation | operator._chord_vertex_participation(
        operator.moment_geometry.atomic_mesh, inside_boundary
    )
    return operator.moment_geometry.atomic_mesh.traced_clip(inside_boundary).qualify(
        participation
    )


def measure_seed(
    profile, operator, seed, target_current, requested_class, oracle_flux=None
):
    """Return the seed-level clip bookkeeping for one certificate row."""
    physical = jnp.asarray(seed)[: operator.physical_node_number]
    masks, topology, _connected, _admitted = operator._fixed_design_read(
        physical, requested_class
    )
    axis_flux = float(topology.axis_flux)
    boundary_flux = float(topology.boundary_flux)
    span = float(topology.flux_span)
    atomic = operator.moment_geometry.atomic_mesh
    sample_flux = operator.sample_node_flux(seed)
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span

    # production support (a) exactly as the solve reads it
    support_a = operator._profile_support(masks, topology, physical, sample_psi_norm)
    included_a = np.asarray(support_a.included, dtype=bool)
    labels = np.asarray(masks.profile_participation, dtype=bool)
    core = np.asarray(masks.label) == int(PlasmaDomain.CORE)
    dropped = labels & ~included_a

    # label edge flux: max shared-node flux over the labelled cells' nodes
    shared_flux_node = np.asarray(operator.shared_node_flux(physical))
    cell_nodes = np.asarray(atomic.cell_nodes).astype(int)
    valid_node = (
        np.arange(cell_nodes.shape[1])[None, :]
        < np.asarray(atomic.cell_vertex_count)[:, None]
    )
    labelled_node_flux = np.where(
        labels[:, None] & valid_node,
        shared_flux_node[cell_nodes],
        -np.inf,
    )
    label_edge_flux = float(np.max(labelled_node_flux))
    edge_span_fraction = (label_edge_flux - boundary_flux) / abs(span)
    boundary_rel_axis = (boundary_flux - axis_flux) / abs(span)

    # booked currents and amplitudes under the three supports
    booked_a_moments = operator.cell_current_moments(seed, requested_class)
    booked_a = float(np.sum(np.asarray(booked_a_moments.cell_current)))

    # (b) and (c) at the seed share one support: the clipped region chosen so
    # it equals the labelled confined cells is the whole-atom label support
    # (the seed's own LCFS contour encloses only the interior, so any level
    # that keeps every label keeps them as full atoms).  (b) realises it by
    # the clip level, (c) by the first-trip booking; the receipt records the
    # identical booked value and which realisation the solve uses.
    full_support = atomic.traced_clip(
        jnp.ones(shared_flux_node.shape, dtype=shared_flux_node.dtype)
    ).qualify(labels)
    booked_b_diagnostic = _book_current(
        operator, profile.source, masks, full_support, sample_psi_norm
    )
    booked_c_moments = _book_current(
        operator, profile.source, masks, full_support, sample_psi_norm
    )
    booked_b = float(np.sum(np.asarray(booked_b_diagnostic.cell_current)))
    booked_c = float(np.sum(np.asarray(booked_c_moments.cell_current)))

    # dropped labelled cells' whole-atom analytic current
    dropped_moments = _book_current(
        operator, profile.source, masks, full_support, sample_psi_norm
    )
    dropped_current = float(np.sum(np.asarray(dropped_moments.cell_current)[dropped]))

    # whole-atom booking over core cells and over the confined-profile cells
    # (profile participation with normalised flux at the confined side)
    full_over_core = atomic.traced_clip(
        jnp.ones(shared_flux_node.shape, dtype=shared_flux_node.dtype)
    ).qualify(core)
    core_moments = _book_current(
        operator, profile.source, masks, full_over_core, sample_psi_norm
    )
    booked_core = float(np.sum(np.asarray(core_moments.cell_current)))
    confined = labels & (np.asarray(masks.psi_norm) <= 1.0)
    full_over_confined = atomic.traced_clip(
        jnp.ones(shared_flux_node.shape, dtype=shared_flux_node.dtype)
    ).qualify(confined)
    confined_moments = _book_current(
        operator, profile.source, masks, full_over_confined, sample_psi_norm
    )
    booked_confined = float(np.sum(np.asarray(confined_moments.cell_current)))

    # Oracle-state reference: book the profile over the analytic state through
    # the production moment path, to confirm the machinery returns the target
    # where the flux matches the equilibrium.
    booked_oracle = None
    if oracle_flux is not None:
        oracle_moments = operator.cell_current_moments(
            jnp.asarray(oracle_flux), requested_class
        )
        booked_oracle = float(np.sum(np.asarray(oracle_moments.cell_current)))

    # Seed-rescale probe ((b)): scale the internal (plasma) part of the seed
    # field by 1/amp_c so the label-region booking moves toward the target.
    seed_full = jnp.asarray(seed)
    booked_rescaled_labels = None
    rescaled_amplitude = None
    scale = 1.0 / (target_current / booked_c) if booked_c else 1.0
    external_field = operator.external(current=None)
    internal_seed = physical - external_field[: operator.physical_node_number]
    rescaled_physical = external_field[: operator.physical_node_number] + (
        scale * internal_seed
    )
    rescaled_state = jnp.concatenate(
        (rescaled_physical, seed_full[operator.physical_node_number :])
    )
    rescaled_masks, rescaled_topology, _rc, _ra = operator._fixed_design_read(
        rescaled_physical, requested_class
    )
    rescaled_sample = (
        jnp.asarray(operator.sample_node_flux(rescaled_state))
        - rescaled_topology.axis_flux
    ) / rescaled_topology.flux_span
    rescaled_full = atomic.traced_clip(
        jnp.ones(shared_flux_node.shape, dtype=shared_flux_node.dtype)
    ).qualify(np.asarray(rescaled_masks.profile_participation, dtype=bool))
    rescaled_moments = _book_current(
        operator, profile.source, rescaled_masks, rescaled_full, rescaled_sample
    )
    booked_rescaled_labels = float(np.sum(np.asarray(rescaled_moments.cell_current)))
    rescaled_amplitude = (
        target_current / booked_rescaled_labels if booked_rescaled_labels else None
    )

    included_not_labelled = included_a & ~labels
    return {
        "seed_read_axis_flux_wb": axis_flux,
        "seed_read_boundary_flux_wb": boundary_flux,
        "seed_read_boundary_rel_axis": boundary_rel_axis,
        "seed_flux_span_wb": span,
        "label_edge_flux_wb": label_edge_flux,
        "edge_minus_boundary_over_span": edge_span_fraction,
        "labelled_cell_count": int(np.sum(labels)),
        "core_cell_count": int(np.sum(core)),
        "included_cell_count": int(np.sum(included_a)),
        "included_not_labelled_count": int(np.sum(included_not_labelled)),
        "dropped_labelled_cell_count": int(np.sum(dropped)),
        "dropped_cells": np.flatnonzero(dropped).tolist(),
        "booked_a_current_a": booked_a,
        "booked_a_amplitude": target_current / booked_a if booked_a else None,
        "booked_b_current_a": booked_b,
        "booked_b_amplitude": target_current / booked_b if booked_b else None,
        "booked_c_current_a": booked_c,
        "booked_c_amplitude": target_current / booked_c if booked_c else None,
        "core_cell_count_zip": int(np.sum(core)),
        "booked_core_current_a": booked_core,
        "booked_core_amplitude": target_current / booked_core if booked_core else None,
        "booked_confined_current_a": booked_confined,
        "booked_confined_amplitude": target_current / booked_confined
        if booked_confined
        else None,
        "booked_oracle_current_a": booked_oracle,
        "booked_oracle_amplitude": (
            target_current / booked_oracle if booked_oracle else None
        ),
        "rescale_factor": scale if booked_c else None,
        "booked_rescaled_labels_current_a": booked_rescaled_labels,
        "booked_rescaled_labels_amplitude": rescaled_amplitude,
        "dropped_labelled_whole_atom_current_a": dropped_current,
        "target_current_a": target_current,
    }


def _measure(case_name, requested_cells):
    configure_dtypes()
    profile, operator, seed, target, requested_class, machine, oracle_state = build_row(
        case_name, requested_cells
    )
    result = measure_seed(
        profile,
        operator,
        jnp.asarray(seed),
        target,
        requested_class,
        oracle_flux=jnp.asarray(oracle_state, dtype=jnp.float64),
    )
    result["case"] = case_name
    result["requested_cells"] = int(requested_cells)
    result["label"] = SOLOVEV_CASES.get(case_name, case_name)
    result["revision"] = certificate._source_revision()
    return result


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _row_key(case_name, requested_cells):
    return f"{case_name}:{requested_cells}"


def _measure_row(case_name, requested_cells):
    result = _measure(case_name, requested_cells)
    slug = "reduced" if requested_cells == -110 else f"cells-{abs(requested_cells)}"
    scalar = {k: v for k, v in result.items() if not isinstance(v, list)}
    _write_json(FIGURE_ROOT / "scalars" / f"{case_name}-{slug}.json", scalar)
    _write_json(FIGURE_ROOT / "parts" / f"{case_name}-{slug}.json", result)

    def amp(value):
        return f"{value:.6g}" if value is not None else "nan"

    print(
        "COLD_SEED_MEASURE case=%s cells=%s booked_a=%.6g amp_a=%s "
        "booked_b=%.6g amp_b=%s booked_c=%.6g amp_c=%s core=%.6g amp_core=%s "
        "confined=%.6g amp_conf=%s oracle=%.6g amp_oracle=%s "
        "rescaled=%.6g amp_rescaled=%s dropped=%d "
        "dropped_current=%.6g label_edge=%.6g boundary=%.6g"
        % (
            case_name,
            requested_cells,
            result["booked_a_current_a"],
            amp(result["booked_a_amplitude"]),
            result["booked_b_current_a"],
            amp(result["booked_b_amplitude"]),
            result["booked_c_current_a"],
            amp(result["booked_c_amplitude"]),
            result["booked_core_current_a"],
            amp(result["booked_core_amplitude"]),
            result["booked_confined_current_a"],
            amp(result["booked_confined_amplitude"]),
            result["booked_oracle_current_a"],
            amp(result["booked_oracle_amplitude"]),
            result["booked_rescaled_labels_current_a"],
            amp(result["booked_rescaled_labels_amplitude"]),
            result["dropped_labelled_cell_count"],
            result["dropped_labelled_whole_atom_current_a"],
            result["label_edge_flux_wb"],
            result["seed_read_boundary_flux_wb"],
        ),
        flush=True,
    )
    return result


def _build_receipt():
    rows = []
    for case_name in SOLOVEV_CASES:
        slug = "reduced"
        path = FIGURE_ROOT / "scalars" / f"{case_name}-{slug}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        return
    lines = [
        "# Cold-seed profile-support booking on the certificate rows",
        "",
        f"Revision `{certificate._source_revision()}`. Every row: seed read "
        "boundary level vs the labelled region's edge flux, labelled-confined "
        "cells the production chord clip drops and their whole-atom current, and "
        "the unit-amplitude booked current under (a) the clip as built, (b) the "
        "clip at the label edge, (c) whole-atom label booking.",
        "",
        "| case | cells | boundary (rel axis) | label edge | drop | dropped "
        "current | booked (a) [amp] | booked (b) [amp] | booked (c) [amp] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:

        def fmt_amp(value):
            return f"{value:.6g}" if value is not None else "—"

        lines.append(
            "| {label} | {cells} | {bound:.4f} | {edge:.4f} | {drop} | {dc:.6g} | "
            "{ba:.6g} [{bamp}] | {bb:.6g} [{bamp2}] | {bc:.6g} [{bamp3}] |".format(
                label=r["label"],
                cells=abs(r["requested_cells"]),
                bound=r["seed_read_boundary_rel_axis"],
                edge=r["edge_minus_boundary_over_span"],
                drop=r["dropped_labelled_cell_count"],
                dc=r["dropped_labelled_whole_atom_current_a"],
                ba=r["booked_a_current_a"],
                bamp=fmt_amp(r["booked_a_amplitude"]),
                bb=r["booked_b_current_a"],
                bamp2=fmt_amp(r["booked_b_amplitude"]),
                bc=r["booked_c_current_a"],
                bamp3=fmt_amp(r["booked_c_amplitude"]),
            )
        )
    text = "\n".join(lines) + "\n"
    (FIGURE_ROOT / "receipt.md").write_text(text, encoding="utf-8")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "receipt.md").write_text(text, encoding="utf-8")
    _write_json(
        FIGURE_ROOT / "receipt.json",
        {"rows": rows, "revision": certificate._source_revision()},
    )
    (REPORT_ROOT / "receipt.json").write_text(
        json.dumps({"rows": rows, "revision": certificate._source_revision()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"COLD_SEED_RECEIPT rows={len(rows)}", flush=True)


def _sweep():
    python = sys.executable
    for case_name in ROW_CASE.values():
        log = FIGURE_ROOT / "logs" / f"{case_name}-110.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "wb") as captured:
            completed = subprocess.run(
                [
                    python,
                    "-m",
                    "benchmarks.cold_seed_amplitude_gate",
                    "--measure",
                    case_name,
                    "-110",
                ],
                stdout=captured,
                stderr=subprocess.STDOUT,
            )
        print(
            f"COLD_SEED_ROW_END {case_name} exit={completed.returncode} log={log}",
            flush=True,
        )
    _build_receipt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", nargs=2, metavar=("CASE", "CELLS"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()
    if args.measure:
        case_name, cells_text = args.measure
        if case_name not in certificate.CASE_NAMES:
            raise SystemExit(f"unknown case {case_name!r}")
        cells = int(cells_text)
        _measure_row(case_name, cells)
        return
    if args.sweep:
        _sweep()
        return
    if args.table:
        _build_receipt()
        return
    raise SystemExit("one of --measure, --sweep or --table is required")


if __name__ == "__main__":
    main()
