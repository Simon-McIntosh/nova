"""Classify the stored closed-form and alternate equilibrium roots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from nova.equilibrium.topology import boundary_mode
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures.measure import (
    FIXTURE_REQUESTS,
    TOTAL_FLUX_FACTOR,
    WALL_POINT_COUNT,
    analytic_case,
    cached_machine,
    forward_operator,
)
from scripts.dual_basin_fixtures.build_diverted_fixture import (
    BANK_PATH as DIVERTED_BANK_PATH,
    RECEIPT_PATH as DIVERTED_RECEIPT_PATH,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


OUTPUT = Path(__file__).resolve().parent
REPOSITORY_ROOT = OUTPUT.parents[1]
BANK = REPOSITORY_ROOT / "scripts" / "oracle_rebaseline"
RESOLUTIONS = ("coarse", "fine")
ROOTS = (
    ("closed_form_analytic", "oracle_state"),
    ("alternate_fixed_point", "root_state"),
)


def _digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _point(values) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        return None
    return array.tolist()


def _finite_rows(values) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array[:, 0])].tolist()


def _strict_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _root_read(operator, state: np.ndarray, wall: np.ndarray) -> dict[str, object]:
    grid_flux, wall_flux = operator.topology.split_flux_map(jnp.asarray(state))
    extrema, saddles = operator.topology.grid(grid_flux)
    topology_masks, topology = operator.read(jnp.asarray(state))
    del topology_masks
    finite_saddles = _finite_rows(saddles)
    contact = np.asarray(topology.wall_point, dtype=np.float64)
    return {
        "class": boundary_mode(topology).value,
        "diverted": bool(topology.diverted),
        "axis_m": _point(topology.axis),
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_point_m": _point(topology.boundary),
        "boundary_flux_wb": float(topology.boundary_flux),
        "wall_contact_point_m": _point(topology.wall_point),
        "wall_contact_flux_wb": float(topology.wall_point_flux),
        "boundary_wall_point_distance_m": float(
            np.linalg.norm(np.asarray(topology.boundary) - contact)
        ),
        "boundary_wall_flux_difference_wb": float(
            topology.boundary_flux - topology.wall_point_flux
        ),
        "wall_contact_nearest_vertex_distance_m": float(
            np.min(np.linalg.norm(wall - contact, axis=1))
        ),
        "finite_o_point_count": len(_finite_rows(extrema)),
        "finite_o_points": _finite_rows(extrema),
        "finite_x_point_count": len(finite_saddles),
        "finite_x_points": finite_saddles,
        "selected_x_point_m": _point(topology.x_point),
        "selected_x_point_flux_wb": (
            None
            if not np.isfinite(topology.x_point_flux)
            else float(topology.x_point_flux)
        ),
        "state_precision": str(state.dtype),
        "null_fit_precision": str(operator.topology.grid.fit_dtype),
        "wall_state_precision": str(np.asarray(wall_flux).dtype),
    }


def _analytic_stationary_points() -> dict[str, object]:
    case = analytic_case()
    coordinate_label = -(case.major_radius**2)
    saddle_hessian_per_radian = np.array(
        [
            -2.0 * case._flux_offset_derivative(coordinate_label),
            -2.0 * case.field_coefficient,
        ],
        dtype=np.float64,
    )
    axis_hessian_per_radian = np.array(
        [
            -4.0 * case.major_radius**2 * case.pressure_coefficient,
            -2.0 * case.field_coefficient,
        ],
        dtype=np.float64,
    )
    return {
        "physical_domain": "R > 0",
        "derivation": (
            "dpsi/dZ=-2*k_z*Z and dpsi/dR=-2*R*G'(R^2-R0^2); "
            "for R>0, G' vanishes only at R=R0"
        ),
        "physical_stationary_points": [
            {
                "coordinate_m": [case.major_radius, 0.0],
                "kind": "maximum_magnetic_axis",
                "hessian_eigenvalues_per_radian_wb_per_m2": (
                    axis_hessian_per_radian.tolist()
                ),
                "hessian_eigenvalues_total_flux_wb_per_m2": (
                    TOTAL_FLUX_FACTOR * axis_hessian_per_radian
                ).tolist(),
            }
        ],
        "coordinate_axis_stationary_point": {
            "coordinate_m": [0.0, 0.0],
            "kind": "saddle",
            "physical_x_point": False,
            "reason": (
                "the cylindrical coordinate axis is excluded from the R>0 plasma "
                "domain and lies outside every banked grid and wall"
            ),
            "flux_per_radian_wb": float(case.flux(0.0, 0.0)),
            "flux_total_wb": float(TOTAL_FLUX_FACTOR * case.flux(0.0, 0.0)),
            "hessian_eigenvalues_per_radian_wb_per_m2": (
                saddle_hessian_per_radian.tolist()
            ),
            "hessian_eigenvalues_total_flux_wb_per_m2": (
                TOTAL_FLUX_FACTOR * saddle_hessian_per_radian
            ).tolist(),
            "hessian_determinant_per_radian_squared": float(
                np.prod(saddle_hessian_per_radian)
            ),
        },
        "physical_x_point_count": 0,
        "physical_topology": "limited",
    }


def measure() -> tuple[dict[str, object], dict[str, object]]:
    configure_dtypes()
    case = analytic_case()
    analytic = _analytic_stationary_points()
    fixtures: dict[str, object] = {}
    plot_data: dict[str, object] = {}
    for resolution in RESOLUTIONS:
        receipt_path = BANK / f"receipt-{resolution}.json"
        bank_path = BANK / f"root-{resolution}.npz"
        source_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        machine = cached_machine(
            case, FIXTURE_REQUESTS[resolution], wall_nodes=WALL_POINT_COUNT
        )
        if not machine.cache["hit"]:
            raise RuntimeError(
                f"{resolution} semantic carrier was not a warm cache hit"
            )
        operator = forward_operator(case, machine)
        stored_roots: dict[str, object] = {}
        plot_roots: dict[str, object] = {}
        with np.load(bank_path, allow_pickle=False) as stored:
            for root_name, array_name in ROOTS:
                state = np.asarray(stored[array_name])
                expected = source_receipt["root_artifact"]["arrays"][array_name]
                identity = {
                    "shape": list(state.shape),
                    "dtype": state.dtype.str,
                    "sha256": _digest(state),
                }
                if identity != expected:
                    raise ValueError(
                        f"{resolution} {array_name} does not match its banked identity"
                    )
                topology = _root_read(operator, state, machine.wall_node)
                if topology["class"] != "limited":
                    raise AssertionError(
                        f"{resolution} {root_name} unexpectedly read "
                        f"{topology['class']}"
                    )
                if topology["finite_x_point_count"] != 0:
                    raise AssertionError(
                        f"{resolution} {root_name} unexpectedly carries an X-point"
                    )
                stored_roots[root_name] = {
                    "array": array_name,
                    "identity": identity,
                    "topology": topology,
                }
                plot_roots[root_name] = topology
        oracle_axis = np.asarray(
            stored_roots["closed_form_analytic"]["topology"]["axis_m"]
        )
        alternate_axis = np.asarray(
            stored_roots["alternate_fixed_point"]["topology"]["axis_m"]
        )
        coordinate_saddle = np.array([0.0, 0.0])
        fixtures[resolution] = {
            "source_receipt": str(receipt_path.relative_to(REPOSITORY_ROOT)),
            "source_bank": str(bank_path.relative_to(REPOSITORY_ROOT)),
            "requested_cells": source_receipt["requested_cells"],
            "realised_cells": source_receipt["realised_cells"],
            "cache_semantic_key": machine.cache["semantic_key"],
            "cache_warm_hit": bool(machine.cache["hit"]),
            "grid_node_bounds_m": {
                "minimum": np.min(machine.node, axis=0).tolist(),
                "maximum": np.max(machine.node, axis=0).tolist(),
            },
            "wall_vertex_bounds_m": {
                "minimum": np.min(machine.wall_node, axis=0).tolist(),
                "maximum": np.max(machine.wall_node, axis=0).tolist(),
            },
            "coordinate_axis_saddle_nearest_grid_node_distance_m": float(
                np.min(np.linalg.norm(machine.node - coordinate_saddle, axis=1))
            ),
            "coordinate_axis_saddle_nearest_wall_vertex_distance_m": float(
                np.min(np.linalg.norm(machine.wall_node - coordinate_saddle, axis=1))
            ),
            "source_receipt_topology_labels": {
                "closed_form_analytic": source_receipt["oracle_topology"]["class"],
                "alternate_fixed_point": source_receipt["root_topology"]["class"],
            },
            "roots": stored_roots,
            "axis_separation_m": float(np.linalg.norm(alternate_axis - oracle_axis)),
            "alternate_terminal_residual": source_receipt["terminal_root"][
                "terminal_residual"
            ],
        }
        plot_data[resolution] = {
            "nodes": machine.node,
            "wall": machine.wall_node,
            "roots": plot_roots,
        }
    diverted_receipt = json.loads(DIVERTED_RECEIPT_PATH.read_text(encoding="utf-8"))
    diverted_arrays: dict[str, np.ndarray] = {}
    with np.load(DIVERTED_BANK_PATH, allow_pickle=False) as stored:
        if set(stored.files) != set(diverted_receipt["arrays"]):
            raise ValueError("diverted bank array names do not match its receipt")
        for array_name in stored.files:
            values = np.asarray(stored[array_name])
            identity = {
                "shape": list(values.shape),
                "dtype": values.dtype.str,
                "sha256": _digest(values),
            }
            if identity != diverted_receipt["arrays"][array_name]:
                raise ValueError(
                    f"diverted {array_name} does not match its banked identity"
                )
            diverted_arrays[array_name] = values
    machine = cached_machine(
        case, FIXTURE_REQUESTS["fine"], wall_nodes=WALL_POINT_COUNT
    )
    if not machine.cache["hit"]:
        raise RuntimeError("fine semantic carrier was not a warm cache hit")
    operator = forward_operator(case, machine)
    diverted_topology = _root_read(
        operator, diverted_arrays["state"], machine.wall_node
    )
    if diverted_topology["class"] != "diverted":
        raise AssertionError("banked diverted oracle did not read diverted")
    if diverted_topology["finite_x_point_count"] != 1:
        raise AssertionError("banked diverted oracle did not carry exactly one X-point")
    receipt_topology = diverted_receipt["stored_precision_production_read"]
    if (
        diverted_topology["selected_x_point_m"]
        != receipt_topology["selected_x_point_m"]
    ):
        raise AssertionError("diverted classification does not reproduce its receipt")
    analytic_x = np.asarray(
        diverted_receipt["analytic_stationary_points"]["x_point"]["coordinate_m"]
    )
    locator_error = float(
        np.linalg.norm(np.asarray(diverted_topology["selected_x_point_m"]) - analytic_x)
    )
    diverted_fixture = {
        "source_receipt": str(DIVERTED_RECEIPT_PATH.relative_to(REPOSITORY_ROOT)),
        "source_bank": str(DIVERTED_BANK_PATH.relative_to(REPOSITORY_ROOT)),
        "arrays": diverted_receipt["arrays"],
        "carrier": diverted_receipt["carrier"],
        "closed_form": diverted_receipt["closed_form"],
        "analytic_stationary_points": diverted_receipt["analytic_stationary_points"],
        "production_x_point_localization_error_m": locator_error,
        "topology": diverted_topology,
    }
    plot_data["fine"]["roots"]["diverted_analytic"] = diverted_topology
    report = {
        "schema": "nova.oracle-topology-classification",
        "schema_version": 2,
        "measurement": {
            "operation": "read-only classification of serialized root arrays",
            "solver_executed": False,
            "receipt_labels_used_as_input": False,
            "jax_backend": jax.default_backend(),
            "jax_x64_enabled": bool(jax.config.x64_enabled),
            "state_precision": "binary64",
            "topology_locator_precision": "float64",
            "flux_gauge": (
                "each serialized field retains its banked exact-exterior gauge; "
                "axis, wall and boundary flux are read from that same field"
            ),
        },
        "analytic_stationary_point_analysis": analytic,
        "fixtures": fixtures,
        "diverted_fixture": diverted_fixture,
        "classification": {
            "closed_form_analytic": "limited",
            "alternate_fixed_point": "limited",
            "diverted_analytic": "diverted",
            "claim_reconciliation": {
                "stated_assumption": (
                    "the closed-form root supplies the diverted member of a "
                    "limited/diverted pair"
                ),
                "banked_receipts": (
                    "both coarse and fine receipts label both roots limited"
                ),
                "independent_measurement": (
                    "all four stored-precision reads are limited, wall-bound, "
                    "and contain zero finite in-domain X-points"
                ),
                "verdict": "assumption contradicted by analytic and stored evidence",
            },
            "genuinely_diverted_fixture_in_bank": True,
            "diverted_fixture_action": (
                "completed by an independently banked Solov'ev-family state whose "
                "finite in-domain X-point binds the separatrix"
            ),
            "two_class_acceptance_ready": True,
            "acceptance_implication": (
                "the unchanged fixed-point roots remain a two-root limited case; "
                "the additional exact saddle-bound state supplies an evidenced "
                "diverted class for limited/diverted acceptance"
            ),
        },
    }
    return report, plot_data


def draw(plot_data: dict[str, object]) -> Path:
    fine = plot_data["fine"]
    nodes = np.asarray(fine["nodes"])
    wall = np.asarray(fine["wall"])
    wall = np.vstack((wall, wall[0]))
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), sharex=True, sharey=True)
    panels = (
        ("closed_form_analytic", "Closed-form analytic state"),
        ("alternate_fixed_point", "Alternate fixed point"),
        ("diverted_analytic", "Saddle-bound analytic state"),
    )
    for axis, (root_name, title) in zip(axes, panels, strict=True):
        topology = fine["roots"][root_name]
        axis.scatter(nodes[:, 0], nodes[:, 1], s=2, color="0.82", rasterized=True)
        axis.plot(wall[:, 0], wall[:, 1], color="0.15", linewidth=1.2)
        axis.scatter(*topology["axis_m"], s=30, color="C0", label="magnetic axis")
        if topology["class"] == "limited":
            axis.scatter(
                *topology["wall_contact_point_m"],
                s=42,
                marker="D",
                color="C3",
                label="binding wall contact",
            )
            axis.scatter(0.0, 0.0, s=48, marker="x", color="C1")
            axis.annotate(
                "coordinate-axis saddle\n(outside physical domain)",
                (0.0, 0.0),
                xytext=(0.14, -0.5),
                arrowprops={"arrowstyle": "-", "color": "0.35"},
                fontsize=8,
            )
            summary = "0 finite in-domain X-points\nwall-bound: limited"
        else:
            axis.scatter(
                *topology["selected_x_point_m"],
                s=54,
                marker="x",
                linewidth=2.0,
                color="C1",
                label="binding X-point",
            )
            axis.scatter(
                *topology["wall_contact_point_m"],
                s=36,
                marker="D",
                color="C3",
                label="non-binding wall extremum",
            )
            summary = "1 finite in-domain X-point\nsaddle-bound: diverted"
        axis.text(0.04, 0.96, summary, transform=axis.transAxes, va="top", fontsize=9)
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    axes[2].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_xlim(-0.08, 2.24)
    axes[0].set_ylim(-1.12, 1.12)
    figure.tight_layout()
    path = OUTPUT / "topology-classification.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_markdown(report: dict[str, object], figure: Path) -> Path:
    coarse = report["fixtures"]["coarse"]
    fine = report["fixtures"]["fine"]
    stationary = report["analytic_stationary_point_analysis"]
    saddle = stationary["coordinate_axis_stationary_point"]
    diverted = report["diverted_fixture"]
    diverted_topology = diverted["topology"]
    diverted_saddle = diverted["analytic_stationary_points"]["x_point"]
    diverted_axis = diverted["analytic_stationary_points"]["axis"]
    rows = []
    for resolution in RESOLUTIONS:
        fixture = report["fixtures"][resolution]
        for root_name, _array_name in ROOTS:
            root = fixture["roots"][root_name]
            topology = root["topology"]
            residual = (
                "exact closed form"
                if root_name == "closed_form_analytic"
                else f"{fixture['alternate_terminal_residual']:.17g}"
            )
            rows.append(
                "| {resolution} | {root} | {klass} | {count} | "
                "`[{contact_r:.16g}, {contact_z:.16g}]` | {flux:.17g} | "
                "{residual} |".format(
                    resolution=resolution,
                    root=root_name.replace("_", " "),
                    klass=topology["class"],
                    count=topology["finite_x_point_count"],
                    contact_r=topology["wall_contact_point_m"][0],
                    contact_z=topology["wall_contact_point_m"][1],
                    flux=topology["wall_contact_flux_wb"],
                    residual=residual,
                )
            )
    rows.append(
        "| fine | diverted analytic | {klass} | {count} | "
        "`[{point_r:.16g}, {point_z:.16g}]` | {flux:.17g} | exact polynomial |".format(
            klass=diverted_topology["class"],
            count=diverted_topology["finite_x_point_count"],
            point_r=diverted_topology["selected_x_point_m"][0],
            point_z=diverted_topology["selected_x_point_m"][1],
            flux=diverted_topology["selected_x_point_flux_wb"],
        )
    )
    del figure
    saddle_eigenvalues = saddle["hessian_eigenvalues_per_radian_wb_per_m2"]
    saddle_determinant = saddle["hessian_determinant_per_radian_squared"]
    saddle_wall_distance = fine["coordinate_axis_saddle_nearest_wall_vertex_distance_m"]
    saddle_grid_distance = fine["coordinate_axis_saddle_nearest_grid_node_distance_m"]
    fine_grid_minimum = fine["grid_node_bounds_m"]["minimum"][0]
    fine_wall_minimum = fine["wall_vertex_bounds_m"]["minimum"][0]
    figure_link = "![Fine-carrier topology reads](topology-classification.png)"
    table_header = (
        "| Resolution | Root | Class | Finite X-points | Binding point "
        "[m] | Boundary flux [Wb] | Fixed-point evidence |"
    )
    table_rows = "\n".join(rows)
    coarse_separation_mm = 1000.0 * coarse["axis_separation_m"]
    fine_separation_mm = 1000.0 * fine["axis_separation_m"]
    coarse_residual = coarse["alternate_terminal_residual"]
    fine_residual = fine["alternate_terminal_residual"]
    diverted_eigenvalues = diverted_saddle["hessian_eigenvalues_wb_per_m2"]
    diverted_determinant = diverted_saddle["hessian_determinant_wb2_per_m4"]
    analytic_x = diverted_saddle["coordinate_m"]
    production_x = diverted_topology["selected_x_point_m"]
    locator_error = diverted["production_x_point_localization_error_m"]
    wall_distance = diverted_saddle["nearest_wall_vertex_distance_m"]
    state_identity = diverted["arrays"]["state"]
    coefficient_identity = diverted["arrays"]["coefficients"]
    stationary_identity = diverted["arrays"]["stationary_points"]
    coefficients = diverted["closed_form"]["coefficients"]
    axis_eigenvalues = diverted_axis["hessian_eigenvalues_wb_per_m2"]
    boundary_wall_distance = diverted_topology["boundary_wall_point_distance_m"]
    boundary_wall_flux_difference = diverted_topology[
        "boundary_wall_flux_difference_wb"
    ]
    text = f"""# Topology classification of the banked oracle roots

## Verdict

The two original fixed-point roots remain **limited** in all four stored-precision
reads. Their production boundaries coincide with wall contacts and their fields
contain zero finite in-domain X-points. This still contradicts the earlier claim
that the original closed-form root supplies the diverted class; relabeling either
original receipt would not change its field topology.

The bank now also contains an independently constructed exact Solov'ev-family
state. Its fifth production read is **diverted**, locates exactly one finite
in-domain X-point at `[{production_x[0]:.17g}, {production_x[1]:.17g}] m`, and
selects that saddle as the boundary. The additional state therefore supplies the
genuinely diverted member needed for a two-class acceptance test while preserving
the original four limited classifications.

{figure_link}

## Original closed-form saddle analysis

For the closed form, `dpsi/dZ = -2*k_z*Z` and
`dpsi/dR = -2*R*G'(R^2-R0^2)`. In the physical cylindrical domain `R > 0`,
`G'` vanishes only at `R = R0`, so the sole physical stationary point is the
magnetic-axis maximum at `(1.7, 0) m`; there is no physical X-point.

Extending the formula to the excluded coordinate axis produces a mathematical saddle at
`(0, 0) m`. Its per-radian Hessian eigenvalues are
`[{saddle_eigenvalues[0]:.17g}, {saddle_eigenvalues[1]:.17g}] Wb/m^2`, with
determinant `{saddle_determinant:.17g}`, hence opposite curvature signs. That point
is not a plasma X-point: it lies outside every wall and stored solve lattice. Its
nearest wall-vertex distance is `{saddle_wall_distance:.17g} m` and its nearest
fine-grid node is `{saddle_grid_distance:.17g} m` away. The fine grid starts at
`R = {fine_grid_minimum:.17g} m`; the wall starts at
`R = {fine_wall_minimum:.17g} m`.

## Constructed Solov'ev-family saddle

The added total-flux field is
`Phi = a*R^4 + b*Z^2 + c0 + c1*R^2 + c2*Z + c3*R^2*Z
+ c4*(R^4 - 4*R^2*Z^2)` with binary64 coefficients
`{coefficients}`. The last four non-gauge terms are homogeneous solutions and
`Delta-star(Phi) = 8*a*R^2 + 2*b`, so the field is an exact static
Solov'ev-family Grad-Shafranov state with constant flux-function gradients. The
gauge is fixed by `Phi(X) = 0`; no numerical equilibrium solve or topology label
was used to construct the field.

The analytic magnetic axis is at
`[{diverted_axis["coordinate_m"][0]:.17g}, {diverted_axis["coordinate_m"][1]:.17g}] m`
with Hessian eigenvalues `[{axis_eigenvalues[0]:.17g},
{axis_eigenvalues[1]:.17g}] Wb/m^2`, both negative. The analytic X-point is at
`[{analytic_x[0]:.17g}, {analytic_x[1]:.17g}] m`; its Hessian eigenvalues are
`[{diverted_eigenvalues[0]:.17g}, {diverted_eigenvalues[1]:.17g}] Wb/m^2` and
its determinant is `{diverted_determinant:.17g} Wb^2/m^4`. The opposite signs
prove a saddle. It is strictly inside the wall polygon, with nearest sampled wall
vertex `{wall_distance:.17g} m` away. The production locator differs from the
analytic X-point by `{locator_error:.17g} m`.

## Stored-precision production reads

{table_header}
|---|---|---:|---:|---|---:|---:|
{table_rows}

The locator used its float64 local quadratic fit on unchanged binary64 bank arrays.
Input SHA-256 identities were checked against each source receipt before
classification. Neither receipt label was used as a classification input, and no
nonlinear solve was run. The raw field gauge was retained separately for each state;
its axis, wall, and boundary values were all read from that same field.

For the added fixture, the binary64 state has shape `{state_identity["shape"]}` and
SHA-256 `{state_identity["sha256"]}`. Its coefficient array SHA-256 is
`{coefficient_identity["sha256"]}` and its two-row analytic stationary-point array
SHA-256 is `{stationary_identity["sha256"]}`. The diverted boundary is separated
from the wall extremum by `{boundary_wall_distance:.17g} m`, and its boundary-minus-
wall flux is `{boundary_wall_flux_difference:.17g} Wb`; both nonzero reads distinguish
the saddle-bound separatrix from a wall contact at stored precision.

The independent root separation is retained: the coarse axes differ by
`{coarse_separation_mm:.9g} mm` and the fine axes by
`{fine_separation_mm:.9g} mm`. The alternate roots remain criterion-qualified at
relative residuals `{coarse_residual:.17g}` and `{fine_residual:.17g}`. Distinct
fixed points do not imply distinct topology classes.

## Acceptance implication

The original root pair can exercise two distinct fixed points and same-class
portfolio behavior. The added exact state supplies an independent diverted oracle,
so the combined bank is ready to exercise both limited and diverted acceptance.
The two roles remain separate: the added state proves topology classification, not
a second nonlinear fixed point of the original closed-form carrier.

Machine-readable receipts: `diverted-receipt.json` and
`topology-classification.json`.
"""
    path = OUTPUT / "topology-classification.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    report, plot_data = measure()
    figure = draw(plot_data)
    report["artifacts"] = {
        "classification": str(
            (OUTPUT / "topology-classification.md").relative_to(REPOSITORY_ROOT)
        ),
        "figure": str(figure.relative_to(REPOSITORY_ROOT)),
    }
    _strict_json(OUTPUT / "topology-classification.json", report)
    markdown = write_markdown(report, figure)
    print(
        "CLASSIFIED "
        f"closed_form={report['classification']['closed_form_analytic']} "
        f"alternate={report['classification']['alternate_fixed_point']} "
        "diverted_in_bank="
        f"{report['classification']['genuinely_diverted_fixture_in_bank']} "
        f"receipt={OUTPUT / 'topology-classification.json'} "
        f"report={markdown}"
    )


if __name__ == "__main__":
    main()
