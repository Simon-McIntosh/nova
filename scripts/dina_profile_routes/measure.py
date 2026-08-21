"""Bank the DINA declared-profile and map-extracted forcing routes."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np

from nova.equilibrium.convention import (
    flux_function_pressure,
    flux_function_toroidal_field,
)
from nova.equilibrium.map_extraction import extract_flux_functions


OUTPUT = Path(__file__).resolve().parent
FIGURES = Path(__file__).resolve().parents[2] / "docs/figures/dina-profile-routes"
REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
MAPPING_PATH = Path("benchmarks/dina_reference_mapping.py")
ANCHOR_PATH = Path("scripts/normalization_discriminator/results.json")
CACHE_KEYS = {"coarse": "746fbe1553c4b242", "fine": "f0f96aa214aa9459"}
SURFACES = np.linspace(0.05, 0.95, 19)


def load_module(path: Path, name: str):
    """Load a repository module without invoking its command entry point."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finite_statistics(values: np.ndarray) -> dict[str, float]:
    """Summarise a finite vector without discarding its sign."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("statistics require finite values")
    return {
        "signed_mean": float(np.mean(array)),
        "mean_absolute": float(np.mean(np.abs(array))),
        "rms": float(np.sqrt(np.mean(array**2))),
        "sup": float(np.max(np.abs(array))),
    }


def tail_integral(coordinate: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Integrate a sampled gradient from each coordinate to the outer sample."""
    coordinate = np.asarray(coordinate, dtype=float)
    values = np.asarray(values, dtype=float)
    segment = 0.5 * (values[:-1] + values[1:]) * np.diff(coordinate)
    tail = np.zeros_like(values)
    tail[:-1] = np.cumsum(segment[::-1])[::-1]
    return tail


def extraction_receipt(mapped, stored, anchors: dict[str, object]) -> dict[str, object]:
    """Extract map profiles and express them on the declared coordinate."""
    constants = anchors["normalization_constants"]
    map_anchor = constants["production_topology"]
    map_axis = float(map_anchor["axis_flux_wb"])
    map_boundary = float(map_anchor["boundary_flux_wb"])
    map_flux = np.asarray(mapped.grid_flux, dtype=float)
    map_norm = (map_flux - map_axis) / (map_boundary - map_axis)
    declared_norm_map = (map_flux - mapped.flux_axis) / (
        mapped.flux_boundary - mapped.flux_axis
    )
    grid_r, grid_z = np.meshgrid(stored.grid_radius, stored.grid_height, indexing="ij")
    inside_outline = (
        PolygonPath(stored.boundary)
        .contains_points(np.c_[grid_r.ravel(), grid_z.ravel()])
        .reshape(map_flux.shape)
    )
    prescribed_support = (
        inside_outline & (declared_norm_map >= 0.0) & (declared_norm_map <= 1.0)
    )
    extracted = extract_flux_functions(
        stored.grid_radius,
        stored.grid_height,
        map_flux,
        map_norm,
        surfaces=SURFACES,
        plasma_mask=prescribed_support,
        min_samples=6,
    )
    absolute_flux = map_axis + extracted.psi_norm * (map_boundary - map_axis)
    declared_coordinate = (absolute_flux - mapped.flux_axis) / (
        mapped.flux_boundary - mapped.flux_axis
    )
    declared_p_prime = np.interp(declared_coordinate, mapped.psi_norm, mapped.p_prime)
    declared_ff_prime = np.interp(declared_coordinate, mapped.psi_norm, mapped.ff_prime)
    reliable = np.asarray(extracted.reliable, dtype=bool)
    if np.count_nonzero(reliable) < 2:
        raise RuntimeError("map extraction produced fewer than two reliable shells")
    return {
        "surface_receipt": extracted,
        "absolute_flux": absolute_flux,
        "declared_coordinate": declared_coordinate,
        "declared_p_prime": declared_p_prime,
        "declared_ff_prime": declared_ff_prime,
        "prescribed_support_count": int(np.count_nonzero(prescribed_support)),
        "map_coordinate_range_on_prescribed_support": [
            float(np.min(map_norm[prescribed_support])),
            float(np.max(map_norm[prescribed_support])),
        ],
    }


def primitive_checks(mapped, extraction: dict[str, object]) -> dict[str, object]:
    """Integrate both gradient routes against the stored primitive profiles."""
    surface = extraction["surface_receipt"]
    coordinate = np.asarray(extraction["declared_coordinate"], dtype=float)
    reliable = np.asarray(surface.reliable, dtype=bool)
    coordinate = coordinate[reliable]
    order = np.argsort(coordinate)
    coordinate = coordinate[order]
    extracted_p = np.asarray(surface.p_prime)[reliable][order]
    extracted_ff = np.asarray(surface.ff_prime)[reliable][order]
    declared_p = np.asarray(extraction["declared_p_prime"])[reliable][order]
    declared_ff = np.asarray(extraction["declared_ff_prime"])[reliable][order]
    stored_pressure = np.interp(coordinate, mapped.psi_norm, mapped.pressure)
    stored_field = np.interp(coordinate, mapped.psi_norm, mapped.field_function)
    outer_pressure = float(stored_pressure[-1])
    outer_field = float(stored_field[-1])
    span = float(mapped.flux_boundary - mapped.flux_axis)

    def integrate(p_prime: np.ndarray, ff_prime: np.ndarray):
        pressure = flux_function_pressure(
            outer_pressure, span, tail_integral(coordinate, p_prime)
        )
        field_squared = flux_function_toroidal_field(
            outer_field, span, tail_integral(coordinate, ff_prime)
        )
        field = np.sign(outer_field) * np.sqrt(np.maximum(field_squared, 0.0))
        return np.asarray(pressure), np.asarray(field), np.asarray(field_squared)

    declared_pressure, declared_field, declared_field_squared = integrate(
        declared_p, declared_ff
    )
    extracted_pressure, extracted_field, extracted_field_squared = integrate(
        extracted_p, extracted_ff
    )
    return {
        "psi_norm_declared": coordinate,
        "stored_pressure": stored_pressure,
        "stored_field_function": stored_field,
        "declared": {
            "pressure": declared_pressure,
            "field_function": declared_field,
            "field_function_squared": declared_field_squared,
            "pressure_error": finite_statistics(declared_pressure - stored_pressure),
            "field_function_error": finite_statistics(declared_field - stored_field),
        },
        "map_extracted": {
            "pressure": extracted_pressure,
            "field_function": extracted_field,
            "field_function_squared": extracted_field_squared,
            "pressure_error": finite_statistics(extracted_pressure - stored_pressure),
            "field_function_error": finite_statistics(extracted_field - stored_field),
        },
        "integration_support": {
            "policy": (
                "reliable extracted shells only; both route integrals are anchored "
                "to the stored primitive at the outermost common reliable shell"
            ),
            "minimum_psi_norm_declared": float(coordinate[0]),
            "maximum_psi_norm_declared": float(coordinate[-1]),
            "sample_count": int(coordinate.size),
        },
    }


def forcing_profiles(mapped, extraction: dict[str, object]):
    """Return the declared and reliable extracted source tables."""
    surface = extraction["surface_receipt"]
    reliable = np.asarray(surface.reliable, dtype=bool)
    coordinate = np.asarray(extraction["declared_coordinate"])[reliable]
    order = np.argsort(coordinate)
    return {
        "declared": (
            np.asarray(mapped.psi_norm),
            np.asarray(mapped.p_prime),
            np.asarray(mapped.ff_prime),
        ),
        "map_extracted": (
            coordinate[order],
            np.asarray(surface.p_prime)[reliable][order],
            np.asarray(surface.ff_prime)[reliable][order],
        ),
    }


def forcing_receipt(reference, case, mapped, extraction) -> dict[str, object]:
    """Evaluate both source routes on the warm cached fixture carriers."""
    profiles = forcing_profiles(mapped, extraction)
    fixtures = {}
    for name, requested, wall_nodes in (
        ("coarse", reference.SUITE_CELLS, 3),
        ("fine", 2 * reference.SUITE_CELLS, 6),
    ):
        reference.WALL_NODES = wall_nodes
        expected_key = CACHE_KEYS[name]
        store = reference.ZarrStore(
            filename=reference.MACHINE_CACHE_FILENAME,
            dirname=".nova",
            group=expected_key,
        )
        started = perf_counter()
        with reference._machine_cache_lock(store) as lock_wait_seconds:
            store.load()
            identity = json.loads(store.data.attrs["semantic_identity"])
            if identity["discretisation"]["cells"] != requested:
                raise RuntimeError(
                    f"{name} banked carrier has a different cell request"
                )
            machine = reference._machine_from_dataset(
                store.data, identity, expected_key
            )
            arrays_verified, bytes_verified = (
                reference.assert_machine_arrays_bitwise_identical(machine, machine)
            )
        load_seconds = perf_counter() - started
        seed = reference.seed_flux(case, machine)
        route_images = {}
        route_rows = {}
        core = None
        for route, (coordinate, p_prime, ff_prime) in profiles.items():
            route_case = replace(
                case,
                psi_norm=np.asarray(coordinate),
                p_prime=np.asarray(p_prime),
                ff_prime=np.asarray(ff_prime),
            )
            operator = reference.forward_operator(route_case, machine)
            image = operator.flux_map()(seed)
            jax.block_until_ready(image)
            forcing = np.asarray(image - seed)
            if core is None:
                masks = operator._support_partition(seed)[0]
                core = np.asarray(masks.core, dtype=bool)
            grid_forcing = forcing[: len(machine.node)]
            route_images[route] = np.asarray(image)
            route_rows[route] = {
                "grid_core_forcing_wb": finite_statistics(grid_forcing[core]),
                "state_forcing_wb": finite_statistics(forcing),
                "profile_support": {
                    "minimum_psi_norm_declared": float(np.min(coordinate)),
                    "maximum_psi_norm_declared": float(np.max(coordinate)),
                    "samples": int(len(coordinate)),
                    "outside_support": (
                        "edge-held by the existing jnp.interp source primitive"
                        if route == "map_extracted"
                        else "not applicable"
                    ),
                },
            }
        difference = route_images["map_extracted"] - route_images["declared"]
        fixtures[name] = {
            "requested_cells": int(requested),
            "realised_cells": int(len(machine.node)),
            "state_nodes": int(len(seed)),
            "core_cells": int(np.count_nonzero(core)),
            "cache": {
                "warm_hit": True,
                "semantic_key": expected_key,
                "load_seconds": float(load_seconds),
                "lock_wait_seconds": float(lock_wait_seconds),
                "arrays_verified": int(arrays_verified),
                "bytes_verified": int(bytes_verified),
                "bitwise_stored_precision": True,
                "store": str(store.filepath),
                "identity_policy": (
                    "open the banked semantic group and validate its stored identity; "
                    "never fall through to the current helper's cold-build branch"
                ),
                "route_policy": identity["routes"],
            },
            "routes": route_rows,
            "map_extracted_minus_declared_image_wb": finite_statistics(difference),
        }
    return fixtures


def _figure_footer(figure, anchor: dict[str, object]) -> None:
    """State gauge and both anchor provenances inside every figure."""
    declared = anchor["exact_case"]
    mapped = anchor["production_topology"]
    figure.text(
        0.5,
        0.006,
        (
            "Gauge: Nova total poloidal flux Φ = −ψ_IDS (COCOS 11→17). "
            "\n"
            f"Declared IDS anchors: {declared['axis_flux_wb']:.6f}, "
            f"{declared['boundary_flux_wb']:.6f} Wb. "
            "Banked cached-map topology anchors: "
            f"{mapped['axis_flux_wb']:.6f}, {mapped['boundary_flux_wb']:.6f} Wb."
        ),
        ha="center",
        va="bottom",
        fontsize=6.5,
    )


def render_figures(report: dict[str, object]) -> list[str]:
    """Render profile, deviation, primitive, anchor, and forcing evidence."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    common = report["common_profile_base"]
    psi = np.asarray(common["psi_norm_declared"])
    reliable = np.asarray(common["reliable"], dtype=bool)
    anchor = report["anchors"]["normalization_constants"]
    paths = []

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    for axis, name, ylabel in (
        (axes[0], "p_prime", "p′ [Pa/Wb]"),
        (axes[1], "ff_prime", "FF′ [T² m²/Wb]"),
    ):
        axis.plot(psi, common[f"declared_{name}"], label="IDS declared", color="black")
        axis.errorbar(
            psi[reliable],
            np.asarray(common[f"map_extracted_{name}"])[reliable],
            yerr=np.asarray(common[f"map_extracted_{name}_uncertainty"])[reliable],
            fmt="o-",
            ms=3,
            label="map extracted ± uncertainty",
            color="#0072b2",
        )
        if np.any(~reliable):
            axis.scatter(
                psi[~reliable],
                np.zeros(np.count_nonzero(~reliable)),
                marker="x",
                color="#d55e00",
                label="unreliable shell" if axis is axes[0] else None,
            )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[1].set_xlabel("declared ψ_N")
    axes[0].legend(fontsize=8)
    figure.suptitle("DINA flux functions by declared and map-extracted routes")
    _figure_footer(figure, anchor)
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    path = FIGURES / "profile_routes.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.8), sharex=True)
    for axis, name, ylabel in (
        (axes[0], "p_prime", "map − declared [Pa/Wb]"),
        (axes[1], "ff_prime", "map − declared [T² m²/Wb]"),
    ):
        deviation = np.asarray(common[f"map_extracted_{name}"]) - np.asarray(
            common[f"declared_{name}"]
        )
        uncertainty = np.asarray(common[f"map_extracted_{name}_uncertainty"])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.errorbar(
            psi[reliable], deviation[reliable], yerr=uncertainty[reliable], fmt="o-"
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[1].set_xlabel("declared ψ_N")
    figure.suptitle("Unreconciled route disagreement on reliable extracted shells")
    _figure_footer(figure, anchor)
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    path = FIGURES / "route_deviation.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    primitive = report["primitive_integral_cross_checks"]
    ppsi = np.asarray(primitive["psi_norm_declared"])
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.8), sharex=True)
    axes[0].plot(ppsi, primitive["stored_pressure"], color="black", label="stored")
    axes[0].plot(ppsi, primitive["declared"]["pressure"], label="integrated declared")
    axes[0].plot(ppsi, primitive["map_extracted"]["pressure"], label="integrated map")
    axes[0].set_ylabel("pressure [Pa]")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        ppsi, primitive["stored_field_function"], color="black", label="stored"
    )
    axes[1].plot(
        ppsi, primitive["declared"]["field_function"], label="integrated declared"
    )
    axes[1].plot(
        ppsi, primitive["map_extracted"]["field_function"], label="integrated map"
    )
    axes[1].set_ylabel("F [T m]")
    axes[1].set_xlabel("declared ψ_N")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Primitive integrals against IDS pressure and field function")
    _figure_footer(figure, anchor)
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    path = FIGURES / "primitive_integrals.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    offsets = anchor["offsets"]
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    values = [offsets["axis_flux_wb"], offsets["boundary_flux_wb"]]
    bars = axis.bar(["axis", "boundary / saddle"], values, color=["#56b4e9", "#d55e00"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("map/topology − declared [Wb]")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.6f} Wb",
            ha="center",
            va="bottom",
        )
    axis.set_title(
        "Declared versus map/topology anchors\n"
        f"boundary class = {offsets['boundary_flux_wb']:.6f} Wb = "
        f"{100 * offsets['boundary_flux_in_exact_psi_norm']:.6f}% of declared span"
    )
    _figure_footer(figure, anchor)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    path = FIGURES / "anchor_offsets.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    fixtures = report["reproduction_lane_forcing"]
    labels, declared, extracted, difference = [], [], [], []
    for name, fixture in fixtures.items():
        labels.append(name)
        declared.append(fixture["routes"]["declared"]["grid_core_forcing_wb"]["sup"])
        extracted.append(
            fixture["routes"]["map_extracted"]["grid_core_forcing_wb"]["sup"]
        )
        difference.append(fixture["map_extracted_minus_declared_image_wb"]["sup"])
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    width = 0.25
    axis.bar(x - width, declared, width, label="declared forcing")
    axis.bar(x, extracted, width, label="map-extracted forcing")
    axis.bar(x + width, difference, width, label="route image difference")
    axis.set_xticks(x, labels)
    axis.set_ylabel("sup-norm [Wb]")
    axis.set_yscale("log")
    axis.legend(fontsize=8)
    axis.set_title("Warm-cache reproduction forcing by source route")
    _figure_footer(figure, anchor)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    path = FIGURES / "forcing_routes.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))
    return paths


def jsonable(value):
    """Convert arrays and non-finite qualifications into strict JSON values."""
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_html(report: dict[str, object]) -> None:
    """Write a compact browser report with project-absolute figure sources."""
    profile = report["profile_route_disagreement"]
    primitive = report["primitive_integral_cross_checks"]
    forcing = report["reproduction_lane_forcing"]
    figures = "\n".join(
        f'<figure><img src="/nova/figures/dina-profile-routes/{Path(path).name}" '
        f'alt="{Path(path).stem.replace("_", " ")}"></figure>'
        for path in report["figures"]
    )
    rows = "".join(
        f"<tr><td>{name}</td><td>{item['cache']['load_seconds']:.6f}</td>"
        f"<td>{item['routes']['declared']['grid_core_forcing_wb']['sup']:.8g}</td>"
        f"<td>{item['routes']['map_extracted']['grid_core_forcing_wb']['sup']:.8g}</td>"
        f"<td>{item['map_extracted_minus_declared_image_wb']['sup']:.8g}</td></tr>"
        for name, item in forcing.items()
    )
    time_s = report["reference"]["time_s"]
    reliable_shells = report["qualification"]["reliable_shells"]
    total_shells = report["qualification"]["total_shells"]
    p_prime_rms = profile["p_prime"]["rms"]
    ff_prime_rms = profile["ff_prime"]["rms"]
    declared_pressure_rms = primitive["declared"]["pressure_error"]["rms"]
    map_pressure_rms = primitive["map_extracted"]["pressure_error"]["rms"]
    declared_field_rms = primitive["declared"]["field_function_error"]["rms"]
    map_field_rms = primitive["map_extracted"]["field_function_error"]["rms"]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>DINA dual-route receipt</title>
<link rel="stylesheet" href="/_shared/foundation.css"></head><body><main>
<h1>DINA dual-route flux-function receipt</h1>
<p>Pulse 135011, run 7, slice index 353 at {time_s:.7f} s was read with
imas-python at its written DD 3.39.0. The map-extracted and IDS-declared
routes remain separate.</p>
<p>Reliable extracted shells: {reliable_shells} / {total_shells}.
Pressure-gradient disagreement RMS: {p_prime_rms:.8g} Pa/Wb;
FF-prime disagreement RMS: {ff_prime_rms:.8g} T² m²/Wb.</p>
<p>Primitive closure RMS, declared/map: pressure
{declared_pressure_rms:.8g} / {map_pressure_rms:.8g} Pa;
F {declared_field_rms:.8g} / {map_field_rms:.8g} T m.</p>
<table><thead><tr><th>fixture</th><th>warm load [s]</th>
<th>declared forcing sup [Wb]</th><th>map forcing sup [Wb]</th>
<th>route image difference sup [Wb]</th></tr></thead>
<tbody>{rows}</tbody></table>
{figures}
</main></body></html>"""
    (OUTPUT / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    """Read, extract, force, render, and serialize the dual-route receipt."""
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("set JAX_PLATFORMS=cpu for the reproduction lane")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = load_module(REFERENCE_PATH, "dina_route_reference")
    mapping = load_module(MAPPING_PATH, "dina_route_mapping")
    reference.configure_dtypes()
    stored = mapping.read_stored_profiles()
    mapped = mapping.derive_mapping(stored)
    case = reference.require_reference()
    comparison = mapping.mapping_comparison(case, mapped)
    if not all(row["bit_identical"] for row in comparison.values()):
        raise AssertionError(
            "independent native-DD mapping differs from the stored reader"
        )
    anchors = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
    extraction = extraction_receipt(mapped, stored, anchors)
    surface = extraction["surface_receipt"]
    reliable = np.asarray(surface.reliable, dtype=bool)
    common_psi = np.asarray(extraction["declared_coordinate"])
    p_deviation = (
        np.asarray(surface.p_prime)[reliable]
        - np.asarray(extraction["declared_p_prime"])[reliable]
    )
    ff_deviation = (
        np.asarray(surface.ff_prime)[reliable]
        - np.asarray(extraction["declared_ff_prime"])[reliable]
    )
    primitive = primitive_checks(mapped, extraction)
    forcing = forcing_receipt(reference, case, mapped, extraction)
    report = {
        "schema": "nova.dina-dual-route-flux-function-receipt",
        "reference": {
            "pulse": mapping.PULSE,
            "run": mapping.RUN,
            "time_slice_index": mapping.TIME_SLICE,
            "time_s": float(stored.time),
            "dd_version": mapping.DD_VERSION,
            "reader": "imas.DBEntry",
            "source_cocos": mapping.SOURCE_COCOS,
            "target_cocos": mapping.TARGET_COCOS,
            "mapping_bit_identical_to_stored_reader": True,
        },
        "gauge": (
            "Nova total poloidal flux Phi = -psi_IDS after COCOS 11 to 17; "
            "declared anchors bound reproduction support, map/topology anchors "
            "only parameterize affine extraction"
        ),
        "anchors": {
            "provenance": (
                "scripts/normalization_discriminator/results.json; banked cached-map "
                "topology against native-DD declared IDS constants"
            ),
            "normalization_constants": anchors["normalization_constants"],
        },
        "common_profile_base": {
            "definition": (
                "map shell centres converted to absolute Nova total flux through "
                "map anchors, then into the declared-anchor psi_N coordinate"
            ),
            "psi_norm_declared": common_psi,
            "absolute_flux_wb": extraction["absolute_flux"],
            "reliable": reliable,
            "declared_p_prime": extraction["declared_p_prime"],
            "declared_ff_prime": extraction["declared_ff_prime"],
            "map_extracted_p_prime": surface.p_prime,
            "map_extracted_ff_prime": surface.ff_prime,
            "map_extracted_p_prime_uncertainty": surface.p_prime_uncertainty,
            "map_extracted_ff_prime_uncertainty": surface.ff_prime_uncertainty,
            "projection_rms": surface.projection_rms,
            "condition_number": surface.condition_number,
            "uncertainty_inflation": surface.uncertainty_inflation,
            "minimum_gradient": surface.minimum_gradient,
            "sample_count": surface.sample_count,
        },
        "qualification": {
            "total_shells": int(len(surface.psi_norm)),
            "reliable_shells": int(np.count_nonzero(reliable)),
            "unreliable_shells": int(np.count_nonzero(~reliable)),
            "prescribed_support_map_nodes": extraction["prescribed_support_count"],
            "map_coordinate_range_on_prescribed_support": extraction[
                "map_coordinate_range_on_prescribed_support"
            ],
            "policy": "unreliable shells are retained and masked, never reconciled",
        },
        "profile_route_disagreement": {
            "definition": "map extracted minus IDS declared on reliable shells",
            "p_prime": finite_statistics(p_deviation),
            "ff_prime": finite_statistics(ff_deviation),
        },
        "primitive_integral_cross_checks": primitive,
        "reproduction_lane_forcing": forcing,
        "route_policy": (
            "Both route images are banked. Their disagreement is reported and no "
            "route is rescaled, shifted, blended, or selected as a correction."
        ),
    }
    report["figures"] = render_figures(report)
    strict = jsonable(report)
    (OUTPUT / "receipt.json").write_text(
        json.dumps(strict, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_html(report)
    print(
        "DINA_DUAL_ROUTE_EXIT=0 "
        f"reliable={np.count_nonzero(reliable)}/{len(reliable)} "
        f"pprime_rms={report['profile_route_disagreement']['p_prime']['rms']:.9g} "
        f"ffprime_rms={report['profile_route_disagreement']['ff_prime']['rms']:.9g}"
    )


if __name__ == "__main__":
    main()
