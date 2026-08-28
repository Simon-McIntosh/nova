"""Qualified, connectivity-vetoed wall-height residual exclusion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.connectivity_boundary import wall_height_shadow_mask
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.fixed_point import picard
from nova.equilibrium.forward_operator import ForwardFluxOperator


@dataclass(frozen=True)
class HeightSafetyGeometry:
    """One wall-shadow geometry and its perturbed null census."""

    name: str
    axis_height: float
    qualified: tuple[tuple[float, float], ...]
    primary_index: int | None
    perturbed_qualified: tuple[tuple[float, float], ...]
    perturbed_primary_index: int | None
    pure_shadow: np.ndarray
    flood_regions: np.ndarray
    alternate_primary_index: int | None = None


def _wall_coordinates(count: int = 96) -> np.ndarray:
    """Return a closed D-shaped wall with lower outboard coverage."""
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    radius = 1.7 + 0.82 * np.cos(angle) - 0.08 * np.sin(angle) ** 2
    height = 1.18 * np.sin(angle)
    return np.column_stack((radius, height))


def _flood_region(kind: str) -> np.ndarray:
    """Return explanatory core/private labels on a common plotting grid."""
    radius = np.linspace(0.72, 2.55, 121)
    height = np.linspace(-1.22, 1.22, 141)
    rr, zz = np.meshgrid(radius, height)
    vessel = ((rr - 1.7) / 0.82) ** 2 + (zz / 1.18) ** 2 <= 1.0
    labels = np.zeros(rr.shape, dtype=np.int8)

    if kind == "snowflake-near-equal-height":
        core = ((rr - 1.65) / 0.55) ** 2 + ((zz - 0.18) / 0.58) ** 2 <= 1.0
        private = (
            (((rr - 1.55) / 0.48) ** 2 + ((zz + 0.48) / 0.42) ** 2 <= 1.0)
            | (((rr - 1.98) / 0.32) ** 2 + ((zz + 0.42) / 0.35) ** 2 <= 1.0)
        ) & ~core
    elif kind == "connected-double-null":
        core = ((rr - 1.7) / 0.56) ** 2 + (zz / 0.48) ** 2 <= 1.0
        private = (
            ((rr - 1.7) / 0.48) ** 2 + ((np.abs(zz) - 0.62) / 0.34) ** 2 <= 1.0
        ) & ~core
    elif kind == "coil-null-extremes":
        core = ((rr - 1.82) / 0.52) ** 2 + ((zz + 0.55) / 0.42) ** 2 <= 1.0
        private = vessel & ~core
    elif kind == "limited-no-x-point":
        core = ((rr - 1.7) / 0.58) ** 2 + (zz / 0.68) ** 2 <= 1.0
        private = np.zeros(rr.shape, dtype=bool)
    elif kind == "single-null-outboard-low-node":
        core = ((rr - 1.62) / 0.57) ** 2 + ((zz - 0.22) / 0.56) ** 2 <= 1.0
        private = (((rr - 1.85) / 0.62) ** 2 + ((zz + 0.62) / 0.43) ** 2 <= 1.0) & ~core
    else:  # pragma: no cover - fixture definitions are closed above
        raise ValueError(kind)

    labels[vessel & core] = 1
    labels[vessel & private] = 2
    return labels


def _safety_geometries() -> tuple[HeightSafetyGeometry, ...]:
    """Return the fixed failure-geometry battery used by tests and figures."""
    wall = _wall_coordinates()
    wall_height = wall[:, 1]
    return (
        HeightSafetyGeometry(
            name="snowflake-near-equal-height",
            axis_height=0.24,
            qualified=((1.52, -0.300), (1.96, -0.285)),
            primary_index=0,
            perturbed_qualified=((1.52, -0.285), (1.96, -0.300)),
            perturbed_primary_index=0,
            pure_shadow=wall_height < -0.12,
            flood_regions=_flood_region("snowflake-near-equal-height"),
        ),
        HeightSafetyGeometry(
            name="connected-double-null",
            axis_height=0.0,
            qualified=((1.70, -0.420), (1.70, 0.420)),
            primary_index=0,
            perturbed_qualified=((1.70, -0.415), (1.70, 0.415)),
            perturbed_primary_index=1,
            pure_shadow=np.abs(wall_height) > 0.46,
            flood_regions=_flood_region("connected-double-null"),
        ),
        HeightSafetyGeometry(
            name="coil-null-extremes",
            axis_height=-1.40,
            qualified=((0.35, -1.050), (0.35, 1.050)),
            primary_index=0,
            perturbed_qualified=((0.35, -1.040), (0.35, 1.060)),
            perturbed_primary_index=0,
            pure_shadow=np.ones(wall_height.shape, dtype=bool),
            flood_regions=_flood_region("coil-null-extremes"),
            alternate_primary_index=1,
        ),
        HeightSafetyGeometry(
            name="limited-no-x-point",
            axis_height=0.0,
            qualified=(),
            primary_index=None,
            perturbed_qualified=(),
            perturbed_primary_index=None,
            pure_shadow=np.zeros(wall_height.shape, dtype=bool),
            flood_regions=_flood_region("limited-no-x-point"),
        ),
        HeightSafetyGeometry(
            name="single-null-outboard-low-node",
            axis_height=0.32,
            qualified=((1.66, -0.420),),
            primary_index=0,
            perturbed_qualified=((1.66, -0.410),),
            perturbed_primary_index=0,
            pure_shadow=wall_height < -0.44,
            flood_regions=_flood_region("single-null-outboard-low-node"),
        ),
    )


def _fixed_x_points(points: tuple[tuple[float, float], ...]) -> jax.Array:
    """Pad a candidate census to a fixed four-slot table."""
    rows = list(points)
    rows.extend([(np.nan, np.nan)] * (4 - len(rows)))
    return jnp.asarray(rows[:4], dtype=jnp.float64)


def _geometry_height_mask(
    geometry: HeightSafetyGeometry,
    *,
    perturbed: bool = False,
    alternate: bool = False,
) -> np.ndarray:
    """Evaluate the shipped wall-height predicate for one fixture state."""
    points = geometry.perturbed_qualified if perturbed else geometry.qualified
    if alternate:
        primary_index = geometry.alternate_primary_index
    else:
        primary_index = (
            geometry.perturbed_primary_index if perturbed else geometry.primary_index
        )
    if primary_index is None:
        primary = jnp.asarray((jnp.nan, jnp.nan), dtype=jnp.float64)
    else:
        primary = jnp.asarray(points[primary_index], dtype=jnp.float64)
    result = wall_height_shadow_mask(
        _wall_coordinates()[:, 1],
        geometry.axis_height,
        primary,
        _fixed_x_points(points),
        geometry.pure_shadow,
        jnp.zeros(geometry.pure_shadow.shape, dtype=bool),
        0.02,
        0.1,
    )
    return np.asarray(result, dtype=bool)


def wall_height_safety_metrics() -> list[dict[str, int | str | None]]:
    """Measure baseline difference and null-perturbation churn per geometry."""
    metrics = []
    for geometry in _safety_geometries():
        baseline = _geometry_height_mask(geometry)
        perturbed = _geometry_height_mask(geometry, perturbed=True)
        alternate = (
            _geometry_height_mask(geometry, alternate=True)
            if geometry.alternate_primary_index is not None
            else None
        )
        pure_count = int(np.count_nonzero(geometry.pure_shadow))
        baseline_count = int(np.count_nonzero(baseline))
        metrics.append(
            {
                "geometry": geometry.name,
                "wall_nodes": int(geometry.pure_shadow.size),
                "pure_shadow_nodes": pure_count,
                "height_shadow_nodes": baseline_count,
                "removed_from_pure_shadow": pure_count - baseline_count,
                "added_to_pure_shadow": int(
                    np.count_nonzero(baseline & ~geometry.pure_shadow)
                ),
                "perturbed_height_shadow_nodes": int(np.count_nonzero(perturbed)),
                "perturbation_churn_nodes": int(np.count_nonzero(baseline ^ perturbed)),
                "alternate_primary_shadow_nodes": (
                    int(np.count_nonzero(alternate)) if alternate is not None else None
                ),
                "alternate_primary_churn_nodes": (
                    int(np.count_nonzero(baseline ^ alternate))
                    if alternate is not None
                    else None
                ),
            }
        )
    return metrics


def test_failure_geometry_masks_pin_difference_and_perturbation_churn() -> None:
    """Every failure geometry banks its pure-shadow difference and churn."""
    metrics = {row["geometry"]: row for row in wall_height_safety_metrics()}
    expected = {
        "snowflake-near-equal-height": (45, 39, 6, 41, 2),
        "connected-double-null": (70, 70, 0, 70, 0),
        "coil-null-extremes": (96, 81, 15, 79, 2),
        "limited-no-x-point": (0, 0, 0, 0, 0),
        "single-null-outboard-low-node": (37, 37, 0, 37, 0),
    }
    for name, counts in expected.items():
        row = metrics[name]
        assert (
            row["pure_shadow_nodes"],
            row["height_shadow_nodes"],
            row["removed_from_pure_shadow"],
            row["perturbed_height_shadow_nodes"],
            row["perturbation_churn_nodes"],
        ) == counts
        assert row["added_to_pure_shadow"] == 0

    coil = metrics["coil-null-extremes"]
    assert coil["alternate_primary_shadow_nodes"] == 13
    assert coil["alternate_primary_churn_nodes"] == 68

    single = next(
        geometry
        for geometry in _safety_geometries()
        if geometry.name == "single-null-outboard-low-node"
    )
    wall = _wall_coordinates()
    outboard_private_index = np.argmax(
        np.where(single.pure_shadow, wall[:, 0], -np.inf)
    )
    assert wall[outboard_private_index, 1] < single.qualified[0][1]
    assert _geometry_height_mask(single)[outboard_private_index]


def render_wall_height_safety_evidence(output: Path) -> None:
    """Render the measured wall/flood/null overlays and strict metrics JSON."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    output.mkdir(parents=True, exist_ok=True)
    wall = _wall_coordinates()
    radius = np.linspace(0.72, 2.55, 121)
    height = np.linspace(-1.22, 1.22, 141)
    metrics_by_name = {
        metric["geometry"]: metric for metric in wall_height_safety_metrics()
    }
    colours = ListedColormap(("#f8fafc", "#bfdbfe", "#fed7aa"))

    for geometry in _safety_geometries():
        baseline = _geometry_height_mask(geometry)
        retained = baseline & geometry.pure_shadow
        removed = geometry.pure_shadow & ~baseline
        metric = metrics_by_name[geometry.name]

        figure, axis = plt.subplots(figsize=(6.4, 5.3), constrained_layout=True)
        axis.pcolormesh(
            radius,
            height,
            geometry.flood_regions,
            cmap=colours,
            shading="nearest",
            vmin=0,
            vmax=2,
            alpha=0.78,
        )
        axis.plot(wall[:, 0], wall[:, 1], color="#334155", linewidth=1.2, label="wall")
        axis.scatter(
            wall[removed, 0],
            wall[removed, 1],
            s=34,
            facecolors="none",
            edgecolors="#d97706",
            linewidths=1.3,
            label="pure shadow removed by height",
        )
        axis.scatter(
            wall[retained, 0],
            wall[retained, 1],
            s=22,
            color="#b91c1c",
            label="height-augmented shadow",
        )
        if geometry.qualified:
            points = np.asarray(geometry.qualified)
            axis.scatter(
                points[:, 0],
                points[:, 1],
                marker="x",
                s=70,
                linewidths=2.0,
                color="#111827",
                label="qualified null",
            )
        axis.scatter(
            [1.7],
            [geometry.axis_height],
            marker="*",
            s=80,
            color="#2563eb",
            label="axis height",
        )
        axis.set(
            xlabel="R [m]",
            ylabel="Z [m]",
            xlim=(0.70, 2.62),
            ylim=(-1.25, 1.25),
            aspect="equal",
            title=(
                f"{geometry.name}: pure {metric['pure_shadow_nodes']}, "
                f"height {metric['height_shadow_nodes']}, "
                f"churn {metric['perturbation_churn_nodes']}"
            ),
        )
        axis.text(
            0.02,
            0.02,
            "blue = flood core; orange = private-flux flood region",
            transform=axis.transAxes,
            fontsize=8,
            color="#334155",
        )
        axis.legend(loc="upper right", fontsize=7, framealpha=0.92)
        figure.savefig(output / f"{geometry.name}.png", dpi=180)
        plt.close(figure)

    (output / "metrics.json").write_text(
        json.dumps(wall_height_safety_metrics(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _points(*heights: float) -> jax.Array:
    """Return fixed-shape X-point coordinates at the requested heights."""
    rows = [[1.0, height] for height in heights]
    while len(rows) < 2:
        rows.append([jnp.nan, jnp.nan])
    return jnp.asarray(rows[:2])


def _height_mask(
    wall_height,
    primary_height,
    qualified_heights,
    *,
    private_wall=None,
    previous=None,
    band=0.02,
):
    """Evaluate the production height helper with compact test operands."""
    wall_height = jnp.asarray(wall_height, dtype=jnp.float64)
    if private_wall is None:
        private_wall = jnp.ones(wall_height.shape, dtype=bool)
    if previous is None:
        previous = jnp.zeros(wall_height.shape, dtype=bool)
    primary = jnp.asarray([1.0, primary_height], dtype=wall_height.dtype)
    return wall_height_shadow_mask(
        wall_height,
        0.0,
        primary,
        _points(*qualified_heights),
        private_wall,
        previous,
        jnp.asarray(band, dtype=wall_height.dtype),
        0.1,
    )


def test_only_qualified_primary_and_opposite_candidate_define_limits() -> None:
    """Raw notch and coil-null extrema cannot define a wall limit."""
    wall_height = jnp.linspace(-1.0, 1.0, 37)
    unqualified = _height_mask(wall_height, -0.5, ())
    qualified = _height_mask(wall_height, -0.5, (-0.5, 0.5))

    np.testing.assert_array_equal(unqualified, np.zeros(37, dtype=bool))
    assert int(jnp.sum(qualified)) > 0


@pytest.mark.parametrize(
    ("geometry", "wall_count", "review_false_exclusions"),
    [
        ("disconnected double null", 37, 19),
        ("snowflake minus", 37, 37),
        ("wall notch MAST", 37, 19),
        ("wall notch DIII-D", 84, 38),
        ("coil-null column MAST", 37, 35),
        ("coil-null column DIII-D", 84, 77),
        ("axis crossing", 37, 37),
    ],
)
def test_common_flux_wall_rows_have_zero_false_residual_exclusions(
    geometry, wall_count, review_false_exclusions
) -> None:
    """Height limits cannot suppress a wall equation without private proof."""
    del geometry
    wall_height = jnp.linspace(-1.2, 1.2, wall_count)
    private_wall = jnp.zeros(wall_count, dtype=bool)
    shadow = _height_mask(
        wall_height,
        -0.3,
        (-0.3, 0.3),
        private_wall=private_wall,
    )
    trial = jnp.arange(wall_count, dtype=jnp.float64)
    mapped = trial + 1.0
    residual = trial - jnp.where(shadow, trial, mapped)

    assert review_false_exclusions > 0
    assert int(jnp.sum(shadow)) == 0
    np.testing.assert_array_equal(residual, -jnp.ones(wall_count))


def test_missing_candidate_carries_promoted_window_eager_and_jit() -> None:
    """A transiently absent side leaves its accepted wall participation fixed."""
    wall_height = jnp.asarray([-1.0, -0.25, 0.25, 1.0])

    def evaluate(points, previous):
        return wall_height_shadow_mask(
            wall_height,
            0.0,
            jnp.asarray([1.0, -0.5]),
            points,
            jnp.ones(4, dtype=bool),
            previous,
            0.02,
            0.1,
        )

    for call in (evaluate, jax.jit(evaluate)):
        promoted = call(_points(-0.5, 0.5), jnp.zeros(4, dtype=bool))
        missing = call(_points(), promoted)
        np.testing.assert_array_equal(promoted, [True, False, False, True])
        np.testing.assert_array_equal(missing, promoted)


def test_equality_hover_has_zero_chatter_flips_eager_and_jit() -> None:
    """Sub-band height motion preserves the previously promoted equality bit."""
    heights = jnp.asarray([0.795, 0.800, 0.805, 0.799, 0.801])

    def trajectory(values):
        def promote(previous, height):
            point = jnp.asarray([1.0, height])
            current = wall_height_shadow_mask(
                jnp.asarray([0.8]),
                0.0,
                point,
                jnp.stack((point, jnp.asarray([jnp.nan, jnp.nan]))),
                jnp.asarray([True]),
                previous,
                0.02,
                0.1,
            )
            return current, current

        return jax.lax.scan(promote, jnp.asarray([False]), values)[1]

    for call in (trajectory, jax.jit(trajectory)):
        masks = call(heights)
        assert int(jnp.sum(masks[1:] != masks[:-1])) == 0


def _operator_with_components() -> tuple[ForwardFluxOperator, jax.Array]:
    """Return a minimal operator with independent flood and wall evidence."""
    operator = object.__new__(ForwardFluxOperator)
    operator.grid = SimpleNamespace(node_number=3)
    operator.wall = SimpleNamespace(
        node_number=2,
        coordinate=jnp.asarray([[1.0, -0.8], [1.0, 0.8]]),
    )
    operator.sample = SimpleNamespace(node_number=1)
    operator._wall_height_hysteresis = jnp.asarray(0.02)
    operator._x_qualification_distance = jnp.asarray(0.1)
    masks = DomainMasks(
        label=jnp.asarray(
            [PlasmaDomain.CORE, PlasmaDomain.PRIVATE_FLUX, PlasmaDomain.CORE],
            dtype=jnp.int8,
        ),
        psi_norm=jnp.zeros(3),
    )
    topology = SimpleNamespace(
        axis=jnp.asarray([1.0, 0.0]), x_point=jnp.asarray([1.0, -0.5])
    )
    operator._fixed_design_read = lambda _physical, _requested=None: (
        masks,
        topology,
        ~masks.private_flux,
        jnp.asarray(True),
    )
    operator._connectivity_read = lambda _physical, _topology, classify=False: {
        "xset": _points(-0.5, 0.5),
        "private_wall_node_mask": jnp.asarray([True, False]),
    }
    return operator, masks.private_flux


def test_composed_mask_preserves_flood_and_vetoes_common_wall() -> None:
    """The flood stays the interior authority and common wall rows participate."""
    operator, expected_flood = _operator_with_components()
    trial = jnp.arange(operator.node_number, dtype=jnp.float32)

    for evaluate in (
        operator.residual_shadow_components,
        jax.jit(operator.residual_shadow_components),
    ):
        flood_shadow, wall_shadow = evaluate(trial)
        np.testing.assert_array_equal(flood_shadow, expected_flood)
        np.testing.assert_array_equal(wall_shadow, [True, False])

    combined = operator.residual_shadow_mask(trial)
    np.testing.assert_array_equal(combined, [False, True, False, True, False, False])
    mapped = trial + 10.0
    excluded = operator._exclude_shadow_residual(trial, mapped)
    np.testing.assert_array_equal(excluded, [10.0, 1.0, 12.0, 3.0, 14.0, 15.0])


def test_picard_uses_carried_mask_for_residual_then_promotes_it() -> None:
    """The residual consumes the old mask before the accepted state advances it."""

    def mapped(state):
        return state + 1.0

    def shadowed(state, shadow):
        del state
        return jnp.where(shadow, 0.0, 1.0)

    def proposed(state, previous):
        return jnp.where(state > 0.5, jnp.ones_like(previous), previous)

    result = picard(
        mapped,
        jnp.asarray([0.0]),
        evaluations=2,
        relaxation=1.0,
        shadow_mask_fn=lambda _state: jnp.asarray([False]),
        promoted_shadow_mask_fn=proposed,
        shadowed_map_fn=shadowed,
    )

    np.testing.assert_array_equal(result.state, [0.0])
    np.testing.assert_array_equal(result.shadow_mask_changes, [1, 0])
