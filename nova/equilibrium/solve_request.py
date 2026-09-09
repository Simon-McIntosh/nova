"""Typed public inputs and provenance for one forward equilibrium solve."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import os
from pathlib import Path
import socket
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from nova import __version__ as NOVA_VERSION

if TYPE_CHECKING:
    import jax

    from nova.equilibrium.constraint import ConstraintPair, ConstraintRecord
    from nova.equilibrium.forward import ForwardEquilibrium
    from nova.equilibrium.forward_operator import ForwardTopologyState
    from nova.equilibrium.observation import ConstraintPinSet
    from nova.equilibrium.source import ForwardSource

SolveRoute = Literal[
    "host",
    "host_krylov",
    "picard",
    "anderson",
    "newton_krylov",
    "reduced_newton",
]
JsonScalar = str | int | float | bool | None


def _evaluate_sampled_flux_function(
    coordinate,
    values,
    edge_width,
    lower_slope,
    upper_slope,
    lower_edge_increment,
    upper_edge_increment,
    psi_norm,
):
    """Evaluate a sampled flux function from its already-derived edge data."""
    grid = coordinate
    samples = values
    evaluation_coordinate = jnp.asarray(psi_norm)
    interior = jnp.interp(evaluation_coordinate, grid, samples)
    below = evaluation_coordinate < grid[0]
    edge_value = jnp.where(below, samples[0], samples[-1])
    del lower_slope, upper_slope
    outward_increment = jnp.where(below, -lower_edge_increment, upper_edge_increment)
    outward_distance = jnp.where(
        below,
        grid[0] - evaluation_coordinate,
        evaluation_coordinate - grid[-1],
    )
    parameter = jnp.clip(outward_distance / edge_width, 0.0, 1.0)
    parameter_cubed = parameter * parameter * parameter
    value_basis = 1.0 + parameter_cubed * (-10.0 + parameter * (15.0 - 6.0 * parameter))
    slope_basis = parameter + parameter_cubed * (
        -6.0 + parameter * (8.0 - 3.0 * parameter)
    )
    exterior = value_basis * edge_value + slope_basis * outward_increment
    return jnp.where(
        (evaluation_coordinate >= grid[0]) & (evaluation_coordinate <= grid[-1]),
        interior,
        exterior,
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class SampledFluxFunction:
    """A member-varying flux function carried entirely by array leaves.

    Values interpolate on a strictly increasing normalized-flux coordinate.
    Outside that coordinate, a quintic end cap preserves the edge value and
    slope before reaching zero one edge interval away. The representation is
    the array-owned equivalent of a closure around ``jnp.interp`` and can be
    stacked without closing any member's table into a compiled program.
    """

    coordinate: object
    values: object
    edge_width: object = field(init=False)
    lower_slope: object = field(init=False)
    upper_slope: object = field(init=False)
    lower_edge_increment: object = field(init=False)
    upper_edge_increment: object = field(init=False)

    def __post_init__(self) -> None:
        coordinate = np.asarray(self.coordinate, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        if coordinate.ndim != 1 or coordinate.size < 2:
            raise ValueError("sampled flux coordinate needs at least two points")
        if values.shape != coordinate.shape:
            raise ValueError("sampled flux values must match their coordinate")
        if not np.all(np.diff(coordinate) > 0.0):
            raise ValueError("sampled flux coordinate must increase strictly")
        grid = jnp.asarray(coordinate)
        samples = jnp.asarray(values)
        edge_width = grid[1] - grid[0]
        lower_slope = (samples[1] - samples[0]) / edge_width
        upper_slope = (samples[-1] - samples[-2]) / edge_width
        lower_edge_increment = edge_width * lower_slope
        upper_edge_increment = edge_width * upper_slope
        object.__setattr__(self, "coordinate", grid)
        object.__setattr__(self, "values", samples)
        object.__setattr__(self, "edge_width", edge_width)
        object.__setattr__(self, "lower_slope", lower_slope)
        object.__setattr__(self, "upper_slope", upper_slope)
        object.__setattr__(self, "lower_edge_increment", lower_edge_increment)
        object.__setattr__(self, "upper_edge_increment", upper_edge_increment)

    def __call__(self, psi_norm):
        """Evaluate the interpolant and its slope-matched compact end caps."""
        return _evaluate_sampled_flux_function(
            self.coordinate,
            self.values,
            self.edge_width,
            self.lower_slope,
            self.upper_slope,
            self.lower_edge_increment,
            self.upper_edge_increment,
            psi_norm,
        )

    def tree_flatten(self):
        """Return profile tables and precomputed edge data as dynamic leaves."""
        return (
            self.coordinate,
            self.values,
            self.edge_width,
            self.lower_slope,
            self.upper_slope,
            self.lower_edge_increment,
            self.upper_edge_increment,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild without asking NumPy to inspect traced member arrays."""
        del aux_data
        instance = object.__new__(cls)
        object.__setattr__(instance, "coordinate", children[0])
        object.__setattr__(instance, "values", children[1])
        object.__setattr__(instance, "edge_width", children[2])
        object.__setattr__(instance, "lower_slope", children[3])
        object.__setattr__(instance, "upper_slope", children[4])
        object.__setattr__(instance, "lower_edge_increment", children[5])
        object.__setattr__(instance, "upper_edge_increment", children[6])
        return instance


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class ForwardSolveMemberData:
    """Per-member solve inputs that gain a leading batch axis together."""

    seed_state: object
    target_current: object | None = None
    current: object | None = None
    prescribed_current: object | None = None

    def tree_flatten(self):
        """Return every numerical solve input as a dynamic leaf."""
        return (
            self.seed_state,
            self.target_current,
            self.current,
            self.prescribed_current,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def default_forward_compilation_cache_root() -> Path:
    """Return the shared, per-host root that keeps forward compiled programs.

    The root defaults to ``~/.cache/nova/forward-compilation-cache`` on the
    shared home filesystem, so a build written by one SLURM allocation survives
    for the next allocation that lands on the same host.  An explicit
    ``NOVA_FORWARD_COMPILATION_CACHE_ROOT`` replaces that base directory for a
    launch that must name its own location; the user and host keys still sit
    below it.

    Isolation and keying.  The root is scoped per user and per host
    (``user-<uid>/host-<hostname>``).  Host keying is what separates builds: a
    cache written by one node is never read by another node, so a
    machine-specific build cannot leak across nodes, and allocations on
    different nodes never write the same file.  Allocations on the same node
    share one subtree and are serialised by JAX's own cache entry lock; JAX
    writes an entry as a plain open+write under that lock, so this layout does
    not rely on the atomic no-clobber create that GPFS does not provide.
    Architecture keying is carried below this root: the persistent cache
    configuration appends ``nova/jax-compilation/runtime-<hash>`` where the
    hash covers the jax/jaxlib versions, backend platform and version, the x64
    flag, and the device topology, so a build for one runtime is never read by
    another.

    The base deliberately sits under ``~/.cache`` rather than ``~/.local``:
    the shared filesystem carries the per-user data tree with its setgid group
    inherited down the hierarchy, and that group has no name in the cluster's
    name service.  JAX's cache runtime resolves every cached file's group
    through ``grp.getgrgid`` while evicting, and a group that cannot be
    resolved aborts the write of every entry after the first, so a root inside
    that tree would silently stop persisting compiles.  ``~/.cache`` is not
    setgid; files written beneath it carry the user's own resolvable group.

    TMPDIR is deliberately not consulted: every SLURM step on this cluster sets
    TMPDIR onto the node-local filesystem, so a TMPDIR-derived root dies with
    the allocation and no job banks a compile for its successor.
    """

    override = os.environ.get("NOVA_FORWARD_COMPILATION_CACHE_ROOT")
    if override:
        base = Path(override)
    else:
        base = Path.home() / ".cache" / "nova" / "forward-compilation-cache"
    return base.expanduser() / f"user-{os.getuid()}" / f"host-{socket.gethostname()}"


@dataclass(frozen=True, slots=True)
class ForwardSolvePolicy:
    """Every resolved numerical and acceptance choice for a forward solve."""

    route: SolveRoute = "newton_krylov"
    newton_steps: int = 10
    gmres_iterations: int = 30
    warmup: int = 0
    relaxation: float = 0.5
    step_cap: float = 10.0
    active_set_steps: int = 16
    kernel_tolerance: float = 1.0e-8
    qualification_tolerance: float = 1.0e-10
    current_pin: bool = True
    settled_exit: bool = True
    own_mask_acceptance: bool = True
    continuation: bool = True
    best_iterate_retention: bool = True
    stagnation_stop: bool = True
    exact_kernels: bool = True
    cached_machine: bool = True
    compilation_cache: bool = True

    def __post_init__(self) -> None:
        """Reject policies that cannot name a bounded numerical solve."""

        if self.route not in {
            "host",
            "host_krylov",
            "picard",
            "anderson",
            "newton_krylov",
            "reduced_newton",
        }:
            raise ValueError(f"unknown forward solve route {self.route!r}")
        for name in (
            "newton_steps",
            "gmres_iterations",
            "active_set_steps",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.warmup < 0:
            raise ValueError("warmup cannot be negative")
        for name in (
            "relaxation",
            "step_cap",
            "kernel_tolerance",
            "qualification_tolerance",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, JsonScalar]:
        """Return the JSON-native policy block written into receipts."""

        return asdict(self)

    def kernel_options(self) -> dict[str, JsonScalar]:
        """Translate this policy into the selected route's public keywords."""

        if self.route == "newton_krylov":
            return {
                "newton_steps": self.newton_steps,
                "gmres_iterations": self.gmres_iterations,
                "warmup": self.warmup,
                "relaxation": self.relaxation,
                "step_cap": self.step_cap,
                "active_set_steps": self.active_set_steps,
                "convergence_tolerance": self.kernel_tolerance,
                "stop_on_active_set_stagnation": self.stagnation_stop,
                "stop_on_active_set_settlement": self.settled_exit,
                "retain_outer_best_iterate": self.best_iterate_retention,
                "continue_newton_trajectory": self.continuation,
                "continue_globalization_state": self.continuation,
                "own_mask_acceptance": self.own_mask_acceptance,
            }
        if self.route == "reduced_newton":
            return {
                "newton_steps": self.newton_steps,
                "active_set_steps": self.active_set_steps,
                "tolerance": self.kernel_tolerance,
            }
        if self.route in {"picard", "anderson"}:
            options: dict[str, JsonScalar] = {
                "evaluations": self.newton_steps,
                "relaxation": self.relaxation,
            }
            if self.route == "anderson":
                options.update(warmup=self.warmup, step_cap=self.step_cap)
            return options
        if self.route == "host":
            return {
                "evaluations": self.newton_steps,
                "relaxation": self.relaxation,
                "tolerance": self.kernel_tolerance,
            }
        if self.route == "host_krylov":
            return {}
        raise ValueError(f"unknown forward solve route {self.route!r}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ForwardSolvePolicy:
        """Restore a policy while refusing missing or additional fields."""

        expected = {item.name for item in fields(cls)}
        received = set(payload)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(
                f"forward solve policy fields differ; missing={missing}, extra={extra}"
            )
        return cls(**dict(payload))


# This is the sole declaration of public forward-solve defaults.  Its key is
# the installed Nova package version; changing a value therefore belongs to a
# Nova release and every resolved receipt retains both the values and that key.
FORWARD_SOLVE_DEFAULTS: Mapping[str, ForwardSolvePolicy] = MappingProxyType(
    {NOVA_VERSION: ForwardSolvePolicy()}
)


def declared_forward_solve_policy(
    nova_version: str = NOVA_VERSION,
) -> ForwardSolvePolicy:
    """Return the immutable defaults declared for one installed Nova version."""

    try:
        return FORWARD_SOLVE_DEFAULTS[nova_version]
    except KeyError as error:
        raise KeyError(
            f"Nova {nova_version!r} has no declared forward solve policy"
        ) from error


def resolve_forward_solve_policy(
    *,
    route: SolveRoute | None = None,
    overrides: Mapping[str, JsonScalar] | None = None,
    nova_version: str = NOVA_VERSION,
) -> ForwardSolvePolicy:
    """Resolve one route and its explicit deviations from the sole default table."""

    policy = declared_forward_solve_policy(nova_version)
    deviations = dict(overrides or {})
    if route is not None:
        deviations["route"] = route
    return replace(policy, **deviations) if deviations else policy


@dataclass(frozen=True, slots=True)
class ExplicitSolveSeed:
    """An explicitly supplied total-flux state used as the solve seed."""

    state: object

    def resolve(self, _profile: object, *, current: object | None = None) -> object:
        """Return the state unchanged so its dtype and bytes remain authoritative."""

        del current
        return self.state

    def provenance(self) -> SolveSeedProvenance:
        """Describe the caller-owned state that supplied this seed."""

        return SolveSeedProvenance(kind="explicit")


@dataclass(frozen=True, slots=True)
class SolveSeedProvenance:
    """Stable receipt description of how a public solve obtained its seed."""

    kind: Literal["explicit", "cold_seed_portfolio"]
    requested_class: Literal["limited", "diverted"] | None = None
    plasma_current: float | None = None
    centroid: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, JsonScalar | tuple[float, float] | None]:
        """Return the JSON-native description carried by a solve receipt."""

        return {
            "kind": self.kind,
            "requested_class": self.requested_class,
            "plasma_current": self.plasma_current,
            "centroid": self.centroid,
        }


@dataclass(frozen=True, slots=True)
class ColdSeedPortfolio:
    """Construct one selected cold branch through ``profile.cold_seed_portfolio``."""

    plasma_current: float
    centroid: tuple[float, float]
    requested_class: Literal["limited", "diverted"] = "limited"
    radius_fraction: float | None = None
    diverted_geometry: object | None = None

    def __post_init__(self) -> None:
        """Freeze the physical inputs that identify a cold-seed construction."""

        centroid = tuple(float(value) for value in self.centroid)
        if len(centroid) != 2:
            raise ValueError("cold-seed centroid must be a radius, height pair")
        if not np.all(np.isfinite(centroid)):
            raise ValueError("cold-seed centroid must be finite")
        if not np.isfinite(self.plasma_current):
            raise ValueError("cold-seed plasma current must be finite")
        if self.requested_class not in {"limited", "diverted"}:
            raise ValueError("cold-seed class must be limited or diverted")
        object.__setattr__(self, "centroid", centroid)

    def resolve(self, profile: object, *, current: object | None = None) -> object:
        """Return the selected branch from the profile's cold portfolio."""

        portfolio = profile.cold_seed_portfolio(
            self.plasma_current,
            self.centroid,
            current=current,
            radius_fraction=self.radius_fraction,
            diverted_geometry=self.diverted_geometry,
        )
        index = 0 if self.requested_class == "limited" else 1
        return portfolio.branches.flux[index]

    def provenance(self) -> SolveSeedProvenance:
        """Record the selected portfolio branch and physical construction inputs."""

        return SolveSeedProvenance(
            kind="cold_seed_portfolio",
            requested_class=self.requested_class,
            plasma_current=float(self.plasma_current),
            centroid=self.centroid,
        )


@dataclass(frozen=True, slots=True)
class ForwardSolveRequest:
    """Physical inputs and one fully resolved policy for a forward solve.

    ``constraint_pairs`` is the static tuple boundary for typed augmented
    constraints.  Existing ``constraint_pins`` remain post-solve validation
    claims, so the two meanings cannot be conflated.
    """

    carrier_identity: str
    source_profile: ForwardSource
    seed_policy: ExplicitSolveSeed | ColdSeedPortfolio
    policy: ForwardSolvePolicy
    route: SolveRoute
    target_current: object | None = None
    constraint_pins: ConstraintPinSet | None = None
    constraint_pairs: tuple[ConstraintPair, ...] = ()
    current: object | None = None
    prescribed_current: object | None = None
    enforce: tuple[str, ...] = ()
    compilation_cache_hit: bool = False

    def __post_init__(self) -> None:
        """Require a self-consistent, statically shaped request."""

        if not self.carrier_identity:
            raise ValueError("carrier_identity cannot be empty")
        if self.route != self.policy.route:
            raise ValueError("request route must equal its resolved policy route")
        object.__setattr__(self, "constraint_pairs", tuple(self.constraint_pairs))
        object.__setattr__(self, "enforce", tuple(self.enforce))

    @classmethod
    def from_defaults(
        cls,
        *,
        carrier_identity: str,
        source_profile: object,
        seed_policy: ExplicitSolveSeed | ColdSeedPortfolio,
        nova_version: str = NOVA_VERSION,
        policy_overrides: Mapping[str, JsonScalar] | None = None,
        **inputs: object,
    ) -> ForwardSolveRequest:
        """Build a request from the version-keyed declaration plus deviations."""

        policy = declared_forward_solve_policy(nova_version)
        if policy_overrides:
            policy = replace(policy, **dict(policy_overrides))
        return cls(
            carrier_identity=carrier_identity,
            source_profile=source_profile,
            seed_policy=seed_policy,
            policy=policy,
            route=policy.route,
            **inputs,
        )


@dataclass(frozen=True, slots=True)
class ResolvedForwardSolveDefaults:
    """Versioned policy values and every deviation that actually ran."""

    nova_version: str
    policy: ForwardSolvePolicy
    deviations: tuple[tuple[str, JsonScalar], ...]
    compilation_cache_directory: str | None

    @classmethod
    def from_policy(
        cls,
        policy: ForwardSolvePolicy,
        *,
        nova_version: str = NOVA_VERSION,
        compilation_cache_directory: str | None = None,
    ) -> ResolvedForwardSolveDefaults:
        """Compare one resolved policy with its version's declared defaults."""

        default = declared_forward_solve_policy(nova_version)
        default_values = default.to_dict()
        actual_values = policy.to_dict()
        deviations = tuple(
            (name, actual_values[name])
            for name in actual_values
            if actual_values[name] != default_values[name]
        )
        return cls(
            nova_version=nova_version,
            policy=policy,
            deviations=deviations,
            compilation_cache_directory=compilation_cache_directory,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON receipt block."""

        return {
            "nova_version": self.nova_version,
            "policy": self.policy.to_dict(),
            "deviations": dict(self.deviations),
            "compilation_cache_directory": self.compilation_cache_directory,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResolvedForwardSolveDefaults:
        """Restore a resolved-defaults block after JSON transport."""

        expected = {
            "nova_version",
            "policy",
            "deviations",
            "compilation_cache_directory",
        }
        if set(payload) != expected:
            raise ValueError(
                "resolved defaults need version, policy, deviations, "
                "and cache directory"
            )
        policy_payload = payload["policy"]
        deviation_payload = payload["deviations"]
        if not isinstance(policy_payload, Mapping) or not isinstance(
            deviation_payload, Mapping
        ):
            raise TypeError("policy and deviations must be mappings")
        policy = ForwardSolvePolicy.from_dict(policy_payload)
        cache_directory = payload["compilation_cache_directory"]
        if cache_directory is not None and not isinstance(cache_directory, str):
            raise TypeError("compilation cache directory must be a string or null")
        restored = cls.from_policy(
            policy,
            nova_version=str(payload["nova_version"]),
            compilation_cache_directory=cache_directory,
        )
        if dict(restored.deviations) != dict(deviation_payload):
            raise ValueError("resolved-default deviations disagree with the policy")
        return restored


@dataclass(frozen=True, slots=True)
class ForwardSolveReceipt:
    """Terminal forward state together with numerical and provenance telemetry."""

    terminal_state: ForwardEquilibrium
    qualified: jax.Array | bool
    termination_reason: jax.Array | int
    residual_history: jax.Array
    mask_history: jax.Array
    globalisation_decisions: tuple[jax.Array, jax.Array]
    amplitude_history: jax.Array
    topology_read: ForwardTopologyState | None
    polish_receipt: Mapping[str, jax.Array] | None
    compilation_cache_hit: bool
    wall_seconds: float
    resolved_defaults: ResolvedForwardSolveDefaults
    seed_provenance: SolveSeedProvenance | None = None

    @property
    def equilibrium(self) -> ForwardEquilibrium:
        """Return the terminal equilibrium under its domain-specific name."""

        return self.terminal_state

    @property
    def constraints(self) -> tuple[ConstraintRecord, ...]:
        """Return terminal augmented-row records in request tuple order."""

        return self.terminal_state.constraints


__all__ = [
    "ColdSeedPortfolio",
    "ExplicitSolveSeed",
    "FORWARD_SOLVE_DEFAULTS",
    "ForwardSolveMemberData",
    "ForwardSolvePolicy",
    "ForwardSolveReceipt",
    "ForwardSolveRequest",
    "ResolvedForwardSolveDefaults",
    "SampledFluxFunction",
    "SolveSeedProvenance",
    "SolveRoute",
    "declared_forward_solve_policy",
    "default_forward_compilation_cache_root",
    "resolve_forward_solve_policy",
]
