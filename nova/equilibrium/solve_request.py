"""Typed public inputs and provenance for one forward equilibrium solve."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import os
from pathlib import Path
import socket
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

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


def default_forward_compilation_cache_root() -> Path:
    """Return a per-user, per-host root on the runtime temporary filesystem."""

    temporary_root = Path(os.environ.get("TMPDIR") or "/tmp").expanduser().resolve()
    return (
        temporary_root
        / "nova-forward-cache"
        / f"user-{os.getuid()}"
        / f"host-{socket.gethostname()}"
    )


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

    def resolve(self, _profile: object) -> object:
        """Return the state unchanged so its dtype and bytes remain authoritative."""

        return self.state


@dataclass(frozen=True, slots=True)
class ForwardSolveRequest:
    """Physical inputs and one fully resolved policy for a forward solve.

    ``constraint_pairs`` is the static tuple boundary for typed augmented
    constraints.  Existing ``constraint_pins`` remain post-solve validation
    claims, so the two meanings cannot be conflated.
    """

    carrier_identity: str
    source_profile: ForwardSource
    seed_policy: ExplicitSolveSeed
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
        seed_policy: ExplicitSolveSeed,
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

    @property
    def equilibrium(self) -> ForwardEquilibrium:
        """Return the terminal equilibrium under its domain-specific name."""

        return self.terminal_state

    @property
    def constraints(self) -> tuple[ConstraintRecord, ...]:
        """Return terminal augmented-row records in request tuple order."""

        return self.terminal_state.constraints


__all__ = [
    "ExplicitSolveSeed",
    "FORWARD_SOLVE_DEFAULTS",
    "ForwardSolvePolicy",
    "ForwardSolveReceipt",
    "ForwardSolveRequest",
    "ResolvedForwardSolveDefaults",
    "SolveRoute",
    "declared_forward_solve_policy",
    "default_forward_compilation_cache_root",
    "resolve_forward_solve_policy",
]
