"""Schema-validated declarative configuration for a Gaussian-process fit.

A fit configuration is authored in YAML and captures every control that
determines a fit's outcome: kernel length scales, the observation nugget,
per-component weights, constraint switches and fiducial-index constraint sets,
a radial offset, the augment and reference version selections, and the random
seed. Unknown top-level keys are rejected so a typo cannot silently disable a
control; a free-form ``extra`` mapping provides a forward-compatible sink for
values newer tooling may add.

The configuration is content-addressable: :attr:`FitConfig.sha256` is the
digest of its normalized canonical-YAML bytes, so a run record can pin the
exact configuration that produced it.
"""

from dataclasses import dataclass, field
import os

from nova.assembly.provenance import digest, yamlio

_DEFAULT_WEIGHTS = (1.0, 1.0, 0.25)


def _as_float(value, name: str) -> float:
    """Coerce a scalar to float, rejecting booleans and non-numbers.

    Parameters
    ----------
    value : object
        Candidate value.
    name : str
        Field name for error messages.

    Returns
    -------
    float
        The value as a float.

    Raises
    ------
    TypeError
        If the value is a bool or is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _as_float_list(value, name: str) -> list[float]:
    """Coerce a sequence to a list of floats.

    Parameters
    ----------
    value : object
        Candidate sequence.
    name : str
        Field name for error messages.

    Returns
    -------
    list[float]
        The coerced list.

    Raises
    ------
    TypeError
        If the value is not a list or contains a non-number.
    """
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")
    return [_as_float(item, f"{name}[{i}]") for i, item in enumerate(value)]


@dataclass
class FitConfig:
    """Declarative Gaussian-process fit configuration.

    Parameters
    ----------
    length_scales : list[float] or None
        Kernel length scale per input dimension.
    nugget : float
        Non-negative observation noise variance added to the kernel diagonal.
    weights : list[float]
        Per-component observation weights, for example ``[1.0, 1.0, 0.25]``.
    constraints : dict[str, bool]
        Named constraint switches.
    fiducial_index : list[int] or None
        Indices of fiducials that participate in constraint sets.
    radial_offset : float
        Radial offset applied to the fit geometry.
    augment_version : str or None
        Version selection for the augmenting dataset.
    reference_version : str or None
        Version selection for the reference dataset.
    random_state : int or None
        Random seed for reproducible draws.
    extra : dict
        Forward-compatible sink for values outside the declared schema.
    """

    length_scales: list[float] | None = None
    nugget: float = 0.0
    weights: list[float] = field(default_factory=lambda: list(_DEFAULT_WEIGHTS))
    constraints: dict[str, bool] = field(default_factory=dict)
    fiducial_index: list[int] | None = None
    radial_offset: float = 0.0
    augment_version: str | None = None
    reference_version: str | None = None
    random_state: int | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "FitConfig":
        """Validate and construct a configuration from a mapping.

        Parameters
        ----------
        payload : dict
            Raw configuration mapping, typically loaded from YAML.

        Returns
        -------
        FitConfig
            The validated, normalized configuration.

        Raises
        ------
        ValueError
            If an unknown top-level key is present or a value fails validation.
        TypeError
            If a field has the wrong type.
        """
        if not isinstance(payload, dict):
            raise TypeError("fit configuration must be a mapping")

        known = {
            "length_scales",
            "nugget",
            "weights",
            "constraints",
            "fiducial_index",
            "radial_offset",
            "augment_version",
            "reference_version",
            "random_state",
            "extra",
        }
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"unknown top-level keys (place forward-compat values under "
                f"'extra'): {sorted(unknown)}"
            )

        length_scales = payload.get("length_scales")
        if length_scales is not None:
            length_scales = _as_float_list(length_scales, "length_scales")

        nugget = _as_float(payload.get("nugget", 0.0), "nugget")
        if nugget < 0:
            raise ValueError(f"nugget must be non-negative, got {nugget}")

        weights = _as_float_list(
            payload.get("weights", list(_DEFAULT_WEIGHTS)), "weights"
        )

        constraints = payload.get("constraints", {})
        if not isinstance(constraints, dict):
            raise TypeError("constraints must be a mapping of name to bool")
        for key, value in constraints.items():
            if not isinstance(key, str) or not isinstance(value, bool):
                raise TypeError("constraints must map str names to bool switches")

        fiducial_index = payload.get("fiducial_index")
        if fiducial_index is not None:
            if not isinstance(fiducial_index, list) or any(
                isinstance(i, bool) or not isinstance(i, int) for i in fiducial_index
            ):
                raise TypeError("fiducial_index must be a list of ints")

        radial_offset = _as_float(payload.get("radial_offset", 0.0), "radial_offset")

        augment_version = _check_optional_str(
            payload.get("augment_version"), "augment_version"
        )
        reference_version = _check_optional_str(
            payload.get("reference_version"), "reference_version"
        )

        random_state = payload.get("random_state")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise TypeError("random_state must be an int or None")

        extra = payload.get("extra", {})
        if not isinstance(extra, dict):
            raise TypeError("extra must be a mapping")

        return cls(
            length_scales=length_scales,
            nugget=nugget,
            weights=weights,
            constraints=dict(constraints),
            fiducial_index=list(fiducial_index) if fiducial_index is not None else None,
            radial_offset=radial_offset,
            augment_version=augment_version,
            reference_version=reference_version,
            random_state=random_state,
            extra=dict(extra),
        )

    def to_dict(self) -> dict:
        """Return the normalized, YAML-serializable configuration mapping.

        Returns
        -------
        dict
            Mapping with normalized scalar and list values.
        """
        return {
            "length_scales": self.length_scales,
            "nugget": self.nugget,
            "weights": self.weights,
            "constraints": self.constraints,
            "fiducial_index": self.fiducial_index,
            "radial_offset": self.radial_offset,
            "augment_version": self.augment_version,
            "reference_version": self.reference_version,
            "random_state": self.random_state,
            "extra": self.extra,
        }

    @property
    def sha256(self) -> str:
        """Content digest of the normalized canonical-YAML bytes.

        Returns
        -------
        str
            Digest string of the form ``"sha256:<hexdigest>"``.
        """
        return digest.digest_bytes(yamlio.canonical_yaml_bytes(self.to_dict()))

    def write(self, path: os.PathLike | str) -> None:
        """Write the configuration to a file as canonical YAML.

        Parameters
        ----------
        path : os.PathLike or str
            Destination file.
        """
        yamlio.dump_yaml(self.to_dict(), path)

    @classmethod
    def load(cls, path: os.PathLike | str) -> "FitConfig":
        """Load and validate a configuration from a YAML file.

        Parameters
        ----------
        path : os.PathLike or str
            Source file.

        Returns
        -------
        FitConfig
            The validated configuration.
        """
        return cls.from_dict(yamlio.load_yaml(path))


def _check_optional_str(value, name: str) -> str | None:
    """Validate an optional string field.

    Parameters
    ----------
    value : object
        Candidate value.
    name : str
        Field name for error messages.

    Returns
    -------
    str or None
        The validated value.

    Raises
    ------
    TypeError
        If the value is neither ``None`` nor a string.
    """
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return value
