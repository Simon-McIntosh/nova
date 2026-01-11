"""Manage trial simulation manifest for parameter tracking and reproducibility."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import ClassVar

import xxhash
import yaml


@dataclass
class TrialManifest:
    """Manage trial simulation manifest.

    Provides parameter tracking, hash-based caching, and reproducibility
    for Monte Carlo trial simulations.

    Initialization modes:
    1. By name only: Load existing simulation by label
    2. By hash only: Load existing simulation by hash
    3. By parameters only: Match existing or fail if no name given
    4. By name + parameters: Create new or update existing entry
    5. By name + hash: Validate hash matches stored parameters

    The hash cannot be set explicitly - it is computed from parameters.

    Parameters
    ----------
    name : str | None
        Simulation label for human-readable identification
    trial_type : str
        Type of trial: 'vault' or 'error_field'
    samples : int | None
        Number of Monte Carlo samples
    theta : list[float] | None
        Half-width parameters for each component
    components : list[str] | None
        Component names
    pdf : list[str] | None
        PDF types for each component ('uniform' or 'normal')
    adjust_gap : bool
        Whether to adjust gap (Vault only)
    max_nominal_gap : float
        Maximum nominal gap (Vault only)
    description : str
        Human-readable description
    manifest_path : Path | None
        Path to manifest YAML file

    Attributes
    ----------
    hash : str
        Computed xxhash32 of input parameters (read-only)
    """

    name: str | None = None
    trial_type: str = "vault"
    samples: int | None = None
    theta: list[float] | None = None
    components: list[str] | None = None
    pdf: list[str] | None = None
    adjust_gap: bool = True
    max_nominal_gap: float = 2.0
    description: str = ""
    manifest_path: Path | None = None

    _manifest: dict = field(init=False, repr=False, default_factory=dict)
    _hash: str | None = field(init=False, repr=False, default=None)

    DEFAULT_MANIFEST: ClassVar[Path] = Path(__file__).parent / "trial_manifest.yml"

    def __post_init__(self):
        """Initialize manifest and resolve parameters."""
        if self.manifest_path is None:
            self.manifest_path = self.DEFAULT_MANIFEST
        self._load_manifest()
        self._resolve_parameters()

    def _load_manifest(self) -> None:
        """Load manifest from YAML file."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self._manifest = yaml.safe_load(f) or {}
        else:
            self._manifest = {"simulations": {}}

    def _save_manifest(self) -> None:
        """Save manifest to YAML file."""
        with open(self.manifest_path, "w") as f:
            yaml.dump(self._manifest, f, default_flow_style=False, sort_keys=False)

    def _compute_hash(self) -> str:
        """Compute hash from current parameters."""
        hasher = xxhash.xxh32()
        hasher.update(str(self.samples).encode())
        hasher.update(str(self.theta).encode())
        hasher.update(str(self.components).encode())
        hasher.update(str(self.pdf).encode())
        if self.trial_type == "vault":
            hasher.update(str(self.adjust_gap).encode())
            hasher.update(str(self.max_nominal_gap).encode())
        return hasher.hexdigest()

    @property
    def hash(self) -> str | None:
        """Return computed hash (read-only)."""
        if self._hash is None and self._has_parameters():
            self._hash = self._compute_hash()
        return self._hash

    def _has_parameters(self) -> bool:
        """Check if required parameters are set."""
        return all(
            [
                self.samples is not None,
                self.theta is not None,
                self.components is not None,
                self.pdf is not None,
            ]
        )

    def _get_defaults(self) -> dict:
        """Get default values for trial type from manifest."""
        defaults = self._manifest.get("defaults", {}).get(self.trial_type, {})
        return defaults

    def _apply_defaults(self) -> None:
        """Apply default values from manifest for missing parameters."""
        defaults = self._get_defaults()
        if self.components is None:
            self.components = defaults.get("components")
        if self.pdf is None:
            self.pdf = defaults.get("pdf")
        if self.trial_type == "vault":
            if "adjust_gap" in defaults and self.adjust_gap is True:
                self.adjust_gap = defaults.get("adjust_gap", True)
            if "max_nominal_gap" in defaults:
                self.max_nominal_gap = defaults.get("max_nominal_gap", 2.0)

    def _resolve_parameters(self) -> None:
        """Resolve parameters from name, hash, or direct input."""
        simulations = self._manifest.get("simulations", {})

        # Case 1: Name provided - load from manifest
        if self.name is not None and self.name in simulations:
            entry = simulations[self.name]
            trial_data = entry.get(self.trial_type, {})

            if trial_data:
                # Load stored parameters if not explicitly provided
                if self.samples is None:
                    self.samples = trial_data.get("samples")
                if self.theta is None:
                    self.theta = trial_data.get("theta")
                # Components/pdf from trial data or defaults
                if self.components is None:
                    self.components = trial_data.get("components")
                if self.pdf is None:
                    self.pdf = trial_data.get("pdf")
                if self.trial_type == "vault":
                    if "adjust_gap" in trial_data:
                        self.adjust_gap = trial_data["adjust_gap"]
                    if "max_nominal_gap" in trial_data:
                        self.max_nominal_gap = trial_data["max_nominal_gap"]
                if not self.description:
                    self.description = entry.get("description", "")

                # Apply defaults for any still-missing values
                self._apply_defaults()

                # Validate hash if stored
                stored_hash = trial_data.get("hash")
                if stored_hash and self._has_parameters():
                    computed = self._compute_hash()
                    if stored_hash != computed:
                        raise ValueError(
                            f"Hash mismatch for '{self.name}': "
                            f"stored={stored_hash}, computed={computed}"
                        )
            return

        # Case 2: Name provided but not in manifest - require parameters
        if self.name is not None and self.name not in simulations:
            self._apply_defaults()
            if not self._has_parameters():
                raise ValueError(
                    f"Simulation '{self.name}' not found in manifest. "
                    "Provide parameters to create a new entry."
                )
            return

        # Case 3: No name, but parameters provided - find by hash
        if self.name is None:
            self._apply_defaults()
            if self._has_parameters():
                computed_hash = self._compute_hash()
                for sim_name, entry in simulations.items():
                    trial_data = entry.get(self.trial_type, {})
                    if trial_data.get("hash") == computed_hash:
                        self.name = sim_name
                        self.description = entry.get("description", "")
                        return
                # No match found - fail if no name given
                raise ValueError(
                    f"No simulation found with hash {computed_hash}. "
                    "Provide a name to create a new entry."
                )

        # Case 4: Neither name nor complete parameters
        if self.name is None and not self._has_parameters():
            raise ValueError(
                "Must provide either a simulation name or complete parameters."
            )

    @classmethod
    def from_hash(
        cls,
        hash_value: str,
        trial_type: str = "vault",
        manifest_path: Path | None = None,
    ) -> "TrialManifest":
        """Load simulation by hash.

        Parameters
        ----------
        hash_value : str
            The hash to search for
        trial_type : str
            Type of trial: 'vault' or 'error_field'
        manifest_path : Path | None
            Path to manifest YAML file

        Returns
        -------
        TrialManifest
            Manifest instance with loaded parameters

        Raises
        ------
        ValueError
            If no simulation with the given hash is found
        """
        path = manifest_path or cls.DEFAULT_MANIFEST
        with open(path) as f:
            manifest = yaml.safe_load(f) or {}

        simulations = manifest.get("simulations", {})
        for sim_name, entry in simulations.items():
            trial_data = entry.get(trial_type, {})
            if trial_data.get("hash") == hash_value:
                return cls(
                    name=sim_name,
                    trial_type=trial_type,
                    manifest_path=path,
                )

        raise ValueError(f"No simulation found with hash {hash_value}")

    def save(self) -> None:
        """Save current parameters to manifest.

        Raises
        ------
        ValueError
            If name is not set or parameters are incomplete
        """
        if self.name is None:
            raise ValueError("Cannot save without a simulation name")
        if not self._has_parameters():
            raise ValueError("Cannot save with incomplete parameters")

        simulations = self._manifest.setdefault("simulations", {})

        # Create or update entry
        if self.name not in simulations:
            simulations[self.name] = {
                "description": self.description,
                "date": date.today().isoformat(),
            }

        entry = simulations[self.name]

        # Update trial-specific data
        entry[self.trial_type] = {
            "samples": self.samples,
            "theta": self.theta,
            "components": self.components,
            "pdf": self.pdf,
            "hash": self.hash,
        }
        if self.trial_type == "vault":
            entry[self.trial_type]["adjust_gap"] = self.adjust_gap
            entry[self.trial_type]["max_nominal_gap"] = self.max_nominal_gap

        self._save_manifest()

    def to_dict(self) -> dict:
        """Return parameters as dictionary."""
        result = {
            "name": self.name,
            "trial_type": self.trial_type,
            "samples": self.samples,
            "theta": self.theta,
            "components": self.components,
            "pdf": self.pdf,
            "hash": self.hash,
            "description": self.description,
        }
        if self.trial_type == "vault":
            result["adjust_gap"] = self.adjust_gap
            result["max_nominal_gap"] = self.max_nominal_gap
        return result

    @classmethod
    def list_simulations(
        cls, trial_type: str | None = None, manifest_path: Path | None = None
    ) -> list[dict]:
        """List all simulations in the manifest.

        Parameters
        ----------
        trial_type : str | None
            Filter by trial type ('vault' or 'error_field')
        manifest_path : Path | None
            Path to manifest YAML file

        Returns
        -------
        list[dict]
            List of simulation summaries
        """
        path = manifest_path or cls.DEFAULT_MANIFEST
        with open(path) as f:
            manifest = yaml.safe_load(f) or {}

        results = []
        for name, entry in manifest.get("simulations", {}).items():
            for tt in ["vault", "error_field"]:
                if trial_type is not None and tt != trial_type:
                    continue
                if tt in entry:
                    results.append(
                        {
                            "name": name,
                            "trial_type": tt,
                            "description": entry.get("description", ""),
                            "date": entry.get("date"),
                            "hash": entry[tt].get("hash"),
                            "samples": entry[tt].get("samples"),
                        }
                    )
        return results

    def print_summary(self) -> None:
        """Print summary of current parameters."""
        from tabulate import tabulate

        print(f"\n{'=' * 60}")
        print(f"Trial: {self.name or '(unnamed)'}")
        print(f"Type: {self.trial_type}")
        print(f"Hash: {self.hash or '(not computed)'}")
        print(f"Description: {self.description or '(none)'}")
        print(f"{'=' * 60}\n")

        if self.components and self.theta and self.pdf:
            rows = []
            for comp, th, p in zip(self.components, self.theta, self.pdf):
                rows.append({"component": comp, "theta": th, "pdf": p})
            print(tabulate(rows, headers="keys", tablefmt="simple"))
        print()
