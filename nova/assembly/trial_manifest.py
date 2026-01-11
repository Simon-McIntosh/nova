"""Manage trial simulation manifest for parameter tracking and reproducibility."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import ClassVar

import yaml


@dataclass
class TrialManifest:
    """Manage trial simulation manifest.

    Provides named parameter sets for Monte Carlo trial simulations.
    The cache hash is computed at runtime by Trial.group_name.

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
    """

    name: str | None = None
    trial_type: str = "vault"
    samples: int | None = None
    theta: list[float] | None = None
    components: list[str] | None = None
    pdf: list[str] | None = None
    adjust_gap: bool = True
    max_nominal_gap: float = 2.0
    measured_sectors: list[int] | None = None
    description: str = ""
    manifest_path: Path | None = None

    _manifest: dict = field(init=False, repr=False, default_factory=dict)

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
                # Measured sectors (optional - coils to load from pit)
                if self.measured_sectors is None:
                    self.measured_sectors = trial_data.get("measured_sectors")
                if self.trial_type == "vault":
                    if "adjust_gap" in trial_data:
                        self.adjust_gap = trial_data["adjust_gap"]
                    if "max_nominal_gap" in trial_data:
                        self.max_nominal_gap = trial_data["max_nominal_gap"]
                if not self.description:
                    self.description = entry.get("description", "")

                # Apply defaults for any still-missing values
                self._apply_defaults()
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

        # Case 3: No name - require parameters for new entry
        if self.name is None:
            self._apply_defaults()
            if not self._has_parameters():
                raise ValueError(
                    "Must provide either a simulation name or complete parameters."
                )

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
        print(f"Description: {self.description or '(none)'}")
        print(f"{'=' * 60}\n")

        if self.components and self.theta and self.pdf:
            rows = []
            for comp, th, p in zip(self.components, self.theta, self.pdf):
                rows.append({"component": comp, "theta": th, "pdf": p})
            print(tabulate(rows, headers="keys", tablefmt="simple"))
        print()
