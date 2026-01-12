"""Run Monte Carlo simulations for candidate vault assemblies."""

from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from functools import cached_property
from time import time
from typing import ClassVar, Self

import numpy as np
import xarray
import xxhash

from nova.assembly import structural, electromagnetic, overlap
from nova.assembly.gap import WedgeGap
from nova.assembly.model import Dataset
from nova.assembly.progress import TrialProgress
from nova.assembly.trial_manifest import TrialManifest
from nova.graphics.plot import Plot1D


@dataclass
class TrialAttrs:
    """Manage trial attributes."""

    samples: int = 100_000
    component: list[str] = field(default_factory=list)
    theta: list[float] = field(default_factory=list)
    pdf: list[str] = field(default_factory=list)
    adjust_gap: bool = True
    max_nominal_gap: float = 2.0
    sead: int = 2025
    chunk_size: int = field(default=50_000, repr=False)
    measured_sectors: list[int] | None = field(default=None)
    fixed_coils: dict | None = field(default=None, repr=False)
    force: bool = field(default=False, repr=False)

    ncoil: ClassVar[int] = 18

    @cached_property
    def field_names(self):
        """Return list of field names."""
        return [attr.name for attr in fields(TrialAttrs)]

    @property
    def n_chunks(self) -> int:
        """Return number of chunks for processing."""
        return (self.samples + self.chunk_size - 1) // self.chunk_size

    @property
    def chunk_ranges(self) -> list[tuple[int, int]]:
        """Return list of (start, end) tuples for each chunk."""
        ranges = []
        for i in range(self.n_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, self.samples)
            ranges.append((start, end))
        return ranges

    @property
    def attrs(self):
        """Return trial attrs."""
        # Exclude transient fields from serialization
        exclude = {"fixed_coils", "force"}
        attrs = {}
        for attr in self.field_names:
            if attr in exclude:
                continue
            value = getattr(self, attr)
            if value is None:
                continue
            if isinstance(value, bool):
                attrs[attr] = int(value)
                continue
            if not isinstance(value, list):
                attrs[attr] = value
        return attrs


@dataclass
class Trial(Dataset, TrialAttrs, Plot1D):
    """Run stastistical analysis on trial assemblies."""

    filename: str = "trial"
    xxh32: xxhash.xxh32 = field(repr=False, init=False, default_factory=xxhash.xxh32)

    def __post_init__(self):
        """Set dataset group for netCDF file load/store."""
        self.group = self.group_name
        self.rng = np.random.default_rng(self.sead)
        # Initialize FilePath (sets fsys) before potential early build
        self.host = self.hostname
        self.path = self.dirname
        if self.force:
            self.build()
        else:
            super().__post_init__()

    @property
    def nominal_gap(self):
        """Retrun nominal gap."""
        try:
            return self.data.attrs["nominal_gap"]
        except KeyError:
            return self.max_nominal_gap

    @nominal_gap.setter
    def nominal_gap(self, nominal_gap):
        self.data.attrs["nominal_gap"] = nominal_gap

    @property
    def group_name(self):
        """Return group name as xxh32 hex hash.

        The hash includes measured_sectors only when specified, so existing
        cached data for simulations without measured sectors is preserved.
        """
        self.xxh32.reset()
        hash_data = list(self.attrs.values()) + self.theta + self.pdf
        # Include measured_sectors in hash only if specified
        if self.measured_sectors is not None:
            hash_data.extend(sorted(self.measured_sectors))
        self.xxh32.update(np.array(hash_data).tobytes())
        return self.xxh32.hexdigest()

    @classmethod
    def from_manifest(
        cls,
        name: str | None = None,
        samples: int | None = None,
        force: bool = False,
        chunk_size: int | None = None,
        **kwargs,
    ) -> Self:
        """Load trial from manifest by name or create with parameters.

        Parameters
        ----------
        name : str | None
            Simulation label to load from manifest
        samples : int | None
            Override samples from manifest
        force : bool
            Force rebuild even if cached data exists
        chunk_size : int | None
            Process samples in chunks of this size for memory efficiency
        **kwargs
            Additional parameters passed to Trial/TrialManifest

        Returns
        -------
        Trial
            Trial instance with loaded parameters

        Examples
        --------
        >>> trial = Vault.from_manifest("baseline_2021")
        >>> trial = ErrorField.from_manifest("baseline_2021", samples=500000)
        >>> trial = Vault.from_manifest("baseline_2021", force=True)  # rebuild
        >>> trial = Vault.from_manifest("baseline_2021", chunk_size=50000)
        """
        # Determine trial_type from class name
        trial_type = "error_field" if cls.__name__ == "ErrorField" else "vault"

        manifest = TrialManifest(name=name, trial_type=trial_type)

        # Override samples if provided
        if samples is not None:
            manifest.samples = samples

        # Build kwargs for Trial constructor
        trial_kwargs = {
            "samples": manifest.samples,
            "theta": manifest.theta,
            "component": manifest.components,
            "pdf": manifest.pdf,
            "force": force,
        }

        # Add chunk_size if specified
        if chunk_size is not None:
            trial_kwargs["chunk_size"] = chunk_size

        # Add measured_sectors if specified
        if manifest.measured_sectors is not None:
            trial_kwargs["measured_sectors"] = manifest.measured_sectors

        # Add vault-specific parameters
        if trial_type == "vault":
            trial_kwargs["adjust_gap"] = manifest.adjust_gap
            trial_kwargs["max_nominal_gap"] = manifest.max_nominal_gap

        # Merge with any additional kwargs
        trial_kwargs.update(kwargs)

        # Pre-load measured positions before construction when force=True
        # This ensures fixed_coils is available during build in __post_init__
        if manifest.measured_sectors is not None and force:
            fixed_coils = cls._load_fixed_coils_from_sectors(manifest.measured_sectors)
            trial_kwargs["fixed_coils"] = fixed_coils

        trial = cls(**trial_kwargs)

        # Load measured coil positions if sectors specified (for non-force case)
        if trial.measured_sectors is not None and trial.fixed_coils is None:
            trial._load_measured_positions()

        return trial

    def save_to_manifest(self, name: str, description: str = "") -> None:
        """Save current trial parameters to manifest.

        Parameters
        ----------
        name : str
            Label for the simulation
        description : str
            Human-readable description
        """
        trial_type = "error_field" if "vertical" in self.component else "vault"
        manifest = TrialManifest(
            name=name,
            trial_type=trial_type,
            samples=self.samples,
            theta=self.theta,
            components=self.component,
            pdf=self.pdf,
            description=description,
        )
        manifest.save()
        print(f"Saved to manifest: {name}")

    @classmethod
    def from_pit(
        cls,
        name: str | None = None,
        samples: int | None = None,
        pit_kwargs: dict | None = None,
        **kwargs,
    ) -> Self:
        """Create hybrid trial with fixed values from installed pit sectors.

        Loads measured coil positions from FiducialPit and injects them
        as fixed values for installed coils. Remaining coils are sampled
        from distributions defined by the manifest.

        Parameters
        ----------
        name : str | None
            Manifest simulation label for theta/pdf parameters
        samples : int | None
            Override samples from manifest
        pit_kwargs : dict | None
            Arguments passed to FiducialPit (sectors, phase, pcr, etc.)
            If None, uses default installed sectors.
        **kwargs
            Additional parameters passed to Trial

        Returns
        -------
        Trial
            Trial instance with fixed_coils populated from pit measurements

        Examples
        --------
        >>> error = ErrorField.from_pit("baseline_2021", samples=100000)
        >>> error.build()  # Uses measured values for 8 installed coils
        """
        from nova.assembly.fiducialpit import FiducialPit

        # Default pit configuration for currently installed sectors
        if pit_kwargs is None:
            pit_kwargs = {
                "sectors": {6: [12, 13], 7: [8, 9], 5: [16, 5], 8: [4, 11]},
                "phase": "latest",
                "pcr": True,
            }

        # Load pit data and extract trial-formatted positions
        pit = FiducialPit(**pit_kwargs)
        positions = pit.extract_trial_positions()

        # Convert to fixed_coils dict format
        fixed_coils = {}
        for _, row in positions.iterrows():
            trial_idx = int(row["trial_index"])
            fixed_coils[trial_idx] = {
                col: row[col]
                for col in positions.columns
                if col not in ["trial_index", "coil", "sector"]
            }

        # Create trial via from_manifest with fixed_coils injected
        trial = cls.from_manifest(name=name, samples=samples, **kwargs)
        trial.fixed_coils = fixed_coils

        return trial

    @staticmethod
    def _load_fixed_coils_from_sectors(
        measured_sectors: list[int], pit_kwargs: dict | None = None
    ) -> dict:
        """Load fixed coil values from FiducialPit for given sectors.

        Static method that can be called before instance creation to
        pre-populate fixed_coils when force=True.

        Parameters
        ----------
        measured_sectors : list[int]
            List of sector numbers (5, 6, 7, 8) to load
        pit_kwargs : dict | None
            Arguments passed to FiducialPit

        Returns
        -------
        dict
            fixed_coils dict mapping trial_index to component values
        """
        from nova.assembly.fiducialpit import FiducialPit

        # Default sector to coil mapping
        sector_coils = {
            5: [16, 5],
            6: [12, 13],
            7: [8, 9],
            8: [4, 11],
        }

        if pit_kwargs is None:
            sectors = {
                s: sector_coils[s] for s in measured_sectors if s in sector_coils
            }
            pit_kwargs = {
                "sectors": sectors,
                "phase": "latest",
                "pcr": True,
            }

        pit = FiducialPit(**pit_kwargs)
        positions = pit.extract_trial_positions()

        fixed_coils = {}
        for _, row in positions.iterrows():
            trial_idx = int(row["trial_index"])
            fixed_coils[trial_idx] = {
                col: row[col]
                for col in positions.columns
                if col not in ["trial_index", "coil", "sector"]
            }
        return fixed_coils

    def _load_measured_positions(self, pit_kwargs: dict | None = None) -> None:
        """Load measured positions from FiducialPit for measured_sectors.

        Populates self.fixed_coils with measured values for coils in the
        specified measured_sectors. Called automatically by from_manifest
        when measured_sectors is specified in the manifest.

        Parameters
        ----------
        pit_kwargs : dict | None
            Arguments passed to FiducialPit. If None, builds sectors dict
            from self.measured_sectors using default coil assignments.
        """
        from nova.assembly.fiducialpit import FiducialPit

        if self.measured_sectors is None:
            return

        # Default sector to coil mapping
        sector_coils = {
            5: [16, 5],
            6: [12, 13],
            7: [8, 9],
            8: [4, 11],
        }

        if pit_kwargs is None:
            # Build sectors dict from measured_sectors
            sectors = {
                s: sector_coils[s] for s in self.measured_sectors if s in sector_coils
            }
            pit_kwargs = {
                "sectors": sectors,
                "phase": "latest",
                "pcr": True,
            }

        # Load pit data and extract trial-formatted positions
        pit = FiducialPit(**pit_kwargs)
        positions = pit.extract_trial_positions()

        # Convert to fixed_coils dict format
        self.fixed_coils = {}
        for _, row in positions.iterrows():
            trial_idx = int(row["trial_index"])
            self.fixed_coils[trial_idx] = {
                col: row[col]
                for col in positions.columns
                if col not in ["trial_index", "coil", "sector"]
            }

    def normal(self, variance: float):
        """Return sample with normal distribution."""
        scale = np.sqrt(variance)
        return self.rng.normal(scale=scale, size=(self.samples, self.ncoil))

    def uniform(self, bound: float):
        """Return sample with uniform distribution."""
        return self.rng.uniform(-bound, bound, size=(self.samples, self.ncoil))

    def normal_chunk(self, variance: float, n_samples: int):
        """Return chunk of samples with normal distribution."""
        scale = np.sqrt(variance)
        return self.rng.normal(scale=scale, size=(n_samples, self.ncoil))

    def uniform_chunk(self, variance: float, n_samples: int):
        """Return chunk of samples with uniform distribution."""
        return self.rng.uniform(-variance, variance, size=(n_samples, self.ncoil))

    def reset_rng(self):
        """Reset RNG to initial seed state."""
        self.rng = np.random.default_rng(self.sead)

    def generate_chunk_signals(self, n_samples: int) -> dict[str, np.ndarray]:
        """Generate signal arrays for a chunk of samples.

        This consumes RNG in a different order than non-chunked builds,
        producing statistically equivalent but not bit-for-bit identical
        results. For exact reproducibility, use file caching with non-chunked
        builds.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate for this chunk

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping component names to signal arrays of shape
            (n_samples, ncoil)
        """
        # Temporarily override self.samples for generation
        original_samples = self.samples
        self.samples = n_samples

        signals = {}
        for i, component in enumerate(self.component):
            theta = self.theta[i]
            pdf = self.pdf[i]
            samples = getattr(self, pdf)(theta)

            # Inject fixed values for measured coils
            if self.fixed_coils is not None:
                for trial_idx, coil_data in self.fixed_coils.items():
                    if component in coil_data:
                        samples[:, trial_idx] = coil_data[component]

            signals[component] = samples

        # Restore original samples count
        self.samples = original_samples
        return signals

    def initialize_dataset(self):
        """Initialize empty dataset with coordinates but no signal data."""
        self.data = xarray.Dataset(attrs=self.attrs)
        self.data["sample"] = range(self.samples)
        self.data["index"] = range(self.ncoil)
        self.data["component"] = self.component
        self.data["signal"] = ["radial", "tangential"]
        self.data["coordinate"] = ["x", "y"]
        self.data["theta"] = "component", self.theta
        self.data["pdf"] = "component", self.pdf

    def build_signal(self):
        """Build input distributions.

        If fixed_coils is set, measured values are injected for specified
        coil indices. Fixed coils use their measured values (broadcast across
        all samples) while remaining coils are sampled from distributions.

        The fixed_coils dict should have structure:
            {trial_index: {component: value, ...}, ...}
        where trial_index is 0-17 and component names match self.component.
        """
        self.data = xarray.Dataset(attrs=self.attrs)
        self.data["sample"] = range(self.samples)
        self.data["index"] = range(self.ncoil)
        for component in self.component:
            self.data[component] = xarray.DataArray(0.0, self.data.coords)
        self.data["component"] = self.component
        self.data["signal"] = ["radial", "tangential"]
        self.data["coordinate"] = ["x", "y"]
        self.data["theta"] = "component", self.theta
        self.data["pdf"] = "component", self.pdf
        for i, component in enumerate(self.component):
            theta = self.theta[i]
            pdf = self.pdf[i]
            samples = getattr(self, pdf)(theta)

            # Inject fixed values for measured coils
            if self.fixed_coils is not None:
                for trial_idx, coil_data in self.fixed_coils.items():
                    if component in coil_data:
                        samples[:, trial_idx] = coil_data[component]

            self.data[component] = ("sample", "index"), samples

    @cached_property
    def gap(self):
        """Return gap samples."""
        return self.data.gap.sum(axis=-1).data + self.nominal_gap

    @cached_property
    def cumulative_gap(self):
        """Return cumulative gap."""
        return self.gap.sum(axis=-1)

    def build_positive_gap(self, nmax=20, eps=1e-3, progress_task=None):
        """Built gap waveform via iterative loop.

        Parameters
        ----------
        nmax : int
            Maximum number of iterations
        eps : float
            Convergence tolerance for negative gaps
        progress_task : StepTask | None
            Optional progress task for tracking iterations
        """
        self.build_gap()
        for i in range(nmax):
            if self.adjust_gap:
                self.adjust_nominal_gap()
            gap = self.gap
            sample_index = (gap < -eps).any(axis=1)
            if sample_index.sum() == 0:
                if progress_task is not None:
                    progress_task.update(completed=nmax)
                return
            offset = gap[sample_index]
            offset[offset >= 0] = 0
            self.data.tangential[sample_index] += offset
            self.build_gap()
            if progress_task is not None:
                progress_task.advance()
        raise ValueError(
            f"gap itteration failure at iteration {nmax} "
            "negitive samples "
            f"{100 * sample_index.sum() / len(gap):1.0f}%"
        )

    def build_gap(self):
        """Build vault gap from radial and toroidal waveforms."""
        self.data["gap"] = (
            ("sample", "index", "signal"),
            np.zeros(
                (self.data.sizes["sample"], self.ncoil, self.data.sizes["signal"])
            ),
        )
        self.data.gap[..., 0] = np.pi / self.ncoil * self.data["radial"]
        self.data.gap[:, :-1, 0] += np.pi / self.ncoil * self.data["radial"][:, 1:].data
        self.data.gap[:, -1, 0] += np.pi / self.ncoil * self.data["radial"][:, 0].data
        self.data.gap[..., 1] = -self.data["tangential"]
        self.data.gap[:, :-1, 1] += self.data["tangential"][:, 1:].data
        self.data.gap[:, -1, 1] += self.data["tangential"][:, 0].data
        try:
            delattr(self, "gap")
            delattr(self, "cumulative_gap")
        except AttributeError:
            pass

    def adjust_nominal_gap(self):
        """Adjust nominal gap to ensure a cumulative gap below threshold."""
        gap_quantile = np.quantile(self.cumulative_gap, 0.99)
        self.nominal_gap -= gap_quantile / self.ncoil - self.max_nominal_gap

    @contextmanager
    def timer(self):
        """Time build."""
        start_time = time()
        yield
        print(f"build time {time() - start_time:1.0f}s")

    @contextmanager
    def progress(self, steps: list[str]):
        """Track build progress with rich display.

        Parameters
        ----------
        steps : list[str]
            Names of the build steps to track

        Yields
        ------
        TrialProgress
            Progress monitor with step() context manager
        """
        description = f"{self.__class__.__name__} ({self.samples:,} samples)"
        with TrialProgress(description) as monitor:
            monitor.configure(steps)
            yield monitor

    def pdf_text(self, wall=False, fancy=False):
        """Return pdf text label."""
        text = ""
        for i, component in enumerate(self.component):
            if component == "wall" and not wall:
                continue
            if fancy:
                attr = component.split("_")[-1]
                if attr in ["radial", "tangential"]:
                    attr = attr[0]
                    if attr == "t":
                        text += r"$r$"
                        attr = r"\phi"
                    text += rf"$\Delta {attr}_{{{component.split('_')[0]}}}$"
                else:
                    text += component.split("_")[-1]
            else:
                text += component
            theta = self.data.theta[i].data
            if self.data.pdf[i] == "normal":
                pdf = rf"$\mathcal{{N}}\,(0, {theta:1.1f})$"
            elif self.data.pdf[i] == "uniform":
                pdf = rf"$\mathcal{{U}}\,(\pm{theta:1.1f})$"
            text += ": " + pdf
            text += "\n"
        text += "\n"
        text += f"samples: {self.samples:,}"
        self.text(text)

    def text(self, text):
        """Add multi-line text to current axes."""
        self.axes.text(
            1.0,
            0.8,
            text,
            fontsize="x-small",
            transform=self.axes.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="w", boxstyle="round, pad=0.5", linewidth=0.5),
        )

    def label_quantile(
        self, data, label: str, qth=0.99, height=0.1, color="gray", precision=1.2
    ):
        """Label quantile."""
        ylim = self.axes.get_ylim()
        yline = ylim[0] + np.array([0, height * (ylim[1] - ylim[0])])
        quantile = np.quantile(data, qth)
        self.axes.plot(quantile * np.ones(2), yline, "-", color="k", alpha=0.75)
        text = rf"q({qth:1.2f}): ${label}={quantile:{precision}f}$"
        self.axes.text(
            quantile,
            yline[1],
            text,
            ha="left",
            va="bottom",
            fontsize="small",
            color=color,
            bbox=dict(facecolor="w", edgecolor=color),
        )

    def plot_pdf(self, bins=51):
        """Plot pdf."""
        pdf, edges = np.histogram(self.data.peaktopeak, bins, density=True)
        self.axes.plot((edges[:-1] + edges[1:]) / 2, pdf)

    def sample(self, quantile, offset=True):
        """Return sample index closest to quantile."""
        label = "peaktopeak"
        if offset:
            label += "_offset"
        peaktopeak = np.quantile(self.data[label], quantile)
        return np.argmin((self.data[label].data - peaktopeak) ** 2)


@dataclass
class Vault(Trial, Plot1D):
    """Run vault assembly Monte Carlo trials."""

    filename: str = "vault_trial"
    component: list[str] = field(
        default_factory=lambda: [
            "radial",
            "tangential",
            "roll_length",
            "yaw_length",
            "radial_ccl",
            "tangential_ccl",
            "radial_wall",
        ]
    )
    theta: list[float] = field(default_factory=lambda: [1.5, 1.5, 3, 3, 2, 2, 5])
    pdf: list[str] = field(
        default_factory=lambda: [
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "normal",
            "normal",
            "uniform",
        ]
    )
    modes: int = 3
    energize: int | bool = True
    wall: int | bool = True

    def __post_init__(self):
        """Initialize model instances."""
        self.energize = int(self.energize)
        self.wall = int(self.wall)
        self.field_names += ["modes", "energize", "wall"]
        self.structural_model = structural.Model()
        self.electromagnetic_model = electromagnetic.Model()
        super().__post_init__()

    def build(self):
        """Build Monte Carlo dataset."""
        if self.samples > self.chunk_size:
            return self.build_chunked()
        return self.build_full()

    def build_full(self):
        """Build Monte Carlo dataset (non-chunked, all in memory)."""
        steps = [
            "Signal generation",
            "Gap optimization",
            "Structural",
            "Electromagnetic",
        ]
        if self.wall:
            steps.append("Wall prediction")
        steps.append("Store results")

        with self.progress(steps) as monitor:
            with monitor.step("Signal generation"):
                self.build_signal()
            with monitor.step("Gap optimization", total=20) as task:
                self.build_positive_gap(progress_task=task)
            with monitor.step("Structural"):
                self.predict_structure()
            with monitor.step("Electromagnetic"):
                self.predict_electromagnetic()
            if self.wall:
                with monitor.step("Wall prediction"):
                    self.predict_wall()
            with monitor.step("Store results"):
                self.store()
        return self

    def build_chunked(self):
        """Build Monte Carlo dataset in chunks for memory efficiency.

        Processes samples in chunks to reduce peak memory usage. Signal
        generation occurs chunk-by-chunk, consuming RNG in a different order
        than non-chunked builds. Results are statistically equivalent but
        not bit-for-bit identical to non-chunked builds with the same seed.

        For exact reproducibility, use file caching: run non-chunked once
        with `force=True`, then reload cached results.

        Uses two-pass approach when adjust_gap=True:
        - Pass 1: Generate signals and compute gap statistics
        - Pass 2: Regenerate signals and run full pipeline with fixed nominal_gap

        Uses single-pass when adjust_gap=False.
        """
        n_chunks = self.n_chunks

        if self.adjust_gap:
            # Two-pass approach for adjust_gap=True
            steps = [
                f"Pass 1: Gap calibration ({n_chunks} chunks)",
                f"Pass 2: Processing ({n_chunks} chunks)",
                "Store results",
            ]
        else:
            steps = [
                f"Processing ({n_chunks} chunks)",
                "Store results",
            ]

        with self.progress(steps) as monitor:
            if self.adjust_gap:
                # Pass 1: Compute gap statistics to determine nominal_gap
                with monitor.step(
                    f"Pass 1: Gap calibration ({n_chunks} chunks)", total=n_chunks
                ) as task:
                    self._calibrate_gap_chunked(task)

                # Reset RNG for second pass
                self.reset_rng()

                # Pass 2: Full processing with fixed nominal_gap
                with monitor.step(
                    f"Pass 2: Processing ({n_chunks} chunks)", total=n_chunks
                ) as task:
                    self._process_chunks(task)
            else:
                # Single pass when adjust_gap=False
                with monitor.step(
                    f"Processing ({n_chunks} chunks)", total=n_chunks
                ) as task:
                    self._process_chunks(task)

            with monitor.step("Store results"):
                self.store()

        return self

    def _calibrate_gap_chunked(self, progress_task=None):
        """Pass 1: Compute gap statistics across all chunks.

        Collects cumulative gap values to compute the 99th percentile
        needed for nominal_gap adjustment.

        NOTE: Uses chunk-by-chunk signal generation for memory efficiency.
        Results are statistically equivalent but not bit-for-bit identical
        to non-chunked builds. For exact reproducibility, use file caching.
        """
        all_cumulative_gaps = []

        for start, end in self.chunk_ranges:
            n_samples = end - start
            signals = self.generate_chunk_signals(n_samples)

            # Compute gap for this chunk
            chunk_gap = self._compute_chunk_gap(signals, n_samples)
            cumulative = chunk_gap.sum(axis=1)  # Sum across coils
            all_cumulative_gaps.append(cumulative)

            if progress_task is not None:
                progress_task.advance()

        # Compute global quantile
        all_cumulative = np.concatenate(all_cumulative_gaps)
        gap_quantile = np.quantile(all_cumulative, 0.99)
        self._calibrated_nominal_gap = self.max_nominal_gap - (
            gap_quantile / self.ncoil - self.max_nominal_gap
        )

    def _compute_chunk_gap(
        self, signals: dict[str, np.ndarray], n_samples: int
    ) -> np.ndarray:
        """Compute gap values for a chunk of signals.

        Returns array of shape (n_samples, ncoil) with total gap per coil.
        """
        radial = signals["radial"]
        tangential = signals["tangential"]

        # Gap computation matching build_gap logic
        gap = np.zeros((n_samples, self.ncoil, 2))

        # Radial component
        gap[..., 0] = np.pi / self.ncoil * radial
        gap[:, :-1, 0] += np.pi / self.ncoil * radial[:, 1:]
        gap[:, -1, 0] += np.pi / self.ncoil * radial[:, 0]

        # Tangential component
        gap[..., 1] = -tangential
        gap[:, :-1, 1] += tangential[:, 1:]
        gap[:, -1, 1] += tangential[:, 0]

        return gap.sum(axis=-1) + self.nominal_gap

    def _process_chunks(self, progress_task=None):
        """Process all chunks and accumulate results.

        NOTE: Uses chunk-by-chunk signal generation for memory efficiency.
        Results are statistically equivalent but not bit-for-bit identical
        to non-chunked builds. For exact reproducibility, use file caching.
        """
        # Save calibrated nominal_gap before initialize_dataset overwrites it
        calibrated_nominal_gap = getattr(
            self, "_calibrated_nominal_gap", self.max_nominal_gap
        )

        # Initialize output arrays
        self.initialize_dataset()

        # Restore calibrated nominal_gap
        self.nominal_gap = calibrated_nominal_gap

        # Pre-allocate result arrays
        peaktopeak = np.zeros(self.samples)
        peaktopeak_offset = np.zeros(self.samples)
        offset = np.zeros((self.samples, 2))
        gap = np.zeros((self.samples, self.ncoil, 2))  # Store gap for plotting

        for chunk_idx, (start, end) in enumerate(self.chunk_ranges):
            n_samples = end - start
            signals = self.generate_chunk_signals(n_samples)

            # Process chunk
            chunk_results = self._process_single_chunk(signals, n_samples)

            # Store results
            peaktopeak[start:end] = chunk_results["peaktopeak"]
            peaktopeak_offset[start:end] = chunk_results["peaktopeak_offset"]
            offset[start:end] = chunk_results["offset"]
            gap[start:end] = chunk_results["gap"]

            if progress_task is not None:
                progress_task.advance()

        # Store final results in dataset
        self.data["peaktopeak"] = "sample", peaktopeak
        self.data["peaktopeak_offset"] = "sample", peaktopeak_offset
        self.data["offset"] = ("sample", "coordinate"), offset
        self.data["gap"] = ("sample", "index", "signal"), gap
        self.data.attrs["nominal_gap"] = self.nominal_gap

    def _process_single_chunk(
        self, signals: dict[str, np.ndarray], n_samples: int
    ) -> dict[str, np.ndarray]:
        """Process a single chunk through the full pipeline.

        Returns dict with peaktopeak, peaktopeak_offset, offset arrays.
        """
        # Compute gap with adjustment for negative values
        gap = self._compute_chunk_gap(signals, n_samples)

        # Adjust tangential to eliminate negative gaps
        tangential = signals["tangential"].copy()
        for _ in range(20):
            sample_index = (gap < -1e-3).any(axis=1)
            if sample_index.sum() == 0:
                break
            offset_gap = gap[sample_index]
            offset_gap[offset_gap >= 0] = 0
            tangential[sample_index] += offset_gap
            # Recompute gap
            gap = np.zeros((n_samples, self.ncoil, 2))
            gap[..., 0] = np.pi / self.ncoil * signals["radial"]
            gap[:, :-1, 0] += np.pi / self.ncoil * signals["radial"][:, 1:]
            gap[:, -1, 0] += np.pi / self.ncoil * signals["radial"][:, 0]
            gap[..., 1] = -tangential
            gap[:, :-1, 1] += tangential[:, 1:]
            gap[:, -1, 1] += tangential[:, 0]
            gap = gap.sum(axis=-1) + self.nominal_gap

        # Structural prediction
        structural = np.zeros((n_samples, self.ncoil, 2))
        if self.energize:
            gap_sum = gap  # Already summed
            roll = signals["roll_length"] - tangential
            yaw = signals["yaw_length"] - tangential
            for i, signal_name in enumerate(["radial", "tangential"]):
                structural[..., i] = self.structural_model.predict(
                    signal_name, gap_sum, roll, yaw
                )

        # Electromagnetic prediction
        electromagnetic = structural.copy()
        electromagnetic[..., 0] += signals["radial"]
        electromagnetic[..., 1] += tangential
        electromagnetic[..., 0] += signals["radial_ccl"]
        electromagnetic[..., 1] += signals["tangential_ccl"]

        self.electromagnetic_model.predict(
            electromagnetic[..., 0], electromagnetic[..., 1]
        )

        # Compute peaktopeak
        if self.wall:
            ndiv = self.electromagnetic_model.fieldline.shape[1]
            wall_hat = np.fft.rfft(signals["radial_wall"])
            firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
            wall_hat[..., 1] += self.electromagnetic_model.axis_offset * (
                self.ncoil // 2
            )
            offset_firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
            deviation = self.electromagnetic_model.fieldline.data - firstwall
            peaktopeak = self.electromagnetic_model.peaktopeak(
                deviation, modes=self.modes
            )
            offset_deviation = (
                self.electromagnetic_model.fieldline.data - offset_firstwall
            )
            peaktopeak_offset = self.electromagnetic_model.peaktopeak(
                offset_deviation, modes=self.modes
            )
        else:
            peaktopeak = self.electromagnetic_model.peaktopeak(modes=self.modes)
            peaktopeak_offset = self.electromagnetic_model.peaktopeak(
                modes=self.modes, axis_offset=True
            )

        axis_offset = self.electromagnetic_model.axis_offset

        # Build gap components array (before nominal_gap addition, for storage)
        # This matches what build_gap() stores in self.data.gap
        gap_components = np.zeros((n_samples, self.ncoil, 2))
        gap_components[..., 0] = np.pi / self.ncoil * signals["radial"]
        gap_components[:, :-1, 0] += np.pi / self.ncoil * signals["radial"][:, 1:]
        gap_components[:, -1, 0] += np.pi / self.ncoil * signals["radial"][:, 0]
        gap_components[..., 1] = -tangential
        gap_components[:, :-1, 1] += tangential[:, 1:]
        gap_components[:, -1, 1] += tangential[:, 0]

        return {
            "peaktopeak": peaktopeak,
            "peaktopeak_offset": peaktopeak_offset,
            "offset": np.column_stack([axis_offset.real, -axis_offset.imag]),
            "gap": gap_components,
        }

    def predict_structure(self):
        """Run structural simulation."""
        self.data["structural"] = (
            ("sample", "index", "signal"),
            np.zeros((self.samples, self.ncoil, self.data.sizes["signal"])),
        )
        if self.energize:
            gap = self.data.gap.sum(axis=-1)
            roll = self.data["roll_length"] - self.data["tangential"]
            yaw = self.data["yaw_length"] - self.data["tangential"]
            for i, signal in enumerate(self.data.signal.values):
                self.data["structural"][..., i] = self.structural_model.predict(
                    signal, gap, roll, yaw
                )

    def predict_electromagnetic(self):
        """Run electromagnetic simulation."""
        self.data["electromagnetic"] = self.data.structural.copy(deep=True)
        self.data.electromagnetic[..., 0] += self.data.radial
        self.data.electromagnetic[..., 1] += self.data.tangential
        self.data.electromagnetic[..., 0] += self.data.radial_ccl
        self.data.electromagnetic[..., 1] += self.data.tangential_ccl
        self.electromagnetic_model.predict(
            self.data.electromagnetic[..., 0], self.data.electromagnetic[..., 1]
        )
        self.data["peaktopeak"] = (
            "sample",
            self.electromagnetic_model.peaktopeak(modes=self.modes),
        )
        self.data["offset"] = (
            ("sample", "coordinate"),
            np.zeros((self.data.sizes["sample"], 2)),
        )
        offset = self.electromagnetic_model.axis_offset
        self.data["offset"][..., 0] = offset.real
        self.data["offset"][..., 1] = -offset.imag
        self.data["peaktopeak_offset"] = (
            "sample",
            self.electromagnetic_model.peaktopeak(modes=self.modes, axis_offset=True),
        )

    def predict_wall(self):
        """Predict combined wall-fieldline deviations."""
        ndiv = self.electromagnetic_model.fieldline.shape[1]
        wall_hat = np.fft.rfft(self.data.radial_wall)
        firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
        wall_hat[..., 1] += self.electromagnetic_model.axis_offset * (self.ncoil // 2)
        offset_firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
        deviation = self.electromagnetic_model.fieldline.data - firstwall.data
        self.data["peaktopeak"] = (
            "sample",
            self.electromagnetic_model.peaktopeak(deviation, modes=self.modes),
        )
        offset_deviation = (
            self.electromagnetic_model.fieldline.data - offset_firstwall.data
        )
        self.data["peaktopeak_offset"] = (
            "sample",
            self.electromagnetic_model.peaktopeak(offset_deviation, modes=self.modes),
        )

    def plot(self, offset=True):
        """Plot peak to peak distribution."""
        self.set_axes()
        self.axes.hist(
            self.data.peaktopeak,
            bins=51,
            density=True,
            rwidth=0.8,
            label="machine axis",
            color="C1",
        )
        if offset:
            self.axes.hist(
                self.data.peaktopeak_offset,
                bins=51,
                density=True,
                rwidth=0.8,
                alpha=0.85,
                color="C2",
                label="magnetic axis",
            )
            self.axes.legend(
                loc="center", bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize="small"
            )
            self.label_quantile(
                self.data.peaktopeak_offset, "H", color="C2", height=0.15
            )
        self.label_quantile(self.data.peaktopeak, "H", color="C1", height=0.04)
        self.axes.set_yticks([])
        self.axes.set_xlabel(r"peak to peak deviation $H$, mm")
        self.axes.set_ylabel(r"$P(H)$")
        self.pdf_text()

    def plot_offset(self):
        """Plot pdf of field line axis offset."""
        offset = np.linalg.norm(self.data.offset, axis=-1)
        self.set_axes()
        self.axes.hist(offset, bins=51, density=True, rwidth=0.8)
        self.axes.set_yticks([])
        self.axes.set_xlabel(r"magnetic axis offset $\zeta$, mm")
        self.axes.set_ylabel(r"$P(\zeta)$")

        self.label_quantile(offset, r"\zeta")
        self.pdf_text()

    def plot_gap(self):
        """Plot gap PDF."""
        self.set_axes()
        self.axes.hist(
            self.gap.flatten(), bins=51, density=True, rwidth=0.8, color="C1"
        )
        self.axes.set_yticks([])
        self.axes.set_xlabel(r"ILIS gap, mm")
        self.axes.set_ylabel(r"$P(gap)$")

        self.label_quantile(self.gap, "gap", precision=1.1)
        lower_quantile = (self.gap <= 0.5).mean()
        self.label_quantile(
            self.gap, "gap", qth=lower_quantile, precision=1.1, height=0.5
        )

        self.text(f"nominal_gap: {self.nominal_gap:1.2f}mm")

    def plot_cumlative_gap(self):
        """Plot cumalitive gap PDF."""
        self.set_axes()
        self.axes.hist(
            self.cumulative_gap, bins=51, density=True, rwidth=0.8, color="C0"
        )
        self.axes.set_yticks([])
        self.axes.set_xlabel(r"cumulative ILIS gap, mm")
        self.axes.set_ylabel(r"$P(\Sigma gap)$")

        self.label_quantile(self.cumulative_gap, r"\Sigma gap", precision=1.1)
        if self.adjust_gap is False:
            qth = (self.cumulative_gap < self.max_nominal_gap * self.ncoil).mean()
            self.label_quantile(
                self.cumulative_gap, r"\Sigma gap", precision=1.1, qth=qth, height=1
            )

        self.text(f"nominal_gap: {self.nominal_gap:1.2f}mm")

    def plot_sample(self, quantile=0.99, offset=True, plot_deviation=False):
        """Plot waveforms from single sample."""
        sample = self.sample(quantile, offset)
        self.set_axes(
            nrows=3,
            ncols=1,
            sharex=False,
            sharey=False,
            gridspec_kw=dict(height_ratios=[1, 1, 2]),
        )
        width = 0.8

        signal_width = width / self.data.sizes["component"]
        for i, component in enumerate(self.data.component.values):
            signal = self.data[component]
            bar_offset = (i + 0.5) * signal_width - width / 2
            self.axes[0].bar(
                self.data.index + bar_offset,
                signal[sample],
                color=f"C{i + 1}",
                width=signal_width,
                label=component,
            )
            # self.axes[0].plot(self.data.index,
            #             self.theta[0] * (-1)**i *
            #             np.ones_like(self.data.index), 'C7--', alpha=0.5,
            #             lw=1.5)
        self.axes[0].set_ylabel("vault")
        self.axes[0].legend(fontsize="xx-small", bbox_to_anchor=(1, 1))
        self.axes[0].set_xticks([])

        # signal_width = width / 3
        # for i, signal in ['gap', 'roll', 'yaw']
        self.axes[1].bar(
            self.data.index,
            self.data.gap[sample].sum(axis=-1) + self.data.nominal_gap,
            width=width,
            color="C0",
        )
        self.axes[1].set_ylabel("gap")
        self.axes[1].set_xticks([])

        fieldline = self.electromagnetic_model.predict(
            self.data.electromagnetic[sample, :, 0],
            self.data.electromagnetic[sample, :, 1],
        )[0]
        self.axes[2].plot(fieldline.phi, fieldline, "C6", label="fieldline")

        ndiv = len(fieldline)
        wall_hat = np.fft.rfft(self.data.radial_wall[sample, :])
        firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
        wall_hat[1] += self.electromagnetic_model.axis_offset[0] * (self.ncoil // 2)
        offset_firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
        self.axes[2].plot(fieldline.phi, firstwall, "-.", color="gray", label="wall")
        self.axes[2].plot(
            fieldline.phi, offset_firstwall, "-", color="gray", label="offset wall"
        )

        if plot_deviation:
            longwave = np.fft.irfft(
                np.fft.rfft(fieldline - firstwall)[: self.modes + 1], ndiv
            )
            offset_longwave = np.fft.irfft(
                np.fft.rfft(fieldline - offset_firstwall)[: self.modes + 1], ndiv
            )
            peaktopeak = self.electromagnetic_model.peaktopeak(longwave)
            offset_peaktopeak = self.electromagnetic_model.peaktopeak(offset_longwave)
            self.axes[2].plot(
                fieldline.phi, longwave, "-.C0", label=rf"$H_{{LW}}={peaktopeak:1.1f}$"
            )
            self.axes[2].plot(
                fieldline.phi,
                offset_longwave,
                "-C0",
                label=rf"offset $H_{{LW}}={offset_peaktopeak:1.1f}$",
            )

        self.axes[2].legend(fontsize="xx-small", bbox_to_anchor=(1, 1))
        self.axes[2].set_ylabel("deviation")
        self.axes[2].set_xlabel(r"$\phi$")
        self.plt.suptitle(f"quantile={quantile} offset={offset}")


@dataclass
class ErrorField(Trial, Plot1D):
    """Run Monte Carlo error field trials."""

    filename: str = "errorfield_trial"
    component: list[str] = field(
        default_factory=lambda: [
            "radial",
            "tangential",
            "vertical",
            "radial_ccl",
            "tangential_ccl",
            "vertical_ccl",
            "pitch_length",
            "roll_length",
            "yaw_length",
        ]
    )
    theta: list[float] = field(default_factory=lambda: [5, 5, 5, 2, 2, 2, 5, 10, 10])
    pdf: list[str] = field(
        default_factory=lambda: [
            "uniform",
            "uniform",
            "uniform",
            "normal",
            "normal",
            "normal",
            "uniform",
            "uniform",
            "uniform",
        ]
    )

    def __post_init__(self):
        """Initialize model instances."""
        self.model = overlap.Model()
        super().__post_init__()

    def build(self):
        """Build Monte Carlo dataset."""
        if self.samples > self.chunk_size:
            return self.build_chunked()
        return self.build_full()

    def build_full(self):
        """Build Monte Carlo dataset (non-chunked, all in memory)."""
        steps = ["Signal generation", "Overlap prediction", "Store results"]

        with self.progress(steps) as monitor:
            with monitor.step("Signal generation"):
                self.build_signal()
            with monitor.step("Overlap prediction"):
                self.predict()
            with monitor.step("Store results"):
                self.store()
        return self

    def build_chunked(self):
        """Build Monte Carlo dataset in chunks for memory efficiency."""
        n_chunks = self.n_chunks
        step_label = f"Processing ({n_chunks} chunks)"
        steps = [step_label, "Store results"]

        with self.progress(steps) as monitor:
            with monitor.step(step_label, total=n_chunks) as task:
                self._process_chunks_errorfield(task)
            with monitor.step("Store results"):
                self.store()
        return self

    def _process_chunks_errorfield(self, progress_task=None):
        """Process all chunks for error field calculation.

        NOTE: Uses chunk-by-chunk signal generation for memory efficiency.
        Results are statistically equivalent but not bit-for-bit identical
        to non-chunked builds. For exact reproducibility, use file caching.
        """
        # Initialize dataset
        self.initialize_dataset()
        self.data["plasma"] = self.model.data.plasma
        n_plasma = self.data.sizes["plasma"]

        # Pre-allocate result array
        overlap_results = np.zeros((self.samples, n_plasma))

        for start, end in self.chunk_ranges:
            n_samples = end - start
            signals = self.generate_chunk_signals(n_samples)

            # Compute overlap for this chunk
            radial = signals["radial"] + signals["radial_ccl"]
            tangential = signals["tangential"] + signals["tangential_ccl"]
            vertical = signals["vertical"] + signals["vertical_ccl"]
            pitch = signals["pitch_length"] / (1e3 * WedgeGap.length["pitch"])
            roll = signals["roll_length"] / (1e3 * WedgeGap.length["roll"])
            yaw = signals["yaw_length"] / (1e3 * WedgeGap.length["yaw"])

            for i, plasma in enumerate(self.data.plasma.values):
                overlap_results[start:end, i] = self.model.predict(
                    plasma, radial, tangential, vertical, pitch, roll, yaw
                )

            if progress_task is not None:
                progress_task.advance()

        self.data["overlap"] = ("sample", "plasma"), overlap_results

    def predict(self):
        """Predict overlap error field."""
        self.data["plasma"] = self.model.data.plasma
        self.data["overlap"] = (
            ("sample", "plasma"),
            np.zeros((self.samples, self.data.sizes["plasma"])),
        )
        radial = self.data.radial + self.data.radial_ccl
        tangential = self.data.tangential + self.data.tangential_ccl
        vertical = self.data.vertical + self.data.vertical_ccl
        pitch = self.data.pitch_length / (1e3 * WedgeGap.length["pitch"])
        roll = self.data.roll_length / (1e3 * WedgeGap.length["roll"])
        yaw = self.data.yaw_length / (1e3 * WedgeGap.length["yaw"])
        for i, plasma in enumerate(self.data.plasma.values):
            self.data.overlap[:, i] = self.model.predict(
                plasma, radial, tangential, vertical, pitch, roll, yaw
            )

    def plot(self):
        """Plot overlap errorfield PDFs."""
        self.set_axes()
        self.axes.hist(
            self.data.overlap,
            bins=51,
            density=True,
            rwidth=0.9,
            label=[f"plasma {i}" for i in self.data.plasma.values],
        )
        self.axes.legend(ncol=1, bbox_to_anchor=(0.27, 1), fontsize="x-small")
        self.axes.set_yticks([])
        self.axes.set_xlabel(r"Overlap error field $B/B_{limit}$")
        self.axes.set_ylabel(r"$P(B/B_{limit})$")
        self.pdf_text()

        quantile_index = np.argmax(np.quantile(self.data.overlap, 0.99, axis=0))
        self.label_quantile(
            self.data.overlap[:, quantile_index],
            r"B/B_{limit}",
            color=f"C{quantile_index}",
        )

    def scan(self, quantile=0.99):
        """Run sensitivity scan."""
        if (
            "quantile_scan" in self.data
            and self.data.attrs.get("quantile", None) == quantile
        ):
            return self
        self.data["quantile_scan"] = (
            ("component", "plasma"),
            np.ones((self.data.sizes["component"], self.data.sizes["plasma"])),
        )
        for i, pdf in enumerate(self.pdf):
            theta = list(np.zeros(len(self.pdf)))
            theta[i] = self.data.theta.values[i]
            error = ErrorField(
                self.samples, component=self.component, theta=theta, pdf=self.pdf
            )
            self.data["quantile_scan"][i] = np.quantile(
                error.data.overlap, quantile, axis=0
            )
        self.data.attrs["quantile"] = quantile
        return self.store()

    def plot_scan(self, quantile=0.99):
        """Plot sensitivity scan results."""
        self.scan(quantile)
        component = [
            component.replace("_", " ") for component in self.data.component.values
        ]
        self.set_axes()
        for i, plasma in enumerate(self.data.plasma.values):
            self.axes.bar(
                component,
                self.data.quantile_scan[:, i],
                width=0.8 - i * 0.2,
                label=f"plasma {plasma}",
            )
            self.axes.set_xticks(rotation=90)
        self.axes.legend(fontsize="x-small")
        self.axes.set_ylabel(r"Overlap error field $B/B_{limit}$")


if __name__ == "__main__":
    trial_name = "baseline_2021"
    # trial_name = "S4_refine_pit_2026"
    trial_name = "S4_hybrid_pit_2026"
    samples = 200_000
    force = False

    # Load baseline_2021 from manifest (should use cache)
    vault = Vault.from_manifest(trial_name, samples=samples, force=force)
    print(f"Vault hash: {vault.group_name}")

    vault.plot()
    vault.plot_offset()
    vault.plot_gap()
    vault.plot_cumlative_gap()

    error = ErrorField.from_manifest(trial_name, samples=samples, force=force)
    print(f"ErrorField hash: {error.group_name}")

    error.plot()

    # trial.plot_offset()

    # case -> 1.7/0.3, 2.1/0.8
    # roll -> 0.2/0.1
    # yaw -> 0.2/0.2
    # ccl -> 1.4/0.9, 1.7/0.8
    # wall -> 3.2 / 3.2

    # trial.plot_sample(0.99, False)
    # trial.plot_sample(0.99, True)
