"""Conditional quartile analysis for Monte Carlo trial data."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

import numpy as np
import xarray

from nova.graphics.plot import Plot1D

if TYPE_CHECKING:
    from typing import Self


class TrialProtocol(Protocol):
    """Protocol for Trial-like classes that support quartile analysis."""

    data: xarray.Dataset

    def store(self) -> "Self":
        """Store data to file."""
        ...


@dataclass
class QuartileAnalysis(Plot1D):
    """Conditional quartile analysis for Monte Carlo simulations.

    Mixin class that adds conditional quartile analysis to Trial classes.
    Splits samples into bins based on input parameter values and computes
    quantiles of output metrics for each bin. Results are cached in the
    netCDF file alongside statistical data.

    Subclasses must define:
        - quartile_metrics: dict mapping metric names to data accessors
        - quartile_components: property returning list of component names
    """

    n_bins: ClassVar[int] = 10
    quartile_level: ClassVar[float] = 0.99

    # Override in subclasses via field or property
    quartile_metrics: ClassVar[dict[str, str]] = {}

    @property
    def quartile_components(self) -> list[str]:
        """Return components for quartile analysis. Override in subclass."""
        return []

    def _get_metric_data(self, metric: str) -> np.ndarray:
        """Return data array for a metric name.

        Override in subclasses for custom metric access patterns.
        """
        return self.data[metric].values

    def _get_component_data(self, component: str) -> np.ndarray:
        """Return flattened component data for binning.

        For 2D arrays (sample, index), returns values summed or averaged
        across the index dimension as appropriate for the analysis.
        """
        data = self.data[component].values
        if data.ndim == 2:
            # Use mean across coils for binning
            return data.mean(axis=1)
        return data

    def build_quartile_analysis(self) -> xarray.Dataset:
        """Build conditional quartile analysis dataset.

        For each input component, bins samples by component value and
        computes the specified quantile of each output metric per bin.

        Returns
        -------
        xarray.Dataset
            Dataset with dimensions (component, bin) containing:
            - bin_edges: bin boundaries for each component
            - bin_centers: bin center values for each component
            - For each metric: quantile values per bin
        """
        components = self.quartile_components or list(self.data.component.values)
        metrics = self.quartile_metrics

        n_bins = self.n_bins
        q = self.quartile_level

        # Initialize result arrays
        bin_edges = np.zeros((len(components), n_bins + 1))
        bin_centers = np.zeros((len(components), n_bins))
        metric_results = {name: np.zeros((len(components), n_bins)) for name in metrics}

        for i, component in enumerate(components):
            # Get component data for binning
            comp_data = self._get_component_data(component)

            # Create bins based on component distribution
            edges = np.percentile(comp_data, np.linspace(0, 100, n_bins + 1))
            # Ensure unique edges by adding small jitter if needed
            if len(np.unique(edges)) < len(edges):
                edges = np.linspace(comp_data.min(), comp_data.max(), n_bins + 1)

            bin_edges[i] = edges
            bin_centers[i] = (edges[:-1] + edges[1:]) / 2

            # Assign samples to bins
            bin_indices = np.digitize(comp_data, edges[:-1]) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            # Compute quantile for each metric in each bin
            for metric_name, metric_accessor in metrics.items():
                metric_data = self._get_metric_data(metric_accessor)

                for j in range(n_bins):
                    mask = bin_indices == j
                    if mask.sum() > 0:
                        metric_results[metric_name][i, j] = np.quantile(
                            metric_data[mask], q
                        )
                    else:
                        metric_results[metric_name][i, j] = np.nan

        # Build xarray dataset
        qa_data = xarray.Dataset(
            attrs={
                "quartile_level": q,
                "n_bins": n_bins,
            }
        )
        qa_data["qa_component"] = components
        qa_data["qa_bin"] = range(n_bins)
        qa_data["qa_bin_edges"] = (
            ("qa_component", "qa_bin_edge"),
            bin_edges,
        )
        qa_data["qa_bin_edge"] = range(n_bins + 1)
        qa_data["qa_bin_centers"] = (
            ("qa_component", "qa_bin"),
            bin_centers,
        )

        for metric_name, values in metric_results.items():
            qa_data[f"qa_{metric_name}"] = (
                ("qa_component", "qa_bin"),
                values,
            )

        return qa_data

    def load_quartile_analysis(self) -> bool:
        """Load cached quartile analysis if available.

        Returns
        -------
        bool
            True if cached data was loaded, False otherwise.
        """
        # Check if quartile data exists in dataset
        qa_vars = [v for v in self.data.data_vars if v.startswith("qa_")]
        return len(qa_vars) > 0

    def ensure_quartile_analysis(self, force: bool = False) -> "QuartileAnalysis":
        """Ensure quartile analysis is available, building if needed.

        Uses load/build pattern: attempts to load from cache first,
        builds and stores if not found or if force=True.

        Parameters
        ----------
        force : bool
            Force rebuild even if cached data exists.

        Returns
        -------
        QuartileAnalysis
            Self for method chaining.
        """
        if not force and self.load_quartile_analysis():
            return self

        # Build and merge into main dataset
        qa_data = self.build_quartile_analysis()
        self.data = self.data.merge(qa_data, compat="override")

        # Store updated dataset (self.store() is provided by Trial mixin)
        if hasattr(self, "store"):
            self.store()  # type: ignore[attr-defined]
        return self

    def plot_quartile(
        self,
        metric: str | None = None,
        figsize: tuple[float, float] = (12, 8),
    ):
        """Plot conditional quantile variation across input parameters.

        Shows how the specified quantile of output metrics varies with
        each input parameter, revealing sensitivity and correlations.

        Parameters
        ----------
        metric : str | None
            Metric to plot. If None, plots first available metric.
        figsize : tuple[float, float]
            Figure size (width, height) in inches.
        """
        self.ensure_quartile_analysis()

        # Get available metrics
        qa_metrics = [
            v.replace("qa_", "")
            for v in self.data.data_vars
            if v.startswith("qa_")
            and v not in ["qa_bin_edges", "qa_bin_centers", "qa_component", "qa_bin"]
        ]

        if not qa_metrics:
            raise ValueError(
                "No quartile analysis data found. Run ensure_quartile_analysis first."
            )

        if metric is None:
            metric = qa_metrics[0]
        elif metric not in qa_metrics:
            raise ValueError(f"Metric '{metric}' not found. Available: {qa_metrics}")

        # metric is guaranteed to be str at this point
        assert isinstance(metric, str)

        components = self.data.qa_component.values
        n_components = len(components)

        # Determine grid layout
        ncols = min(4, n_components)
        nrows = (n_components + ncols - 1) // ncols

        self.set_axes(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = np.atleast_1d(self.axes).flatten()

        metric_data = self.data[f"qa_{metric}"].values
        bin_centers = self.data.qa_bin_centers.values

        # Get global quantile for reference line
        metric_key = self.quartile_metrics.get(metric, metric)
        if metric_key is None:
            metric_key = metric
        global_quantile = np.quantile(
            self._get_metric_data(metric_key),
            self.quartile_level,
        )

        for i, component in enumerate(components):
            ax = axes[i]
            ax.plot(
                bin_centers[i],
                metric_data[i],
                "o-",
                color=f"C{i % 10}",
                markersize=4,
                linewidth=1.5,
            )
            ax.axhline(
                global_quantile,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
            ax.set_xlabel(component.replace("_", " "), fontsize="small")
            if i % ncols == 0:
                ax.set_ylabel(f"q({self.quartile_level}) {metric}", fontsize="small")
            ax.tick_params(labelsize="x-small")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        metric_label = self._get_metric_label(metric)
        title = f"Conditional q({self.quartile_level}) of {metric_label}"
        fig = self.fig
        if fig is not None:
            fig.suptitle(f"{title} vs Input Parameters", fontsize="medium")
            fig.tight_layout()

    def plot_quartile_summary(self, figsize: tuple[float, float] = (10, 6)):
        """Plot summary of all metrics' conditional quantile ranges.

        Shows the range of conditional quantiles for each input parameter,
        highlighting which inputs have the strongest influence on outputs.

        Parameters
        ----------
        figsize : tuple[float, float]
            Figure size (width, height) in inches.
        """
        self.ensure_quartile_analysis()

        # Get available metrics
        qa_metrics = [
            v.replace("qa_", "")
            for v in self.data.data_vars
            if v.startswith("qa_")
            and v not in ["qa_bin_edges", "qa_bin_centers", "qa_component", "qa_bin"]
        ]

        if not qa_metrics:
            raise ValueError("No quartile analysis data found.")

        components = self.data.qa_component.values
        n_components = len(components)
        n_metrics = len(qa_metrics)

        self.set_axes(nrows=n_metrics, ncols=1, figsize=figsize, sharex=True)
        axes = np.atleast_1d(self.axes).flatten()

        x = np.arange(n_components)
        width = 0.6

        for i, metric in enumerate(qa_metrics):
            ax = axes[i]
            metric_data = self.data[f"qa_{metric}"].values

            # Compute range (max - min) across bins for each component
            ranges = np.ptp(metric_data, axis=1)

            # Get global quantile for normalization
            metric_key = self.quartile_metrics.get(metric, metric)
            if metric_key is None:
                metric_key = metric
            global_q = np.quantile(
                self._get_metric_data(metric_key),
                self.quartile_level,
            )

            # Normalize ranges by global quantile for comparison
            normalized_ranges = ranges / global_q * 100 if global_q > 0 else ranges

            colors = [f"C{j % 10}" for j in range(n_components)]
            ax.bar(x, normalized_ranges, width, color=colors)

            metric_label = self._get_metric_label(metric)
            ax.set_ylabel(f"{metric_label}\n(% range)", fontsize="small")
            ax.tick_params(labelsize="x-small")

            if i == n_metrics - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [c.replace("_", "\n") for c in components],
                    rotation=45,
                    ha="right",
                    fontsize="x-small",
                )

        title = f"Sensitivity: Conditional q({self.quartile_level}) Range"
        fig = self.fig
        if fig is not None:
            fig.suptitle(f"{title} by Input Parameter", fontsize="medium")
            fig.tight_layout()

    def _get_metric_label(self, metric: str) -> str:
        """Return display label for metric name."""
        labels = {
            "peaktopeak": r"$H$ (mm)",
            "peaktopeak_offset": r"$H_{\zeta}$ (mm)",
            "cumulative_gap": r"$\Sigma$ gap (mm)",
            "axis_offset": r"$\zeta$ (mm)",
            "overlap": r"$B/B_{limit}$",
        }
        return labels.get(metric, metric)
