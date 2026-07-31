"""Run the reproducible MAST catalog geometry census."""

from nova.catalog.mast_geometry import (
    DEFAULT_LEVEL1_ROOT,
    DEFAULT_LEVEL2_ROOT,
    active_component_geometry,
    canonical_cycle,
    main,
    observed_ranges,
    passive_component_geometry,
    physical_digest,
    physical_snapshot,
    scan_catalog,
    source_fingerprint,
)

__all__ = [
    "DEFAULT_LEVEL1_ROOT",
    "DEFAULT_LEVEL2_ROOT",
    "active_component_geometry",
    "canonical_cycle",
    "main",
    "observed_ranges",
    "passive_component_geometry",
    "physical_digest",
    "physical_snapshot",
    "scan_catalog",
    "source_fingerprint",
]


if __name__ == "__main__":
    main()
