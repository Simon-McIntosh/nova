"""Run the single-site test with the per-site base body rebound in place."""

import sys

import pytest

sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
import per_site

per_site.install()
raise SystemExit(
    pytest.main(
        [
            "-p",
            "no:cacheprovider",
            "tests/test_single_site_krylov.py",
            "-vv",
            "--tb=short",
        ]
    )
)
