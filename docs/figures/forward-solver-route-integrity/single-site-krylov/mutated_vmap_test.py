"""Run the single-site test file with the fixed-capacity scan stream restored."""

import sys

sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
import pytest
import scan_stream

scan_stream.install()
raise SystemExit(
    pytest.main(["-p", "no:cacheprovider", "-vv", "tests/test_single_site_krylov.py"])
)
