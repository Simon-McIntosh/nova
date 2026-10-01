"""Run the single-site test file with the carry-selecting exit loop restored."""

import sys

sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
import pytest
import selected_carry_stream

selected_carry_stream.install()
raise SystemExit(
    pytest.main(["-p", "no:cacheprovider", "-vv", "tests/test_single_site_krylov.py"])
)
