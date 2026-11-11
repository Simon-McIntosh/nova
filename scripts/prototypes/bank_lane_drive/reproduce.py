"""Reproduce shared rejection using the archived scalar continuation driver."""

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile

from tests import test_bank_lane_drive as toy

BASE = "eb78afe5355009db83dfbc5750cc5081f37859d7"
source = subprocess.check_output(
    ["git", "show", BASE + ":nova/equilibrium/drive_continuation.py"]
)
with tempfile.TemporaryDirectory(
    prefix="shared-drive-reproduction-", dir=os.environ["TMPDIR"]
) as directory:
    path = Path(directory) / "drive_continuation.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("shared_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    toy.driver = module
    result, attempts = toy.run()
    first, following = result.substeps[:2]
    print("# revision:", BASE, "module:", module.__file__, "cwd:", Path.cwd())
    print("# source sha256:", hashlib.sha256(source).hexdigest())
    assert attempts[0][2][0] <= first.tolerance
    assert attempts[0][2][1] > first.tolerance
    assert not first.converged
    assert following.sub_step == first.sub_step * module.SHRINK
    print(
        "REPRODUCTION easy residual",
        attempts[0][2][0],
        "stiff residual",
        attempts[0][2][1],
        "whole batch rejected",
        not first.converged,
        "shared sub-step",
        first.sub_step,
        "->",
        following.sub_step,
        "sub-solves",
        len(result.substeps),
    )
