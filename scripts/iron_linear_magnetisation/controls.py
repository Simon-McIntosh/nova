"""Shared closed-shell geometry, source and analytic shielding control."""

import hashlib
from pathlib import Path
import subprocess

import numpy as np
import shapely

from nova.biot.greens import greens_bz_br

SHELL_MU = 1000.0
SHELL_INNER_RADIUS = 0.9
SHELL_OUTER_RADIUS = 1.0
LOOP_RADIUS = 0.05
SHELL_PROBES = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
UNIFORM_SIZES = (0.04, 0.02, 0.01)


def require_clean_script(path, expected_blob, expected_revision=None):
    """Refuse tracked edits and verify the executed script's committed bytes."""
    path = Path(path).resolve()
    tree = path.parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()
    if expected_revision is not None and revision != expected_revision:
        raise SystemExit("executed revision differs from its stamp")
    dirty = subprocess.check_output(
        ["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty:
        raise SystemExit("refusing to measure a dirty tree")
    data = path.read_bytes()
    actual = hashlib.sha1(
        b"blob %d\0" % len(data) + data, usedforsecurity=False
    ).hexdigest()
    if actual != expected_blob:
        raise SystemExit("executed script differs from its committed blob")
    return actual


def shell():
    """Return the polygonal half section of the 0.9–1 m spherical shell."""
    return (
        shapely.Point(0, 0)
        .buffer(SHELL_OUTER_RADIUS, 512)
        .difference(shapely.Point(0, 0).buffer(SHELL_INNER_RADIUS, 512))
        .intersection(
            shapely.box(0, -SHELL_OUTER_RADIUS, SHELL_OUTER_RADIUS, SHELL_OUTER_RADIUS)
        )
    )


def loop(points):
    """Return the field of a one-ampere loop of radius 50 mm."""
    bz, br = greens_bz_br(points[:, 0], points[:, 1], LOOP_RADIUS, 0.0)
    return np.column_stack((br, bz))


def exact_shielding(ratio=SHELL_INNER_RADIUS / SHELL_OUTER_RADIUS, mu=SHELL_MU):
    """Dipole shielding factor for a linear spherical shell.

    ``ratio`` is the shell's inner over outer radius and ``mu`` its relative
    permeability; the defaults are this control's 0.9-1 m shell at mu_r = 1000.
    The factor is psi without iron over psi with it, so it is the reduction the
    shell causes.
    """
    return ((2 * mu + 1) * (mu + 2) - 2 * (mu - 1) ** 2 * ratio**3) / (9 * mu)
