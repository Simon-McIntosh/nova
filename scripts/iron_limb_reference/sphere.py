"""Three-mesh permeability benchmark against a uniformly driven sphere."""

import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import ConvexHull

from scripts.iron_limb_reference.surface import MU0, Panels, SurfaceModel


def sphere_panels(level):
    golden = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [(0, a, b * golden) for a in (-1, 1) for b in (-1, 1)]
        + [(a, b * golden, 0) for a in (-1, 1) for b in (-1, 1)]
        + [(b * golden, 0, a) for a in (-1, 1) for b in (-1, 1)]
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    triangles = vertices[ConvexHull(vertices).simplices]
    for _ in range(level):
        a, b, c = (triangles[:, k] for k in range(3))
        ab, bc, ca = (
            v / np.linalg.norm(v, axis=1)[:, None] for v in (a + b, b + c, c + a)
        )
        triangles = np.concatenate(
            (
                np.stack((a, ab, ca), axis=1),
                np.stack((ab, b, bc), axis=1),
                np.stack((ca, bc, c), axis=1),
                np.stack((ab, bc, ca), axis=1),
            )
        )
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    reverse = np.sum(normals * triangles.mean(axis=1), axis=1) < 0
    triangles[reverse] = triangles[reverse, ::-1]
    return Panels(triangles)


def sphere_study(levels=(4, 5, 6), output=None):
    """Report interior-field error and observed order without extrapolation."""
    from scripts.iron_limb_reference import surface

    record = dict(
        solver_sha256=hashlib.sha256(Path(surface.__file__).read_bytes()).hexdigest(),
        levels=list(levels),
        rows=[],
        convergence=[],
        passed=False,
    )
    for level in levels:
        started = time.perf_counter()
        panels = sphere_panels(level)
        print(f"SPHERE_BUILD panels={len(panels.vertices)}", flush=True)
        model = SurfaceModel(
            panels,
            lambda p: np.tile([0.0, 0.0, MU0], (len(p), 1)),
            rotations=1,
            reflect=False,
        )
        count = len(panels.vertices)
        for mu in [50.0, 200.0]:
            charge, residual = model.solve(mu)
            h = model.iron_field(np.zeros((1, 3)), charge)[0] / MU0 + [0.0, 0.0, 1.0]
            exact = 3 / (mu + 2)
            error = float(np.linalg.norm(h - [0.0, 0.0, exact]) / exact)
            row = dict(
                level=level,
                panels=count,
                h=np.sqrt(4 * np.pi / count),
                mu=mu,
                H_over_H0=h.tolist(),
                exact_H_over_H0=exact,
                relative_error=error,
                residual=residual,
                solid_angle_error=float(np.max(np.abs(model.conservation_error))),
                neutrality=model.last_solve,
                elapsed_seconds=time.perf_counter() - started,
            )
            record["rows"].append(row)
            print("SPHERE_ROW " + json.dumps(row), flush=True)
            if output is not None:
                Path(output).write_text(json.dumps(record, indent=2) + "\n")
        del model, panels
        gc.collect()
    for mu in [50.0, 200.0]:
        rows = [r for r in record["rows"] if r["mu"] == mu]
        errors = np.array([r["relative_error"] for r in rows])
        sizes = np.array([r["h"] for r in rows])
        orders = np.log(errors[:-1] / errors[1:]) / np.log(sizes[:-1] / sizes[1:])
        record["convergence"].append(
            dict(
                mu=mu,
                observed_orders=orders.tolist(),
                errors=errors.tolist(),
                panels=[r["panels"] for r in rows],
                passed=bool(np.all(np.diff(errors) < 0) and errors[-1] < 0.01),
            )
        )
    record["passed"] = all(c["passed"] for c in record["convergence"])
    print("SPHERE_CONVERGENCE " + json.dumps(record["convergence"]), flush=True)
    if output is not None:
        Path(output).write_text(json.dumps(record, indent=2) + "\n")
    return record
